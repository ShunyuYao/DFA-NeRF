# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import namedtuple
from pathlib import Path
from core.inference import predToKeypoints, gaussian_modulation_torch, get_final_preds
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from math import sqrt

import numbers
import torch.nn.functional as F


def decode_center_preds(config, output_eye_hm,
                        output_eye_offset,
                        output_eye_regress,
                        output_eye_densewh,
                        heatmap_stride=4):

    pred_eye_hm = output_eye_hm.data.cpu().numpy()
    pred_eye_offset = output_eye_offset.data.cpu().numpy()
    pred_eye_reg = output_eye_regress.data.cpu().numpy()
    pred_eye_densewh = output_eye_densewh.data.cpu().numpy()

    eye_pos, maxvals = predToKeypoints(pred_eye_hm, False)
    eye_l_pos = np.expand_dims(eye_pos[:, 0, :], 1)
    eye_r_pos = np.expand_dims(eye_pos[:, 1, :], 1)

    idx = eye_pos[..., 0] + (eye_pos[..., 1] - 1) * config.MODEL.HEATMAP_SIZE[1]
    # pred_eye_hm = pred_eye_hm.reshape(pred_eye_hm.shape[0], pred_eye_hm.shape[1], -1)
    pred_eye_densewh = pred_eye_densewh.reshape(pred_eye_densewh.shape[0], pred_eye_densewh.shape[1], -1)
    pred_eye_offset = pred_eye_offset.reshape(pred_eye_offset.shape[0], pred_eye_offset.shape[1], -1)
    pred_eye_reg = pred_eye_reg.reshape(pred_eye_reg.shape[0], pred_eye_reg.shape[1], -1)

    idx_eye_l = idx[..., 0].astype(np.int32).reshape(-1, 1, 1)
    idx_eye_r = idx[..., 1].astype(np.int32).reshape(-1, 1, 1)

    pred_eye_l_offset = pred_eye_offset[:, :2]
    pred_eye_r_offset = pred_eye_offset[:, 2:]
    pred_eye_l_reg = pred_eye_reg[:, :config.MODEL.NUM_EYE_JOINTS*2]
    pred_eye_r_reg = pred_eye_reg[:, config.MODEL.NUM_EYE_JOINTS*2:]
    pred_eye_l_densewh = pred_eye_densewh[:, :2]
    pred_eye_r_densewh = pred_eye_densewh[:, 2:]

    pred_eye_l = np.take_along_axis(pred_eye_l_reg, np.tile(idx_eye_l, (1, pred_eye_l_reg.shape[1], 1)), 2)  # * config.MODEL.IMAGE_SIZE[0]
    pred_eye_r = np.take_along_axis(pred_eye_r_reg, np.tile(idx_eye_r, (1, pred_eye_r_reg.shape[1], 1)), 2)  # * config.MODEL.IMAGE_SIZE[0]
    pred_eye_l = pred_eye_l.reshape(pred_eye_l.shape[0], -1, 2)
    pred_eye_r = pred_eye_r.reshape(pred_eye_r.shape[0], -1, 2)

    pred_eye_l_offset = np.take_along_axis(pred_eye_l_offset, np.tile(idx_eye_l, (1, pred_eye_l_offset.shape[1], 1)), 2)
    pred_eye_r_offset = np.take_along_axis(pred_eye_r_offset, np.tile(idx_eye_r, (1, pred_eye_r_offset.shape[1], 1)), 2)
    pred_eye_l_offset = pred_eye_l_offset.reshape(pred_eye_l_offset.shape[0], -1, 2)
    pred_eye_r_offset = pred_eye_r_offset.reshape(pred_eye_r_offset.shape[0], -1, 2)

    pred_eye_l_densewh = np.take_along_axis(pred_eye_l_densewh, np.tile(idx_eye_l, (1, pred_eye_l_densewh.shape[1], 1)), 2)
    pred_eye_r_densewh = np.take_along_axis(pred_eye_r_densewh, np.tile(idx_eye_r, (1, pred_eye_r_densewh.shape[1], 1)), 2)
    pred_eye_l += (pred_eye_l_offset + eye_l_pos) * heatmap_stride + pred_eye_l * config.MODEL.IMAGE_SIZE[1]
    pred_eye_r += (pred_eye_r_offset + eye_r_pos) * heatmap_stride + pred_eye_r * config.MODEL.IMAGE_SIZE[1]
    pred_eye_l_center = (pred_eye_l_offset + eye_l_pos) * heatmap_stride
    pred_eye_r_center = (pred_eye_r_offset + eye_r_pos) * heatmap_stride
    preds_eye_l_wh = pred_eye_l_densewh * config.MODEL.IMAGE_SIZE[1] / 4
    preds_eye_r_wh = pred_eye_r_densewh * config.MODEL.IMAGE_SIZE[1] / 4

    preds_rg = np.concatenate([pred_eye_l, pred_eye_r], 1)
    preds_center = np.concatenate([pred_eye_l_center, pred_eye_r_center], 1)
    preds_wh = np.concatenate([preds_eye_l_wh, preds_eye_r_wh], 1)
    preds_wh = preds_wh.reshape(preds_wh.shape[0], 2, 2)
    preds_bbox = np.concatenate([preds_center - preds_wh / 2, preds_center + preds_wh / 2], 2)

    return preds_rg, preds_bbox

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model, list_params=False):
    optimizer = None
    if list_params:
        model_params = model
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                model_params,
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                model_params,
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(
                model_params,
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WD,
                momentum=cfg.TRAIN.MOMENTUM,
            )
    else:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'adamW':
            assert torch.__version__ >= '1.4.0', 'pytorch version must bigger than 1.4.0 to support adamW optimizer'
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                betas=(cfg.TRAIN.BETA1, cfg.TRAIN.BETA2),
                weight_decay=cfg.TRAIN.WD,
            )
        elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
            )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])
    gflops = flops_sum/(1024**3)

    return details, gflops


def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    maxval, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), rot, True)

    return preds, maxval


def save_landmarks(image, heatmap, save_path, filename, config, meta,
                   down_scale_ratio=2, rgb2bgr=False):
    """Show image with pred_landmarks"""
    # if config.MODEL.HEATMAP_DM:
    #     output = gaussian_modulation_torch(heatmap, config.MODEL.FACE_SIGMA)
    # preds, pred_landmarks, maxvals = get_final_preds(
    #     config, output.detach().cpu().numpy(), meta)
    pred_landmarks, maxvals = predToKeypoints(heatmap)
    pred_landmarks = pred_landmarks * down_scale_ratio
    # print("pred_landmarks: ", pred_landmarks)
    # print("maxvals: ", maxvals)

    image_save = image.copy()
    image_save = image_save.astype(np.uint8)
    image_save = image_save.squeeze()
    # print("image_save shape: ", image_save.shape)
    if rgb2bgr:
        image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)

    pred_landmarks = pred_landmarks.squeeze(0)
    image_save = draw_face(image_save, pred_landmarks)
    cv2.imwrite(os.path.join(save_path, filename), image_save)
    print("Img save to path {}".format(os.path.join(save_path, filename)))
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # # plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], s=0.5, marker='.', c='g')
    # ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], s=0.5, marker='.', c='r')
    # fig.savefig(os.path.join(save_path, filename))
    # plt.close()
    # plt.pause(0.001)  # pause a bit so that plots are updated


def draw_face(input_img, preds, scores=None, thresh=0.5, r=1, color=(0, 0, 255)):
    if scores:
        vis = scores > thresh
    preds = preds.astype(np.int32)
    num_kps = preds.shape[0]
    input_img = input_img.copy()
    if scores:
        for i in range(num_kps):
            if vis[i]:
                cv2.circle(input_img, (preds[i][0], preds[i][1]), r, color, -1)
    else:
        for i in range(num_kps):
            cv2.circle(input_img, (preds[i][0], preds[i][1]), r, color, -1)

    return input_img


class ToTensorTest(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image_small = np.expand_dims(image_small, axis=2)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float().div(255.0)

        return image


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[:, 0]), min(pts[:, 1]), max(pts[:, 0]), max(pts[:, 1])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    llength = llength  # * 0.8
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


class AddBoundary(object):
    def __init__(self, output_size, num_landmarks=68):
        self.num_landmarks = num_landmarks
        self.output_size = output_size

    def __call__(self, landmarks):
        landmarks_64 = np.floor(landmarks)
        if self.num_landmarks == 68:
            boundaries = {}
            boundaries['cheek'] = landmarks_64[0:17]
            boundaries['left_eyebrow'] = landmarks_64[17:22]
            boundaries['right_eyebrow'] = landmarks_64[22:27]
            boundaries['uper_left_eyelid'] = landmarks_64[36:40]
            boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [36, 41, 40, 39]])
            boundaries['upper_right_eyelid'] = landmarks_64[42:46]
            boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [42, 47, 46, 45]])
            boundaries['noise'] = landmarks_64[27:31]
            boundaries['noise_bot'] = landmarks_64[31:36]
            boundaries['upper_outer_lip'] = landmarks_64[48:55]
            boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [60, 61, 62, 63, 64]])
            boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [48, 59, 58, 57, 56, 55, 54]])
            boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
        elif self.num_landmarks == 98:
            boundaries = {}
            boundaries['cheek'] = landmarks_64[0:33]
            boundaries['left_eyebrow'] = landmarks_64[33:38]
            boundaries['right_eyebrow'] = landmarks_64[42:47]
            boundaries['uper_left_eyelid'] = landmarks_64[60:65]
            boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
            boundaries['upper_right_eyelid'] = landmarks_64[68:73]
            boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [68, 75, 74, 73, 72]])
            boundaries['noise'] = landmarks_64[51:55]
            boundaries['noise_bot'] = landmarks_64[55:60]
            boundaries['upper_outer_lip'] = landmarks_64[76:83]
            boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [88, 89, 90, 91, 92]])
            boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [76, 87, 86, 85, 84, 83, 82]])
            boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [88, 95, 94, 93, 92]])
        elif self.num_landmarks == 19:
            boundaries = {}
            boundaries['left_eyebrow'] = landmarks_64[0:3]
            boundaries['right_eyebrow'] = landmarks_64[3:5]
            boundaries['left_eye'] = landmarks_64[6:9]
            boundaries['right_eye'] = landmarks_64[9:12]
            boundaries['noise'] = landmarks_64[12:15]

        elif self.num_landmarks == 29:
            boundaries = {}
            boundaries['upper_left_eyebrow'] = np.stack([
                landmarks_64[0],
                landmarks_64[4],
                landmarks_64[2]
            ], axis=0)
            boundaries['lower_left_eyebrow'] = np.stack([
                landmarks_64[0],
                landmarks_64[5],
                landmarks_64[2]
            ], axis=0)
            boundaries['upper_right_eyebrow'] = np.stack([
                landmarks_64[1],
                landmarks_64[6],
                landmarks_64[3]
            ], axis=0)
            boundaries['lower_right_eyebrow'] = np.stack([
                landmarks_64[1],
                landmarks_64[7],
                landmarks_64[3]
            ], axis=0)
            boundaries['upper_left_eye'] = np.stack([
                landmarks_64[8],
                landmarks_64[12],
                landmarks_64[10]
            ], axis=0)
            boundaries['lower_left_eye'] = np.stack([
                landmarks_64[8],
                landmarks_64[13],
                landmarks_64[10]
            ], axis=0)
            boundaries['upper_right_eye'] = np.stack([
                landmarks_64[9],
                landmarks_64[14],
                landmarks_64[11]
            ], axis=0)
            boundaries['lower_right_eye'] = np.stack([
                landmarks_64[9],
                landmarks_64[15],
                landmarks_64[11]
            ], axis=0)
            boundaries['noise'] = np.stack([
                landmarks_64[18],
                landmarks_64[21],
                landmarks_64[19]
            ], axis=0)
            boundaries['outer_upper_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[24],
                landmarks_64[23]
            ], axis=0)
            boundaries['inner_upper_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[25],
                landmarks_64[23]
            ], axis=0)
            boundaries['outer_lower_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[26],
                landmarks_64[23]
            ], axis=0)
            boundaries['inner_lower_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[27],
                landmarks_64[23]
            ], axis=0)
        functions = {}

        for key, points in boundaries.items():
            temp = points[0]
            new_points = points[0:1, :]
            for point in points[1:]:
                if point[0] == temp[0] and point[1] == temp[1]:
                    continue
                else:
                    new_points = np.concatenate((new_points, np.expand_dims(point, 0)), axis=0)
                    temp = point
            points = new_points
            if points.shape[0] == 1:
                points = np.concatenate((points, points+0.001), axis=0)
            k = min(4, points.shape[0])
            functions[key] = interpolate.splprep([points[:, 0], points[:, 1]], k=k-1,s=0)

        boundary_map = np.zeros(self.output_size, dtype=np.float32)

        fig = plt.figure(figsize=[self.output_size[0]/96.0, self.output_size[1]/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')

        ax.imshow(boundary_map, interpolation='nearest', cmap='gray')
        # plt.show()
        # ax.scatter(landmarks[:, 0], landmarks[:, 1], s=1, marker=',', c='w')

        for key in functions.keys():
            xnew = np.arange(0, 1, 0.01)
            out = interpolate.splev(xnew, functions[key][0], der=0)
            plt.plot(out[0], out[1], ',', linewidth=1, color='w')

        img = fig2data(fig)
        plt.close()

        sigma = 1
        temp = 255-img[:,:,1]
        temp = cv2.distanceTransform(temp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        temp = temp.astype(np.float32)
        temp = np.where(temp < 3*sigma, np.exp(-(temp*temp)/(2*sigma*sigma)), 0 )

        fig = plt.figure(figsize=[self.output_size[0]/96.0, self.output_size[1]/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')
        ax.imshow(temp, cmap='gray')
        plt.close()

        boundary_map = fig2data(fig)

        return boundary_map[:, :, 0]


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis=2)
    return buf


def draw_circle_map(img, preds):
    img_out = img.copy()
    for i in range(preds.shape[0]):
        img_out = cv2.circle(img_out, (preds[i][0], preds[i][1]), 2, 1., -1)
    return img_out


def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
    affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
    affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
    affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
    Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
    x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y], 1), score
