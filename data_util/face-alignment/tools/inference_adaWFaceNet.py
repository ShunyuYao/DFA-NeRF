import argparse
import pprint
import os
import cv2

import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from utils.utils import create_logger, ToTensorTest
from utils.utils import get_preds_fromhm, save_landmarks
from utils.transforms import affine_transform

import matplotlib.pyplot as plt
from dataset.DirectoryImageDataset import DirectoryImageDataset
from models import face_net

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--filepath',
                        help='inference file path',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    filepath = args.filepath
    HG_BLOCKS = cfg.FACE_MODEL.HG_BLOCKS
    END_RELU = False if cfg.FACE_MODEL.END_RELU == 'False' else True
    NUM_LANDMARKS = cfg.FACE_MODEL.NUM_LANDMARKS
    GRAY_SCALE = False

    device = torch.device("cuda:{}".format(cfg.GPUS[0]) if torch.cuda.is_available() else "cpu")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = face_net.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)
    # model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        checkpoint = torch.load(cfg.TEST.MODEL_FILE)
        if 'state_dict' not in checkpoint:
            print('state dicts!!!!')
            model.load_state_dict(checkpoint)
        else:
            print('not state dicts!!!!')
            pretrained_weights = checkpoint['state_dict']
            model_weights = model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            model.load_state_dict(model_weights)

    #     logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    #     model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    # else:
    #     pretrained_weights = checkpoint['state_dict']
    #     model_weights = model.state_dict()
    #     pretrained_weights = {k: v for k, v in pretrained_weights.items() \
    #                           if k in model_weights}
    #     model_weights.update(pretrained_weights)
    #     model.load_state_dict(model_weights)

    model = model.to(device)

    print("filepath: ", filepath)
    dataset = DirectoryImageDataset(filepath, transform=ToTensorTest())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # img, meta = dataset[1]
    # img_show = img.cpu().numpy().transpose((1,2,0)) * 255.0
    # cv2.imwrite("./models/test_pic.jpg", img_show.astype(np.uint8))
    # exit()

    face_save_dir = os.path.join(final_output_dir, 'save_landmarks')
    landmark_coords_dir = os.path.join(final_output_dir, 'landmark_coords')
    if not os.path.exists(face_save_dir):
        os.makedirs(face_save_dir)
    if not os.path.exists(landmark_coords_dir):
        os.makedirs(landmark_coords_dir)
    model.eval()
    with torch.no_grad():
        for i, (img, meta) in enumerate(dataloader):
            if not meta:
                continue
            inv_trans_face = meta['trans'].cpu().numpy()
            filepath = meta['filepath']
            meta_new = {}
            meta_new['center'] = meta['center'] # .cpu().numpy()
            meta_new['scale'] = meta['scale']   # .cpu().numpy()

            filename = os.path.basename(filepath[0])
            # save_face_name = os.path.join(face_save_dir, filename)
            # print("save_face_name: ", save_face_name)

            input = img.to(device)
            outputs, boundary_channels = model(input)
            outputs = outputs[-1]

            for i in range(input.shape[0]):
                img = input[i]
                img = img.cpu().numpy()
                img = img.transpose((1, 2, 0))*255.0

                pred_heatmap = outputs[:, :-1, :, :][i].detach().cpu()
                pred_landmarks, max_vals = get_preds_fromhm(pred_heatmap.unsqueeze(0))
                pred_landmarks = pred_landmarks.squeeze().numpy()
                max_vals = max_vals.squeeze().numpy().reshape(-1, 1)

                # .cpu().numpy()
                save_landmarks(img, pred_heatmap.unsqueeze(0).cpu().numpy(), face_save_dir, filename, cfg, meta_new, 4.0, True)

                pred_landmarks = pred_landmarks * 4.0
                # print('saved_landmarks: ', saved_landmarks)
                num_kps = pred_landmarks.shape[0]
                for i in range(num_kps):
                    pred_landmarks[i] = affine_transform(pred_landmarks[i], inv_trans_face)

                saved_landmarks = np.hstack((pred_landmarks, max_vals))
                save_npy_name = filename.split('.')[0] + '.npy'
                print("save_npy_name: ", save_npy_name)
                np.save(os.path.join(landmark_coords_dir, save_npy_name), saved_landmarks)



if __name__ == '__main__':
    main()
