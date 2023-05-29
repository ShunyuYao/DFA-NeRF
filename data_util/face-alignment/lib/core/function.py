# --------------------------------------------------------
# Licensed under The MIT License
# Written by Shunyu Yao (ysy at sjtu.edu.cn)
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

import cv2
import pickle as pkl
from core.evaluate import accuracy, decode_face_preds, compute_nme
from core.inference import get_final_preds
from core.inference import gaussian_modulation_torch
from core.inference import predToKeypoints
from utils.transforms import flip_back, affine_transform, pts2cs
from utils.vis import save_debug_images
from utils.vis import save_batch_heatmaps_arrays, save_batch_image_with_bbox
from utils.utils import get_preds_fromhm, save_landmarks
from utils.transforms_face import transform_preds
from utils.transforms_face import face_kpts_98_to_68
from utils.utils import decode_center_preds

from utils.utils import draw_face, draw_circle_map


logger = logging.getLogger(__name__)


def test_face_model(config, val_loader, val_dataset,
                    face_model, device, use_gpu=True,
                    output_dir='./', num_landmarks=68):
    face_model.eval()
    wrong_files_dir = os.path.join(output_dir, 'wrong_files')
    face_save_dir = os.path.join(output_dir, 'save_landmarks')
    landmark_coords_dir = os.path.join(output_dir, 'landmark_coords')

    with torch.no_grad():
        for i, (data, meta) in enumerate(val_loader):
            # data = data[0]
            # meat = meta[0]
            if meta['normal_output'] < 1:
                continue
            # print('data, meta: ', data, meta)
            filepath = meta['image']
            imgnum = meta['imgnum']
            annIds = meta['annIds']
            inv_trans_face = meta['inv_trans_face'].cpu().numpy()
            inv_trans_pose = meta['inv_trans_pose'].cpu().numpy()
            detect_wrong_face = meta['detect_wrong_face']
            detect_more_than_one_face = meta['detect_more_than_one_face']
            cropped_img = meta['cropped_img']

            cropped_img = cropped_img.squeeze().cpu().numpy()
            filename = os.path.basename(filepath[0]).split('.')[0]
            save_face_name = '{}_{}.jpg'.format(filename, annIds.cpu().numpy()[0])
            if detect_wrong_face > 0 and detect_more_than_one_face > 0:
                cropped_img = cropped_img.astype(np.uint8)[..., ::-1]
                save_filename = 'filename_' + str(annIds.cpu().numpy()[0]) + '_detect_wrong_face_and_more_than_1_face.jpg'
                cv2.imwrite(os.path.join(wrong_files_dir, save_filename), cropped_img)
            elif detect_wrong_face > 0:
                cropped_img = cropped_img.astype(np.uint8)[..., ::-1]
                save_filename = 'filename_' + str(annIds.cpu().numpy()[0]) + '_detect_wrong_face.jpg'
                print('save_filename: ', os.path.join(wrong_files_dir, save_filename))
                cv2.imwrite(os.path.join(wrong_files_dir, save_filename), cropped_img)
            elif detect_more_than_one_face > 0:
                cropped_img = cropped_img.astype(np.uint8)[..., ::-1]
                save_filename = 'filename_' + str(annIds.cpu().numpy()[0]) + '_detect_more_than_1_face.jpg'
                cv2.imwrite(os.path.join(wrong_files_dir, save_filename), cropped_img)

            step_start = time.time()

            inputs = data.type(torch.FloatTensor)
            if use_gpu:
                inputs = inputs.to(device)
            else:
                inputs = Variable(inputs)

            single_start = time.time()
            # flip test has very bad performance
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()

                inputs_both = torch.cat([inputs, input_flipped])
                outputs_both, boundary_channels = face_model(inputs_both)

                outputs = outputs_both[-1][0][None, ...]
                output_flipped = outputs_both[-1][1][None, ...]
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                outputs = (outputs + output_flipped) * 0.5
            else:
                outputs, boundary_channels = face_model(inputs)
                outputs = outputs[-1]
            single_end = time.time()
            step_end = time.time()
            for i in range(inputs.shape[0]):
                img = inputs[i]
                img = img.cpu().numpy()
                img = img.transpose((1, 2, 0))*255.0
                # img = img.astype(np.uint8)
                # img = Image.fromarray(img)

                pred_heatmap = outputs[:, :-1, :, :][i].detach().cpu()
                pred_landmarks, _, max_vals = get_preds_fromhm(pred_heatmap.unsqueeze(0))
                # print('pred_landmarks: ', pred_landmarks)
                # pos_idx = max_vals > config.FACE_MODEL.TEST_HP_THRE
                # saved_landmarks = pred_landmarks[pos_idx]
                pred_landmarks = pred_landmarks.squeeze().numpy()
                # print('pred_landmarks numpy: ', pred_landmarks)
                max_vals = max_vals.squeeze().numpy().reshape(-1, 1)

                if config.DEBUG.SAVE_FACE_LANDMARKS_PIC:
                    save_landmarks(img, pred_heatmap.numpy(), face_save_dir, save_face_name)
                if config.DEBUG.SAVE_FACE_LANDMARKS_JSON:
                    pred_landmarks = pred_landmarks * 4.0
                    # print('saved_landmarks: ', saved_landmarks)
                    num_kps = pred_landmarks.shape[0]
                    for i in range(num_kps):
                        pred_landmarks[i] = affine_transform(pred_landmarks[i], inv_trans_face)
                        pred_landmarks[i] = affine_transform(pred_landmarks[i], inv_trans_pose)

                    saved_landmarks = np.hstack((pred_landmarks, max_vals))
                    save_npy_name = save_face_name.split('.')[0] + '.npy'
                    np.save(os.path.join(landmark_coords_dir, save_npy_name), saved_landmarks)

            if i % 10 == 0:
                print('Step {} Time: {:.6f} '.format(i, step_end - step_start))


def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if torch.__version__ >= '1.1.0':
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
            # save_batch_heatmaps_arrays(output.clone().detach().cpu().numpy(), prefix, "origin")
            # print('heatmap_array saved')


def train_pose_with_wflw(config, pose_loader, face_loader, model, criterion, optimizer, lr_scheduler, epoch,
                         output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    i = 0
    face_loader_exhausted = False
    train_face = False
    pose_loader = iter(pose_loader)
    face_loader = iter(face_loader)
    len_pose = len(pose_loader)
    len_face = len(face_loader)
    assert len_pose > len_face, 'Pose dataloader must bigger than face dataloader for now.'
    face_step = int(len_pose / len_face)

    while True:
        if i % face_step == 0 and not face_loader_exhausted:
            try:
                input, target, target_weight, meta = face_loader.__next__()
                train_face = True
            except StopIteration:
                face_loader_exhausted = True
                train_face = False
        else:
            try:
                input, target, target_weight, meta = pose_loader.__next__()
            except StopIteration:
                break

        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if config.LOSS.CRITERION == 'ada_wing':
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight, meta['weight_mask'].cuda(non_blocking=True))
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight, meta['weight_mask'].cuda(non_blocking=True))
            else:
                output = outputs
                loss = criterion(output, target, target_weight, meta['weight_mask'].cuda(non_blocking=True))
        elif config.LOSS.CRITERION == 'mse':
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)
        else:
            raise Exception('Loss criterion is not implemented.')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if torch.__version__ >= '1.1.0' and config.TRAIN.USE_LR_SCHEDULER:
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len_pose + len_face, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            if train_face:
                meta['joints'] *= 4
                train_face = False
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
            # save_batch_heatmaps_arrays(output.clone().detach().cpu().numpy(), prefix, "origin")
            # print('heatmap_array saved')
        i += 1


def train_face(config, train_loader, model, criterions, optimizer, epoch,
               output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_weighted_loss = config.LOSS.USE_WEIGHTED_LOSS
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    num_joints = config.MODEL.NUM_FACE_JOINTS
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))

    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    if use_heatmap:
        criterion_hm = criterions['heatmap']
        nme_batch_sum_hm = 0
        nme_count_hm = 0
    if use_regress:
        criterion_rg = criterions['regress']
        nme_batch_sum_rg = 0
        nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_negative_example:
        losses_neg = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()
    if use_multi_eye:
        # losses_s3_eye = AverageMeter()
        losses_s4_eye = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        # print('cuda:{}'.format(config.GPUS[0]))

        input = input.to(device, non_blocking=True)
        target_weight = target_weight.float().to(device, non_blocking=True)

        # compute output
        # print("model device: ", model.device)
        outputs = model(input)
        # target = meta['joints'].float().cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        if use_regress:
            target_regress = meta['joints'].float().to(device, non_blocking=True)
            if use_head_pose:
                target_head_pose = meta['head_pose'].float().to(device, non_blocking=True)
            if use_multi_eye:
                target_eye = meta['eyes'].to(device, non_blocking=True)

            output = outputs['regress']
            output_hw = output.size(-1)
            if use_negative_example:
                if len(config.GPUS) == 1:
                    target_negative_example = meta['negative_example'].float().to(device, non_blocking=True)
                else:
                    target_negative_example = meta['negative_example'].float().cuda(non_blocking=True)
                negative_example = output[:, -1]
                output = output[:, :-1]
                loss_negative_example = F.binary_cross_entropy_with_logits(negative_example, target_negative_example)
                loss_negative_example = loss_negative_example * 0.001
                # if loss_negative_example > 0.00125:
                loss += loss_negative_example

            if use_dense_regression:
                output_stack = output.view(output.size(0), num_joints, 2, -1)
                target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                # target_weight (bs, num_joints, 1)
                target_weight_stack = target_weight.unsqueeze(-1)
                output_for_loss = output_stack * target_weight_stack
                target_regress_for_loss = target_regress * target_weight_stack
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                # NME: Now NME spend too much time so we
                # do not use it in training process
                hw_mid = output_hw // 2
                output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                output_center = output_center.mean([2, 3])
                output_center = output_center.view(output_center.size(0), -1, 2)
                preds = output_center.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds  # * 4
            else:
                if use_head_pose:
                    output_head_pose = outputs['head_pose']
                    loss_head_pose = criterion_rg(output_head_pose, target_head_pose.squeeze())
                    loss += loss_head_pose * 0.1
                if use_multi_eye:
                    # output_eye_s3 = outputs['s3_regress']
                    output_eye_s4 = outputs['s4_regress']
                    # loss_eye_s3 = criterion_rg(output_eye_s3, target_eye.squeeze())
                    loss_eye_s4 = criterion_rg(output_eye_s4, target_eye.squeeze())
                    # loss += loss_eye_s3
                    loss += loss_eye_s4 * 0.2
                output = output.view(output.size(0), -1, 2)
                target_regress = target_regress.view(target_regress.size(0), -1, 2)
                output_for_loss = output * target_weight
                target_regress_for_loss = target_regress * target_weight
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                # NME: Now NME spend too much time so we
                # do not use it in training process
                preds = output_for_loss.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds  # * 4

            # meta['pts'] = meta['pts'] * target_weight.cpu()
            # # Transform back
            # for j in range(preds.size(0)):
            #     preds[j] = transform_preds(preds[j],
            #                                meta['center'][j], meta['scale'][j],
            #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
            # # print('preds, meta: ', preds, meta['pts'])
            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_rg = nme_batch_sum_rg + np.sum(nme_batch)
            # nme_count_rg = nme_count_rg + preds.size(0)

        if use_heatmap:
            if use_boundary_map:
                target = torch.cat([target, meta['boundary_map']], axis=1).to(device, non_blocking=True)
            else:
                target = target.to(device, non_blocking=True)

            output = outputs['heatmap']

            if use_weighted_loss:
                target_mask = meta['weight_mask'].to(device, non_blocking=True)
                loss_hm = criterion_hm(output, target, target_weight, target_mask)
            else:
                loss_hm = criterion_hm(output, target, target_weight)
            loss += loss_hm * config.LOSS.LOSS_HM_RATIO

            if use_aux_head:
                output_aux = outputs['heatmap_aux']
                loss_hm_aux = criterion_hm(output_aux, target, target_weight)
                loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
            # NME
            # score_map = output.data.cpu()
            if use_boundary_map:
                output_hm = output[:, :-1, ...]
                output_bd = output[:, -1, ...]
            else:
                output_hm = output
            if config.MODEL.HEATMAP_DM:
                output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
            preds, preds_hm, maxvals = get_final_preds(
                  config, output_hm.detach().cpu().numpy(), meta)

            # preds, preds_hm = decode_face_preds(score_map, meta,
            #                                     [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
            #                                     [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
            #                                     heatmap_stride)

            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_hm = nme_batch_sum_hm + np.sum(nme_batch)
            # nme_count_hm = nme_count_hm + preds.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if use_regress and use_heatmap:
            losses_rg.update(loss_regress.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
        if use_aux_head:
            losses_aux.update(loss_hm_aux.item(), input.size(0))
        if use_negative_example:
            losses_neg.update(loss_negative_example.item(), input.size(0))
        if use_head_pose:
            losses_head_pose.update(loss_head_pose.item(), input.size(0))
        if use_multi_eye:
            losses_s4_eye.update(loss_eye_s4.item(), input.size(0))
            # losses_s3_eye.update(loss_eye_s3.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)

            if use_regress and use_heatmap:
                msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                           '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                       loss_rg=losses_rg, loss_hm=losses_hm
                                   )
                msg += msg_loss

            if use_aux_head:
                msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                    loss_aux=losses_aux
                )
                msg += msg_loss

            if use_negative_example:
                msg_loss = '\tNegative Example BCE Loss {loss_neg.val:.5f} ({loss_neg.avg:.5f})'.format(
                    loss_neg=losses_neg
                )
                msg += msg_loss

            if use_head_pose:
                msg_loss = '\tHead Pose Loss {loss_hp.val:.5f} ({loss_hp.avg:.5f})'.format(
                    loss_hp=losses_head_pose
                )
                msg += msg_loss
            if use_multi_eye:
                msg_loss = '\tEye Branch Loss {loss_s4.val:.5f} ({loss_s4.avg:.5f})'.format(
                    loss_s4=losses_s4_eye
                )
                msg += msg_loss

            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if use_regress:
                prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                tpts = meta['joints']
                tpts = tpts.reshape(tpts.size(0), -1, 2) * target_weight.cpu()
                tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]

                meta['joints'] = tpts
                output_vis = preds_rg
                # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                debug_reg_config = config
                debug_reg_config.defrost()
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                  prefix)
                debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

            if use_heatmap:
                prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'train'), i)
                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                save_debug_images(config, input, meta, target, preds_hm * heatmap_stride, output,
                                  prefix)

                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = True

    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
    # if use_heatmap:
    #     nme = nme_batch_sum_hm / nme_count_hm
    #     msg += ' nme_hm:{:.4f}'.format(nme)
    # if use_regress:
    #     nme = nme_batch_sum_rg / nme_count_rg
    #     msg += ' nme_rg:{:.4f}'.format(nme)
    logger.info(msg)

    return losses.avg


def validate_face(config, val_loader, val_dataset, model, criterions, epoch, output_dir,
                  tb_log_dir, writer_dict=None):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_weighted_loss = config.LOSS.USE_WEIGHTED_LOSS
    num_joints = config.MODEL.NUM_FACE_JOINTS
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    use_background_hm = config.MODEL.EXTRA.USE_BACKGROUND_HM if "USE_BACKGROUND_HM" in config.MODEL.EXTRA else False
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    nme_count_rg = 0
    nme_count_hm = 0
    nme_batch_sum_rg = 0
    nme_batch_sum_hm = 0
    count_failure_008_rg = 0
    count_failure_010_rg = 0
    count_failure_008_hm = 0
    count_failure_010_hm = 0
    end = time.time()

    if use_heatmap:
        criterion_hm = criterions['heatmap']
    if use_regress:
        criterion_rg = criterions['regress']
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()
    if use_multi_eye:
        # losses_s3_eye = AverageMeter()
        losses_s4_eye = AverageMeter()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU

        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            loss = 0

            device = torch.device('cuda:{}'.format(config.GPUS[0]))
            input = input.to(device, non_blocking=True)
            target_weight = target_weight.float().to(device, non_blocking=True)

            # compute output
            outputs = model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.float().cuda(non_blocking=True)

            if use_regress:
                target_regress = meta['joints'].float().to(device, non_blocking=True)
                if use_head_pose:
                    target_head_pose = meta['head_pose'].float().to(device, non_blocking=True)
                if use_multi_eye:
                    target_eye = meta['eyes'].to(device, non_blocking=True)

                output = outputs['regress']
                output_hw = output.size(-1)
                if use_negative_example:
                    target_negative_example = meta['negative_example'].float().to(device, non_blocking=True)
                    negative_example = output[:, -1]
                    output = output[:, :-1]
                if use_dense_regression:
                    output_stack = output.view(output.size(0), num_joints, 2, -1)
                    target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                    target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                    # target_weight (bs, num_joints, 1)
                    target_weight_stack = target_weight.unsqueeze(-1)
                    output_for_loss = output_stack * target_weight_stack
                    target_regress_for_loss = target_regress * target_weight_stack
                    loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                    loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                    # NME: Now NME spend too much time so we
                    # do not use it in training process
                    hw_mid = output_hw // 2
                    output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                    output_center = output_center.mean([2, 3])
                    output_center = output_center.view(output_center.size(0), -1, 2)
                    preds = output_center.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds  # * 4
                else:
                    if use_head_pose:
                        output_head_pose = outputs['head_pose']
                        loss_head_pose = criterion_rg(output_head_pose, target_head_pose.squeeze())
                        loss += loss_head_pose * 0.1
                    if use_multi_eye:
                        # output_eye_s3 = outputs['s3_regress']
                        output_eye_s4 = outputs['s4_regress']
                        # loss_eye_s3 = criterion_rg(output_eye_s3, target_eye.squeeze())
                        loss_eye_s4 = criterion_rg(output_eye_s4, target_eye.squeeze())
                        # loss += loss_eye_s3
                        loss += loss_eye_s4 * 0.2

                    output = output.view(output.size(0), -1, 2)
                    target_regress = target_regress.view(target_regress.size(0), -1, 2)
                    output_for_loss = output * target_weight
                    target_regress_for_loss = target_regress * target_weight
                    loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                    loss += loss_regress

                    # NME
                    preds = output_for_loss.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds.clone()  # * 4


                if config.FACE_DATASET.TRANSFER_98_TO_68:
                    target_weight_68 = torch.zeros(preds.size(0), 68, 1)
                    pts_68_back = torch.zeros(preds.size(0), 68, 2).float()
                    for i in range(68):
                        pts_68_back[:, i] = preds[:, face_kpts_98_to_68[i]]
                        target_weight_68[:, i] = target_weight[:, face_kpts_98_to_68[i]]
                    preds = pts_68_back
                    meta['pts'] = meta['pts'].float() * target_weight_68.float()
                else:
                    meta['pts'] = meta['pts'] * target_weight.cpu()

                nme_temp = compute_nme(preds, meta)
                # Failure Rate under different threshold
                failure_008_rg = (nme_temp > 0.08).sum()
                failure_010_rg = (nme_temp > 0.10).sum()
                count_failure_008_rg += failure_008_rg
                count_failure_010_rg += failure_010_rg

                nme_batch_sum_rg += np.sum(nme_temp)
                nme_count_rg = nme_count_rg + preds.size(0)

            if use_heatmap:
                if use_boundary_map:
                    target = torch.cat([target, meta['boundary_map']], axis=1).to(device, non_blocking=True)
                else:
                    target = target.float().to(device, non_blocking=True)

                output = outputs['heatmap']
                if use_weighted_loss:
                    target_mask = meta['weight_mask'].to(device, non_blocking=True)
                    loss_hm = criterion_hm(output, target, target_weight, target_mask)
                else:
                    loss_hm = criterion_hm(output, target, target_weight)
                loss += loss_hm
                if use_aux_head:
                    output_aux = outputs['heatmap_aux']
                    loss_hm_aux = criterion_hm(output_aux, target, target_weight)
                    loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO

                # NME
                if use_background_hm:
                    output_hm = output[:, :-1, ...]
                    output_bg = output[:, -1, ...]
                if use_boundary_map:
                    output_hm = output[:, :-1, ...]
                    output_bd = output[:, -1, ...]
                if not use_background_hm and not use_boundary_map:
                    output_hm = output

                if config.MODEL.HEATMAP_DM:
                    output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
                preds, preds_hm, maxvals = get_final_preds(
                      config, output_hm.detach().cpu().numpy(), meta)

                preds = torch.from_numpy(preds)
                nme_temp = compute_nme(preds, meta)
                # Failure Rate under different threshold
                failure_008_hm = (nme_temp > 0.08).sum()
                failure_010_hm = (nme_temp > 0.10).sum()
                count_failure_008_hm += failure_008_hm
                count_failure_010_hm += failure_010_hm

                nme_batch_sum_hm += np.sum(nme_temp)
                nme_count_hm = nme_count_hm + preds.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            if use_regress and use_heatmap:
                losses_rg.update(loss_regress.item(), input.size(0))
                losses_hm.update(loss_hm.item(), input.size(0))
            if use_aux_head:
                losses_aux.update(loss_hm_aux.item(), input.size(0))
            if use_head_pose:
                losses_head_pose.update(loss_head_pose.item(), input.size(0))
            if use_multi_eye:
                losses_s4_eye.update(loss_eye_s4.item(), input.size(0))
                # losses_s3_eye.update(loss_eye_s3.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)

                if use_regress and use_heatmap:
                    msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                               '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                           loss_rg=losses_rg, loss_hm=losses_hm
                                       )
                    msg += msg_loss

                if use_aux_head:
                    msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                        loss_aux=losses_aux
                    )
                    msg += msg_loss

                if use_head_pose:
                    msg_loss = '\tHead Pose Loss {loss_hp.val:.5f} ({loss_hp.avg:.5f})'.format(
                        loss_hp=losses_head_pose
                    )
                    msg += msg_loss
                if use_multi_eye:
                    msg_loss = '\tEye Branch Loss {loss_s4.val:.5f} ({loss_s4.avg:.5f})'.format(
                        loss_s4=losses_s4_eye
                    )
                    msg += msg_loss
                logger.info(msg)

                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'val'), i)
                    tpts = meta['joints']
                    tpts = tpts.reshape(tpts.size(0), -1, 2) * target_weight.cpu()
                    tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # print("meta['joints'] aft denormal: ", tpts)
                    meta['joints'] = tpts
                    output_vis = preds_rg
                    # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                    debug_reg_config = config
                    debug_reg_config.defrost()
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                    save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                      prefix)
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'val'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                    save_debug_images(config, input, meta, target, preds_hm * heatmap_stride, output,  #  * heatmap_stride
                                      prefix)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

        msg = 'Test Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
        if use_regress and use_heatmap:
            msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                       '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                   loss_rg=losses_rg, loss_hm=losses_hm
                               )
            msg += msg_loss
        if use_regress:
            nme_rg = nme_batch_sum_rg / nme_count_rg
            failure_008_rate = count_failure_008_rg / nme_count_rg
            failure_010_rate = count_failure_010_rg / nme_count_rg
            msg += ' nme_rg:{:.4f} [008]:{:.4f} ' \
                   '[010]:{:.4f}'.format(nme_rg, failure_008_rate, failure_010_rate)

        if use_heatmap:
            nme_hm = nme_batch_sum_hm / nme_count_hm
            failure_008_rate = count_failure_008_hm / nme_count_hm
            failure_010_rate = count_failure_010_hm / nme_count_hm
            msg += ' nme_hm:{:.4f} [008]:{:.4f} ' \
                   '[010]:{:.4f}'.format(nme_hm, failure_008_rate, failure_010_rate)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            if use_regress and use_heatmap:
                writer.add_scalar('valid_rg_loss', losses_rg.avg, global_steps)
                writer.add_scalar('valid_hm_loss', losses_hm.avg, global_steps)
            if use_regress:
                writer.add_scalar('valid_nme_rg', nme_rg, global_steps)
            if use_heatmap:
                writer.add_scalar('valid_nme_hm', nme_hm, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        if use_heatmap:
            nme = nme_hm
        if use_regress:
            nme = nme_rg

        return nme, losses.avg


def train_center_face(config, train_loader, model, criterions, optimizer, epoch,
                      output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    num_joints = config.MODEL.NUM_FACE_JOINTS
    use_densewh = config.MODEL.EXTRA.DENSE_WH if "DENSE_WH" in config.MODEL.EXTRA else None
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]

    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    losses_hm = AverageMeter()
    losses_reg = AverageMeter()
    losses_offset = AverageMeter()
    # switch to train mode
    model.train()

    criterion_hm = criterions['heatmap']
    nme_batch_sum_hm = 0
    nme_count_hm = 0

    criterion_rg = criterions['regress']
    nme_batch_sum_rg = 0
    nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_negative_example:
        losses_neg = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        loss_hm = 0
        loss_reg = 0
        loss_offset = 0
        # print('cuda:{}'.format(config.GPUS[0]))
        if len(config.GPUS) == 1:
            device = torch.device('cuda:{}'.format(config.GPUS[0]))
            input = input.to(device, non_blocking=True)
            target_weight = target_weight.float().to(device, non_blocking=True)
        else:
            input = input.cuda()
            target_weight = target_weight.float().cuda(non_blocking=True)
        # compute output
        outputs = model(input)
        # target = meta['joints'].float().cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        if use_regress:
            if len(config.GPUS) == 1:
                target_eye_hm = target.float().to(device, non_blocking=True)
                target_eye_offset = meta['eye_offsets'].float().to(device, non_blocking=True)
                target_eye_regress = meta['eye_reg'].float().to(device, non_blocking=True)
                target_reg_mask = meta['reg_mask'].float().to(device, non_blocking=True)
                if use_densewh:
                    target_eye_densewh = meta['eye_densewh'].float().to(device, non_blocking=True)
            else:
                target_eye_hm = target.float().cuda(non_blocking=True)
                target_eye_offset = meta['eye_offsets'].float().cuda(non_blocking=True)
                target_eye_regress = meta['eye_reg'].float().cuda(non_blocking=True)
                target_reg_mask = meta['reg_mask'].float().to(device, non_blocking=True)
                if use_densewh:
                    target_eye_densewh = meta['eye_densewh'].float().to(device, non_blocking=True)
            output_eye_hm = outputs['hm']
            output_eye_offset = outputs['hm_offset']
            output_eye_regress = outputs['landmarks']
            if use_densewh:
                output_eye_densewh = outputs['densewh']

            if use_dense_regression:
                output_stack = output.view(output.size(0), num_joints, 2, -1)
                target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                # target_weight (bs, num_joints, 1)
                target_weight_stack = target_weight.unsqueeze(-1)
                output_for_loss = output_stack * target_weight_stack
                target_regress_for_loss = target_regress * target_weight_stack
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                # NME: Now NME spend too much time so we
                # do not use it in training process
                hw_mid = output_hw // 2
                output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                output_center = output_center.mean([2, 3])
                output_center = output_center.view(output_center.size(0), -1, 2)
                preds = output_center.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds  # * 4
            else:
                # output = output.view(output.size(0), -1, 2)
                # target_regress = target_regress.view(target_regress.size(0), -1, 2)
                # output_for_loss = output * target_weight
                # target_regress_for_loss = target_regress * target_weight
                # loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                # loss += loss_regress * config.LOSS.LOSS_REG_RATIO
                loss_eye_hm = criterion_hm(output_eye_hm, target_eye_hm)
                loss_eye_offset = criterion_rg(output_eye_offset * target_weight, target_eye_offset * target_weight)
                loss_eye_reg = criterion_rg(output_eye_regress * target_reg_mask, target_eye_regress * target_reg_mask)
                if use_densewh:
                    loss_eye_densewh = criterion_rg(output_eye_densewh, target_eye_densewh)
                    loss += loss_eye_densewh * 100
                loss += loss_eye_hm * 0.1 + loss_eye_offset * 10 + loss_eye_reg * 1000
                loss_hm += loss_eye_hm * 0.1
                loss_reg += loss_eye_reg * 1000 # loss_eye_densewh * 100 +
                loss_offset += loss_eye_offset * 10

                preds_rg, preds_bbox = decode_center_preds(config, output_eye_hm, output_eye_offset, output_eye_regress, output_eye_densewh)

        if use_heatmap:
            if len(config.GPUS) == 1:
                target_hm = target.float().to(device, non_blocking=True)
            else:
                target_hm = target.float().cuda(non_blocking=True)
            output = outputs['heatmap']
            loss_hm = criterion_hm(output, target_hm, target_weight)
            loss += loss_hm * config.LOSS.LOSS_HM_RATIO

            if use_aux_head:
                output_aux = outputs['heatmap_aux']
                loss_hm_aux = criterion_hm(output_aux, target_hm, target_weight)
                loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
            # NME
            score_map = output.data.cpu()
            preds, preds_hm = decode_face_preds(score_map, meta,
                                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
                                                [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
                                                heatmap_stride)

            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_hm = nme_batch_sum_hm + np.sum(nme_batch)
            # nme_count_hm = nme_count_hm + preds.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses_hm.update(loss_hm.item(), input.size(0))
        losses_reg.update(loss_reg.item(), input.size(0))
        losses_offset.update(loss_offset.item(), input.size(0))
        if use_regress and use_heatmap:
            losses_rg.update(loss_regress.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
        if use_aux_head:
            losses_aux.update(loss_hm_aux.item(), input.size(0))
        if use_negative_example:
            losses_neg.update(loss_negative_example.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss_hm {loss_hm.val:.5f} ({loss_hm.avg:.5f})\t' \
                  'Loss_reg {loss_reg.val:.5f} ({loss_reg.avg:.5f})\t' \
                  'Loss_offset {loss_offset.val:.5f} ({loss_offset.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses,
                      loss_hm=losses_hm, loss_reg=losses_reg, loss_offset=losses_offset)

            if use_regress and use_heatmap:
                msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                           '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                       loss_rg=losses_rg, loss_hm=losses_hm
                                   )
                msg += msg_loss

            if use_aux_head:
                msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                    loss_aux=losses_aux
                )
                msg += msg_loss

            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if use_regress:
                prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                tpts = meta['pts_eye']

                meta['joints'] = tpts
                output_vis = preds_rg

                save_debug_images(config, input, meta, target, output_vis, output_eye_hm,
                                  prefix)

                save_batch_image_with_bbox(input, preds_bbox, '{}_bbox_pred.jpg'.format(prefix))
                save_batch_image_with_bbox(input, meta['eye_bbox'].cpu().numpy(), '{}_bbox_gt.jpg'.format(prefix))

            if use_heatmap:
                prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'train'), i)
                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                save_debug_images(config, input, meta, target, preds_hm, output,
                                  prefix)

                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = True

    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} loss_hm:{:.4f} loss_reg:{:.4f} loss_offset:{:.4f}'.format(epoch, batch_time.avg, losses.avg,
             losses_hm.avg, losses_reg.avg, losses_offset.avg)
    # if use_heatmap:
    #     nme = nme_batch_sum_hm / nme_count_hm
    #     msg += ' nme_hm:{:.4f}'.format(nme)
    # if use_regress:
    #     nme = nme_batch_sum_rg / nme_count_rg
    #     msg += ' nme_rg:{:.4f}'.format(nme)
    logger.info(msg)


def validate_center_face(config, val_loader, val_dataset, model, criterions, epoch, output_dir,
                         tb_log_dir, writer_dict=None):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    num_joints = config.MODEL.NUM_FACE_JOINTS
    use_densewh = config.MODEL.EXTRA.DENSE_WH if "DENSE_WH" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hm = AverageMeter()
    losses_reg = AverageMeter()
    losses_offset = AverageMeter()
    acc = AverageMeter()
    sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    nme_count_rg = 0
    nme_count_hm = 0
    nme_batch_sum_rg = 0
    nme_batch_sum_hm = 0
    count_failure_008_rg = 0
    count_failure_010_rg = 0
    count_failure_008_hm = 0
    count_failure_010_hm = 0
    end = time.time()

    criterion_hm = criterions['heatmap']
    criterion_rg = criterions['regress']
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU

        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            loss = 0
            loss_hm = 0
            loss_reg = 0
            loss_offset = 0
            if len(config.GPUS) == 1:
                device = torch.device('cuda:{}'.format(config.GPUS[0]))
                input = input.to(device, non_blocking=True)
                target_weight = target_weight.float().to(device, non_blocking=True)
            else:
                target_weight = target_weight.float().cuda(non_blocking=True)
            # compute output
            outputs = model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.float().cuda(non_blocking=True)

            if use_regress:
                if len(config.GPUS) == 1:
                    target_eye_hm = target.float().to(device, non_blocking=True)
                    target_eye_offset = meta['eye_offsets'].float().to(device, non_blocking=True)
                    target_eye_regress = meta['eye_reg'].float().to(device, non_blocking=True)
                    target_reg_mask = meta['reg_mask'].float().to(device, non_blocking=True)
                    if use_densewh:
                        target_eye_densewh = meta['eye_densewh'].float().to(device, non_blocking=True)
                    if use_head_pose:
                        target_head_pose = meta['head_pose'].float().to(device, non_blocking=True)
                else:
                    target_eye_hm = target.float().cuda(non_blocking=True)
                    target_eye_offset = meta['eye_offsets'].float().cuda(non_blocking=True)
                    target_eye_regress = meta['eye_reg'].float().cuda(non_blocking=True)
                    target_reg_mask = meta['reg_mask'].float().to(device, non_blocking=True)
                    if use_densewh:
                        target_eye_densewh = meta['eye_densewh'].float().to(device, non_blocking=True)
                    if use_head_pose:
                        target_head_pose = meta['head_pose'].float().to(device, non_blocking=True)
                output_eye_hm = outputs['hm']
                output_eye_offset = outputs['hm_offset']
                output_eye_regress = outputs['landmarks']
                if use_densewh:
                    output_eye_densewh = outputs['densewh']

                if use_dense_regression:
                    output_stack = output.view(output.size(0), num_joints, 2, -1)
                    target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                    target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                    # target_weight (bs, num_joints, 1)
                    target_weight_stack = target_weight.unsqueeze(-1)
                    output_for_loss = output_stack * target_weight_stack
                    target_regress_for_loss = target_regress * target_weight_stack
                    loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                    loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                    # NME: Now NME spend too much time so we
                    # do not use it in training process
                    hw_mid = output_hw // 2
                    output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                    output_center = output_center.mean([2, 3])
                    output_center = output_center.view(output_center.size(0), -1, 2)
                    preds = output_center.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds  # * 4
                else:
                    loss_eye_hm = criterion_hm(output_eye_hm, target_eye_hm)
                    loss_eye_offset = criterion_rg(output_eye_offset * target_weight, target_eye_offset * target_weight)
                    loss_eye_reg = criterion_rg(output_eye_regress * target_reg_mask, target_eye_regress * target_reg_mask)
                    if use_densewh:
                        loss_eye_densewh = criterion_rg(output_eye_densewh, target_eye_densewh)
                        loss += loss_eye_densewh * 100
                    loss += loss_eye_hm * 0.1 + loss_eye_offset * 10 + loss_eye_reg * 1000
                    loss_hm += loss_eye_hm * 0.1
                    loss_reg += loss_eye_reg * 1000
                    loss_offset += loss_eye_offset * 10

                    preds_rg, preds_bbox = decode_center_preds(config, output_eye_hm, output_eye_offset, output_eye_regress, output_eye_densewh)

                # if config.FACE_DATASET.TRANSFER_98_TO_68:
                #     target_weight_68 = torch.zeros(preds.size(0), 68, 1)
                #     pts_68_back = torch.zeros(preds.size(0), 68, 2).float()
                #     for i in range(68):
                #         pts_68_back[:, i] = preds[:, face_kpts_98_to_68[i]]
                #         target_weight_68[:, i] = target_weight[:, face_kpts_98_to_68[i]]
                #     preds = pts_68_back
                #     meta['pts'] = meta['pts'].float() * target_weight_68.float()
                # else:
                #     meta['pts'] = meta['pts'] * target_weight.cpu()

                # for j in range(preds.size(0)):
                #     preds[j] = transform_preds(preds[j],
                #                                meta['center'][j], meta['scale'][j],
                #                                [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]])
                # nme_temp = compute_nme(preds, meta)
                # # Failure Rate under different threshold
                # failure_008_rg = (nme_temp > 0.08).sum()
                # failure_010_rg = (nme_temp > 0.10).sum()
                # count_failure_008_rg += failure_008_rg
                # count_failure_010_rg += failure_010_rg
                #
                # nme_batch_sum_rg += np.sum(nme_temp)
                # nme_count_rg = nme_count_rg + preds.size(0)

            if use_heatmap:
                if len(config.GPUS) == 1:
                    target_hm = target.float().to(device, non_blocking=True)
                    target_hm_weight = target_weight.to(device, non_blocking=True)
                else:
                    target_hm = target.float().cuda(non_blocking=True)
                    target_hm_weight = target_weight.cuda(non_blocking=True)
                output = outputs['heatmap']
                loss_hm = criterion_hm(output, target_hm, target_hm_weight)
                loss += loss_hm
                if use_aux_head:
                    output_aux = outputs['heatmap_aux']
                    loss_hm_aux = criterion_hm(output_aux, target_hm, target_weight)
                    loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO

                # NME
                score_map = output.data.cpu()
                preds, preds_hm = decode_face_preds(score_map, meta,
                                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
                                                [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
                                                heatmap_stride)

                nme_temp = compute_nme(preds, meta)
                # Failure Rate under different threshold
                failure_008_hm = (nme_temp > 0.08).sum()
                failure_010_hm = (nme_temp > 0.10).sum()
                count_failure_008_hm += failure_008_hm
                count_failure_010_hm += failure_010_hm

                nme_batch_sum_hm += np.sum(nme_temp)
                nme_count_hm = nme_count_hm + preds.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
            losses_reg.update(loss_reg.item(), input.size(0))
            losses_offset.update(loss_offset.item(), input.size(0))
            if use_regress and use_heatmap:
                losses_rg.update(loss_regress.item(), input.size(0))
                losses_hm.update(loss_hm.item(), input.size(0))
            if use_aux_head:
                losses_aux.update(loss_hm_aux.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Loss_hm {loss_hm.val:.5f} ({loss_hm.avg:.5f})\t' \
                      'Loss_reg {loss_reg.val:.5f} ({loss_reg.avg:.5f})\t' \
                      'Loss_offset {loss_offset.val:.5f} ({loss_offset.avg:.5f})'.format(
                          epoch, i, len(val_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses,
                          loss_hm=losses_hm, loss_reg=losses_reg, loss_offset=losses_offset)

                if use_regress and use_heatmap:
                    msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                               '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                           loss_rg=losses_rg, loss_hm=losses_hm
                                       )
                    msg += msg_loss

                if use_aux_head:
                    msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                        loss_aux=losses_aux
                    )
                    msg += msg_loss
                logger.info(msg)

                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'val'), i)
                    tpts = meta['pts_eye']

                    meta['joints'] = tpts
                    output_vis = preds_rg

                    save_debug_images(config, input, meta, target, output_vis, output_eye_hm,
                                      prefix)

                    save_batch_image_with_bbox(input, preds_bbox, '{}_bbox_pred.jpg'.format(prefix))
                    save_batch_image_with_bbox(input, meta['eye_bbox'].cpu().numpy(), '{}_bbox_gt.jpg'.format(prefix))

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'val'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                    save_debug_images(config, input, meta, target, preds_hm, output,  #  * heatmap_stride
                                      prefix)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

        msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} loss_hm:{:.4f} loss_reg:{:.4f} loss_offset:{:.4f}'.format(epoch, batch_time.avg, losses.avg,
                 losses_hm.avg, losses_reg.avg, losses_offset.avg)
        if use_regress and use_heatmap:
            msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                       '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                   loss_rg=losses_rg, loss_hm=losses_hm
                               )
            msg += msg_loss
        # if use_regress:
        #     nme_rg = nme_batch_sum_rg / nme_count_rg
        #     failure_008_rate = count_failure_008_rg / nme_count_rg
        #     failure_010_rate = count_failure_010_rg / nme_count_rg
        #     msg += ' nme_rg:{:.4f} [008]:{:.4f} ' \
        #            '[010]:{:.4f}'.format(nme_rg, failure_008_rate, failure_010_rate)

        if use_heatmap:
            nme_hm = nme_batch_sum_hm / nme_count_hm
            failure_008_rate = count_failure_008_hm / nme_count_hm
            failure_010_rate = count_failure_010_hm / nme_count_hm
            msg += ' nme_hm:{:.4f} [008]:{:.4f} ' \
                   '[010]:{:.4f}'.format(nme_hm, failure_008_rate, failure_010_rate)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            if use_regress and use_heatmap:
                writer.add_scalar('valid_rg_loss', losses_rg.avg, global_steps)
                writer.add_scalar('valid_hm_loss', losses_hm.avg, global_steps)
            # if use_regress:
            #     writer.add_scalar('valid_nme_rg', nme_rg, global_steps)
            # if use_heatmap:
            #     writer.add_scalar('valid_nme_hm', nme_hm, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        # if use_heatmap:
        #     nme = nme_hm
        # if use_regress:
        #     nme = nme_rg

        return losses.avg


def train_eye(config, train_loader, model, criterions, optimizer, epoch,
              output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    assert config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0] == config.MODEL.IMAGE_SIZE[1] // config.MODEL.HEATMAP_SIZE[1], "heatmap scale ratio must be same"
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    if use_heatmap:
        criterion_hm = criterions['heatmap']
        nme_batch_sum_hm = 0
        nme_count_hm = 0
    if use_regress:
        criterion_rg = criterions['regress']
        nme_batch_sum_rg = 0
        nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    for i, (input, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0

        input = torch.cat(input)
        target_weight = torch.cat(target_weight)
        # for key in meta.keys():
        #     print(meta[key])
        meta = {key: torch.cat(meta[key]).float() for key in meta.keys()}
        # print('cuda:{}'.format(config.GPUS[0]))
        input = input.to(device, non_blocking=True)
        target_weight = target_weight.float().to(device, non_blocking=True)
        # compute output
        outputs = model(input)
        # target = meta['joints'].float().cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        if use_regress:
            target_regress = meta['joints'].float().to(device, non_blocking=True)
            output = outputs  # ['regress']
            output = output.view(output.size(0), -1, 2)
            target_regress = target_regress.view(target_regress.size(0), -1, 2)
            output_for_loss = output * target_weight
            target_regress_for_loss = target_regress * target_weight
            loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
            loss += loss_regress * config.LOSS.LOSS_REG_RATIO

            # NME: Now NME spend too much time so we
            # do not use it in training process
            preds = output_for_loss.data.cpu()
            preds[..., 0] = preds[..., 0] * config.MODEL.HEATMAP_SIZE[0]
            preds[..., 1] = preds[..., 1] * config.MODEL.HEATMAP_SIZE[1]
            # preds = preds.reshape(preds.shape[0], -1, 2)
            preds_rg = preds

            # meta['pts'] = meta['pts'] * target_weight.cpu()
            # # Transform back
            # for j in range(preds.size(0)):
            #     preds[j] = transform_preds(preds[j],
            #                                meta['center'][j], meta['scale'][j],
            #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
            # # print('preds, meta: ', preds, meta['pts'])
            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_rg = nme_batch_sum_rg + np.sum(nme_batch)
            # nme_count_rg = nme_count_rg + preds.size(0)

        if use_heatmap:
            target = meta['target']
            target_hm = target.float().to(device, non_blocking=True)

            output = outputs['heatmap']
            loss_hm = criterion_hm(output, target_hm, target_weight)
            loss += loss_hm * config.LOSS.LOSS_HM_RATIO

            if use_aux_head:
                output_aux = outputs['heatmap_aux']
                loss_hm_aux = criterion_hm(output_aux, target_hm, target_weight)
                loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
            # NME
            score_map = output.data.cpu()
            preds, preds_hm = decode_face_preds(score_map, meta,
                                            [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
                                            [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
                                            heatmap_stride)

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_hm = nme_batch_sum_hm + np.sum(nme_batch)
            # nme_count_hm = nme_count_hm + preds.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if use_regress and use_heatmap:
            losses_rg.update(loss_regress.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
        if use_aux_head:
            losses_aux.update(loss_hm_aux.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)

            if use_heatmap:
                msg_acc = '\tAccuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    acc=acc
                )
                msg += msg_acc

            if use_regress and use_heatmap:
                msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                           '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                       loss_rg=losses_rg, loss_hm=losses_hm
                                   )
                msg += msg_loss

            if use_aux_head:
                msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                    loss_aux=losses_aux
                )
                msg += msg_loss
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if use_regress:
                prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                tpts = meta['joints']
                tpts = tpts.reshape(tpts.size(0), -1, 2)
                tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0] * target_weight.squeeze().cpu()
                tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1] * target_weight.squeeze().cpu()
                # tpts[:, 0] = tpts[:, 0] * config.MODEL.IMAGE_SIZE[0]
                # tpts[:, 1] = tpts[:, 1] * config.MODEL.IMAGE_SIZE[1]
                # print("meta['joints'] aft denormal: ", tpts)
                meta['joints'] = tpts
                output_vis = preds_rg
                # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                debug_reg_config = config
                debug_reg_config.defrost()
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                save_debug_images(debug_reg_config, input, meta, None, output_vis, output,
                                  prefix)
                debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

            if use_heatmap:
                prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'train'), i)
                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = False
                save_debug_images(config, input, meta, target, preds_hm, output,
                                  prefix)

                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = True


    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
    if use_heatmap:
        msg += ' Accuracy:{:.4f}'.format(acc.avg)

    # if use_heatmap:
    #     nme = nme_batch_sum_hm / nme_count_hm
    #     msg += ' nme_hm:{:.4f}'.format(nme)
    # if use_regress:
    #     nme = nme_batch_sum_rg / nme_count_rg
    #     msg += ' nme_rg:{:.4f}'.format(nme)
    logger.info(msg)

    if use_heatmap:
        return acc.avg
    else:
        return losses.avg


def validate_eye(config, val_loader, val_dataset, model, criterions, epoch, output_dir,
                  tb_log_dir, writer_dict=None):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    # nme_count_rg = 0
    # nme_count_hm = 0
    # nme_batch_sum_rg = 0
    # nme_batch_sum_hm = 0
    # count_failure_008_rg = 0
    # count_failure_010_rg = 0
    # count_failure_008_hm = 0
    # count_failure_010_hm = 0
    end = time.time()

    if use_heatmap:
        criterion_hm = criterions['heatmap']
    if use_regress:
        criterion_rg = criterions['regress']
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU

        end = time.time()
        for i, (input, target_weight, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            loss = 0

            input = torch.cat(input)
            target_weight = torch.cat(target_weight)
            meta = {key: torch.cat(meta[key]).float() for key in meta.keys()}

            input = input.to(device, non_blocking=True)
            target_weight = target_weight.float().to(device, non_blocking=True)
            # compute output
            outputs = model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.float().cuda(non_blocking=True)

            if use_regress:
                target_regress = meta['joints'].float().to(device, non_blocking=True)
                output = outputs # ['regress']
                output = output.view(output.size(0), -1, 2)
                target_regress = target_regress.view(target_regress.size(0), -1, 2)
                output_for_loss = output * target_weight
                target_regress_for_loss = target_regress * target_weight
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress

                # NME
                preds = output_for_loss.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.HEATMAP_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.HEATMAP_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds

                # meta['pts'] = meta['pts'] * target_weight.cpu()
                # for j in range(preds.size(0)):
                #     preds[j] = transform_preds(preds[j],
                #                                meta['center'][j], meta['scale'][j],
                #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
                # nme_temp = compute_nme(preds, meta)
                # # Failure Rate under different threshold
                # failure_008_rg = (nme_temp > 0.08).sum()
                # failure_010_rg = (nme_temp > 0.10).sum()
                # count_failure_008_rg += failure_008_rg
                # count_failure_010_rg += failure_010_rg
                #
                # nme_batch_sum_rg += np.sum(nme_temp)
                # nme_count_rg = nme_count_rg + preds.size(0)

            if use_heatmap:
                target = meta['target']
                target_hm = target.float().to(device, non_blocking=True)
                target_hm_weight = target_weight.to(device, non_blocking=True)

                output = outputs['heatmap']
                loss_hm = criterion_hm(output, target_hm, target_hm_weight)
                loss += loss_hm
                if use_aux_head:
                    output_aux = outputs['heatmap_aux']
                    loss_hm_aux = criterion_hm(output_aux, target_hm, target_weight)
                    loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO

                score_map = output.data.cpu()
                preds, preds_hm = decode_face_preds(score_map, meta,
                                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
                                                [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
                                                heatmap_stride)

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # # NME
                # nme_temp = compute_nme(preds, meta)
                # # Failure Rate under different threshold
                # failure_008_hm = (nme_temp > 0.08).sum()
                # failure_010_hm = (nme_temp > 0.10).sum()
                # count_failure_008_hm += failure_008_hm
                # count_failure_010_hm += failure_010_hm
                #
                # nme_batch_sum_hm += np.sum(nme_temp)
                # nme_count_hm = nme_count_hm + preds.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            if use_regress and use_heatmap:
                losses_rg.update(loss_regress.item(), input.size(0))
                losses_hm.update(loss_hm.item(), input.size(0))
            if use_aux_head:
                losses_aux.update(loss_hm_aux.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)

                if use_heatmap:
                    msg_acc = '\tAccuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        acc=acc
                    )
                    msg += msg_acc

                if use_regress and use_heatmap:
                    msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                               '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                           loss_rg=losses_rg, loss_hm=losses_hm
                                       )
                    msg += msg_loss

                if use_aux_head:
                    msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                        loss_aux=losses_aux
                    )
                    msg += msg_loss
                logger.info(msg)

                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'val'), i)
                    tpts = meta['joints']
                    tpts = tpts.reshape(tpts.size(0), -1, 2)
                    tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0] * target_weight.squeeze().cpu()
                    tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1] * target_weight.squeeze().cpu()
                    # tpts[:, 0] = tpts[:, 0] * config.MODEL.IMAGE_SIZE[0]
                    # tpts[:, 1] = tpts[:, 1] * config.MODEL.IMAGE_SIZE[1]
                    # print("meta['joints'] aft denormal: ", tpts)
                    meta['joints'] = tpts
                    output_vis = preds_rg
                    # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                    debug_reg_config = config
                    debug_reg_config.defrost()
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                    save_debug_images(debug_reg_config, input, meta, None, output_vis, output,
                                      prefix)
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'val'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False
                    save_debug_images(config, input, meta, target, preds_hm, output,
                                      prefix)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

        msg = 'Test Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
        if use_regress and use_heatmap:
            msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                       '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                   loss_rg=losses_rg, loss_hm=losses_hm
                               )
            msg += msg_loss
        if use_heatmap:
            msg += ' Accuracy:{:.4f}'.format(acc.avg)

        # if use_regress:
        #     nme_rg = nme_batch_sum_rg / nme_count_rg
        #     failure_008_rate = count_failure_008_rg / nme_count_rg
        #     failure_010_rate = count_failure_010_rg / nme_count_rg
        #     msg += ' nme_rg:{:.4f} [008]:{:.4f} ' \
        #            '[010]:{:.4f}'.format(nme_rg, failure_008_rate, failure_010_rate)
        #
        # if use_heatmap:
        #     nme_hm = nme_batch_sum_hm / nme_count_hm
        #     failure_008_rate = count_failure_008_hm / nme_count_hm
        #     failure_010_rate = count_failure_010_hm / nme_count_hm
        #     msg += ' nme_hm:{:.4f} [008]:{:.4f} ' \
        #            '[010]:{:.4f}'.format(nme_hm, failure_008_rate, failure_010_rate)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            if use_regress and use_heatmap:
                writer.add_scalar('valid_rg_loss', losses_rg.avg, global_steps)
                writer.add_scalar('valid_hm_loss', losses_hm.avg, global_steps)
            # if use_regress:
            #     writer.add_scalar('valid_nme_rg', nme_rg, global_steps)
            # if use_heatmap:
            #     writer.add_scalar('valid_nme_hm', nme_hm, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        # if use_regress:
        #     nme = nme_rg
        # if use_heatmap:
        #     nme = nme_hm
        if use_heatmap:
            perf = acc.avg
        else:
            perf = losses.avg

        return perf, losses.avg


def train_eye_half(config, train_loader, model, criterions, optimizer, epoch,
                   output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    assert config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0] == config.MODEL.IMAGE_SIZE[1] // config.MODEL.HEATMAP_SIZE[1], "heatmap scale ratio must be same"
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    if use_heatmap:
        criterion_hm = criterions['heatmap']
        nme_batch_sum_hm = 0
        nme_count_hm = 0
    if use_regress:
        criterion_rg = criterions['regress']
        nme_batch_sum_rg = 0
        nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    for i, (input, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0

        # input = torch.cat(input)
        # target_weight = torch.cat(target_weight)
        # for key in meta.keys():
        #     print(meta[key])
        # meta = {key: torch.cat(meta[key]).float() for key in meta.keys()}
        # print('cuda:{}'.format(config.GPUS[0]))
        if len(config.GPUS) == 1:
            device = torch.device('cuda:{}'.format(config.GPUS[0]))
            input = input.to(device, non_blocking=True)
            target_weight = target_weight.float().to(device, non_blocking=True)
        else:
            input = input.cuda()
            target_weight = target_weight.float().cuda(non_blocking=True)
        # compute output
        outputs = model(input)
        # target = meta['joints'].float().cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        if use_regress:
            if len(config.GPUS) == 1:
                target_regress = meta['joints'].float().to(device, non_blocking=True)
            else:
                target_regress = meta['joints'].float().cuda(non_blocking=True)
            output = outputs  # ['regress']
            output = output.view(output.size(0), -1, 2)
            target_regress = target_regress.view(target_regress.size(0), -1, 2)
            output_for_loss = output * target_weight
            target_regress_for_loss = target_regress * target_weight
            loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
            loss += loss_regress * config.LOSS.LOSS_REG_RATIO

            # NME: Now NME spend too much time so we
            # do not use it in training process
            preds = output_for_loss.data.cpu()
            preds[..., 0] = preds[..., 0] * config.MODEL.HEATMAP_SIZE[0]
            preds[..., 1] = preds[..., 1] * config.MODEL.HEATMAP_SIZE[1]
            # preds = preds.reshape(preds.shape[0], -1, 2)
            preds_rg = preds

            # meta['pts'] = meta['pts'] * target_weight.cpu()
            # # Transform back
            # for j in range(preds.size(0)):
            #     preds[j] = transform_preds(preds[j],
            #                                meta['center'][j], meta['scale'][j],
            #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
            # # print('preds, meta: ', preds, meta['pts'])
            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_rg = nme_batch_sum_rg + np.sum(nme_batch)
            # nme_count_rg = nme_count_rg + preds.size(0)

        if use_heatmap:
            target = meta['target']
            if len(config.GPUS) == 1:
                target_hm = target.float().to(device, non_blocking=True)
            else:
                target_hm = target.float().cuda(non_blocking=True)
            output = outputs['heatmap']
            loss_hm = criterion_hm(output, target_hm, target_weight)
            loss += loss_hm * config.LOSS.LOSS_HM_RATIO

            if use_aux_head:
                output_aux = outputs['heatmap_aux']
                loss_hm_aux = criterion_hm(output_aux, target_hm, target_weight)
                loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
            # NME
            score_map = output.data.cpu()
            preds, preds_hm = decode_face_preds(score_map, meta,
                                            [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
                                            [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
                                            heatmap_stride)

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_hm = nme_batch_sum_hm + np.sum(nme_batch)
            # nme_count_hm = nme_count_hm + preds.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if use_regress and use_heatmap:
            losses_rg.update(loss_regress.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
        if use_aux_head:
            losses_aux.update(loss_hm_aux.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)

            if use_heatmap:
                msg_acc = '\tAccuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    acc=acc
                )
                msg += msg_acc

            if use_regress and use_heatmap:
                msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                           '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                       loss_rg=losses_rg, loss_hm=losses_hm
                                   )
                msg += msg_loss

            if use_aux_head:
                msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                    loss_aux=losses_aux
                )
                msg += msg_loss
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if use_regress:
                prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                tpts = meta['joints']
                tpts = tpts.reshape(tpts.size(0), -1, 2)
                tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0] * target_weight.squeeze().cpu()
                tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1] * target_weight.squeeze().cpu()
                # tpts[:, 0] = tpts[:, 0] * config.MODEL.IMAGE_SIZE[0]
                # tpts[:, 1] = tpts[:, 1] * config.MODEL.IMAGE_SIZE[1]
                # print("meta['joints'] aft denormal: ", tpts)
                meta['joints'] = tpts
                output_vis = preds_rg
                # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                debug_reg_config = config
                debug_reg_config.defrost()
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                save_debug_images(debug_reg_config, input, meta, None, output_vis, output,
                                  prefix)
                debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

            if use_heatmap:
                prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'train'), i)
                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = False
                save_debug_images(config, input, meta, target, preds_hm, output,
                                  prefix)

                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = True


    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
    if use_heatmap:
        msg += ' Accuracy:{:.4f}'.format(acc.avg)

    # if use_heatmap:
    #     nme = nme_batch_sum_hm / nme_count_hm
    #     msg += ' nme_hm:{:.4f}'.format(nme)
    # if use_regress:
    #     nme = nme_batch_sum_rg / nme_count_rg
    #     msg += ' nme_rg:{:.4f}'.format(nme)
    logger.info(msg)


def validate_eye_half(config, val_loader, val_dataset, model, criterions, epoch, output_dir,
                      tb_log_dir, writer_dict=None):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    # nme_count_rg = 0
    # nme_count_hm = 0
    # nme_batch_sum_rg = 0
    # nme_batch_sum_hm = 0
    # count_failure_008_rg = 0
    # count_failure_010_rg = 0
    # count_failure_008_hm = 0
    # count_failure_010_hm = 0
    end = time.time()

    if use_heatmap:
        criterion_hm = criterions['heatmap']
    if use_regress:
        criterion_rg = criterions['regress']
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU

        end = time.time()
        for i, (input, target_weight, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            loss = 0

            # input = torch.cat(input)
            # target_weight = torch.cat(target_weight)
            # meta = {key: torch.cat(meta[key]).float() for key in meta.keys()}

            if len(config.GPUS) == 1:
                device = torch.device('cuda:{}'.format(config.GPUS[0]))
                input = input.to(device, non_blocking=True)
                target_weight = target_weight.float().to(device, non_blocking=True)
            else:
                target_weight = target_weight.float().cuda(non_blocking=True)
            # compute output
            outputs = model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.float().cuda(non_blocking=True)

            if use_regress:
                if len(config.GPUS) == 1:
                    target_regress = meta['joints'].float().to(device, non_blocking=True)
                else:
                    target_regress = meta['joints'].float().cuda(non_blocking=True)
                output = outputs # ['regress']
                output = output.view(output.size(0), -1, 2)
                target_regress = target_regress.view(target_regress.size(0), -1, 2)
                output_for_loss = output * target_weight
                target_regress_for_loss = target_regress * target_weight
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress

                # NME
                preds = output_for_loss.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.HEATMAP_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.HEATMAP_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds

                # meta['pts'] = meta['pts'] * target_weight.cpu()
                # for j in range(preds.size(0)):
                #     preds[j] = transform_preds(preds[j],
                #                                meta['center'][j], meta['scale'][j],
                #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
                # nme_temp = compute_nme(preds, meta)
                # # Failure Rate under different threshold
                # failure_008_rg = (nme_temp > 0.08).sum()
                # failure_010_rg = (nme_temp > 0.10).sum()
                # count_failure_008_rg += failure_008_rg
                # count_failure_010_rg += failure_010_rg
                #
                # nme_batch_sum_rg += np.sum(nme_temp)
                # nme_count_rg = nme_count_rg + preds.size(0)

            if use_heatmap:
                target = meta['target']
                if len(config.GPUS) == 1:
                    target_hm = target.float().to(device, non_blocking=True)
                    target_hm_weight = target_weight.to(device, non_blocking=True)
                else:
                    target_hm = target.float().cuda(non_blocking=True)
                    target_hm_weight = target_weight.cuda(non_blocking=True)
                output = outputs['heatmap']
                loss_hm = criterion_hm(output, target_hm, target_hm_weight)
                loss += loss_hm
                if use_aux_head:
                    output_aux = outputs['heatmap_aux']
                    loss_hm_aux = criterion_hm(output_aux, target_hm, target_weight)
                    loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO

                score_map = output.data.cpu()
                preds, preds_hm = decode_face_preds(score_map, meta,
                                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
                                                [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
                                                heatmap_stride)

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # # NME
                # nme_temp = compute_nme(preds, meta)
                # # Failure Rate under different threshold
                # failure_008_hm = (nme_temp > 0.08).sum()
                # failure_010_hm = (nme_temp > 0.10).sum()
                # count_failure_008_hm += failure_008_hm
                # count_failure_010_hm += failure_010_hm
                #
                # nme_batch_sum_hm += np.sum(nme_temp)
                # nme_count_hm = nme_count_hm + preds.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            if use_regress and use_heatmap:
                losses_rg.update(loss_regress.item(), input.size(0))
                losses_hm.update(loss_hm.item(), input.size(0))
            if use_aux_head:
                losses_aux.update(loss_hm_aux.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)

                if use_heatmap:
                    msg_acc = '\tAccuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        acc=acc
                    )
                    msg += msg_acc

                if use_regress and use_heatmap:
                    msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                               '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                           loss_rg=losses_rg, loss_hm=losses_hm
                                       )
                    msg += msg_loss

                if use_aux_head:
                    msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                        loss_aux=losses_aux
                    )
                    msg += msg_loss
                logger.info(msg)

                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'val'), i)
                    tpts = meta['joints']
                    tpts = tpts.reshape(tpts.size(0), -1, 2)
                    tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0] * target_weight.squeeze().cpu()
                    tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1] * target_weight.squeeze().cpu()
                    # tpts[:, 0] = tpts[:, 0] * config.MODEL.IMAGE_SIZE[0]
                    # tpts[:, 1] = tpts[:, 1] * config.MODEL.IMAGE_SIZE[1]
                    # print("meta['joints'] aft denormal: ", tpts)
                    meta['joints'] = tpts
                    output_vis = preds_rg
                    # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                    debug_reg_config = config
                    debug_reg_config.defrost()
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                    save_debug_images(debug_reg_config, input, meta, None, output_vis, output,
                                      prefix)
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'val'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False
                    save_debug_images(config, input, meta, target, preds_hm*4, output,
                                      prefix)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

        msg = 'Test Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
        if use_regress and use_heatmap:
            msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                       '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                   loss_rg=losses_rg, loss_hm=losses_hm
                               )
            msg += msg_loss
        if use_heatmap:
            msg += ' Accuracy:{:.4f}'.format(acc.avg)

        # if use_regress:
        #     nme_rg = nme_batch_sum_rg / nme_count_rg
        #     failure_008_rate = count_failure_008_rg / nme_count_rg
        #     failure_010_rate = count_failure_010_rg / nme_count_rg
        #     msg += ' nme_rg:{:.4f} [008]:{:.4f} ' \
        #            '[010]:{:.4f}'.format(nme_rg, failure_008_rate, failure_010_rate)
        #
        # if use_heatmap:
        #     nme_hm = nme_batch_sum_hm / nme_count_hm
        #     failure_008_rate = count_failure_008_hm / nme_count_hm
        #     failure_010_rate = count_failure_010_hm / nme_count_hm
        #     msg += ' nme_hm:{:.4f} [008]:{:.4f} ' \
        #            '[010]:{:.4f}'.format(nme_hm, failure_008_rate, failure_010_rate)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            if use_regress and use_heatmap:
                writer.add_scalar('valid_rg_loss', losses_rg.avg, global_steps)
                writer.add_scalar('valid_hm_loss', losses_hm.avg, global_steps)
            # if use_regress:
            #     writer.add_scalar('valid_nme_rg', nme_rg, global_steps)
            # if use_heatmap:
            #     writer.add_scalar('valid_nme_hm', nme_hm, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        # if use_regress:
        #     nme = nme_rg
        # if use_heatmap:
        #     nme = nme_hm
        if use_heatmap:
            perf = acc.avg
        else:
            perf = losses.avg

        return perf


def test_eye(config, val_loader, val_dataset, model, output_dir):

    save_output = True
    save_dir = './output/eye_inference/'
    # sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU

        for i, (input, meta) in enumerate(val_loader):
            # measure data loading time
            print(meta['use_dataset'][0])
            if meta['use_dataset'][0] == '300W_open':
                continue
            loss = 0

            input = torch.cat(input)
            # meta = {key: torch.cat(meta[key]).float() for key in meta.keys()}

            input = input.cuda(non_blocking=True)
            # compute output
            outputs = model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.float().cuda(non_blocking=True)
            if save_output:
                image_ori = meta['img_ori']
                # image_ori = torch.cat(image_ori)
                image_ori = image_ori.squeeze(0)
                image_ori = image_ori.cpu().numpy()
                print("image_ori shape: ", image_ori.shape)
                heatmap = outputs['heatmap'].cpu().numpy()
                save_path = save_dir
                for bs in range(image_ori.shape[0]):
                    filename = '{:06d}_{:02d}.jpg'.format(i, bs)
                    save_landmarks(image_ori[bs][np.newaxis, ...],
                                   heatmap[bs][np.newaxis, ...],
                                    save_path, filename, rgb2bgr=True)

    return outputs


def inference_eye(config, model, image_path):
    save_output = True
    save_dir = './output/eye_inference/'
    # sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU
        image = cv2.imread(image_path)
        image = cv2.resize(image, (48, 32))
        input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        input = input.astype(np.float32) / 255.0
        input = torch.from_numpy(input).cuda(non_blocking=True)
        input = input.unsqueeze(0).unsqueeze(0)
        print("input shape: ", input.shape)
        outputs = model(input)
        print("outputs: ", outputs)
        if save_output:
            filename = '{}_eye_inference.jpg'.format(os.path.basename(image_path).split('.')[0])
            heatmap = outputs['heatmap'].cpu().numpy()
            print("heatmap shape: ", heatmap.shape)
            save_landmarks(image[np.newaxis, ...],
                           heatmap,  # [np.newaxis, ...],
                           os.path.dirname(image_path), filename, rgb2bgr=True)

    return outputs


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    DM = config.MODEL.HEATMAP_DM
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            if config.MODEL.HEATMAP_DM:
                output = gaussian_modulation_torch(output, sigma)
            preds, maxvals = get_final_preds(
                config, output.detach().cpu().numpy(), meta)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
                # save_batch_heatmaps_arrays(output.clone().cpu().numpy(),
                #                            prefix, "origin")
                # if DM:
                #     save_batch_heatmaps_arrays(output_DM, prefix, "DM")
                # print('heatmap_array saved')

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def train_face_time_se(config, train_loader, model, criterions, optimizer, epoch,
                       output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_weighted_loss = config.LOSS.USE_WEIGHTED_LOSS
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    num_joints = config.MODEL.NUM_FACE_JOINTS
    tracking_face = config.MODEL.EXTRA.TRACKING_FACE
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))
    time_se_channel = config.MODEL.EXTRA.IMG_CHANNEL - 3
    seq_len = config.MODEL.EXTRA.SEQ_LEN if "SEQ_LEN" in config.MODEL.EXTRA else len(train_loader) + 1

    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    if use_heatmap:
        criterion_hm = criterions['heatmap']
        nme_batch_sum_hm = 0
        nme_count_hm = 0
    if use_regress:
        criterion_rg = criterions['regress']
        nme_batch_sum_rg = 0
        nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_negative_example:
        losses_neg = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()
    if use_multi_eye:
        # losses_s3_eye = AverageMeter()
        losses_s4_eye = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    heatmap = np.zeros((time_se_channel, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
    last_vid_idx = -1
    seq_idx = 1

    # ct = 0
    # total_ct = 0

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        # print('cuda:{}'.format(config.GPUS[0]))

        curr_vid_idx = int(meta['vid_idx'][0])
        # if ct > 5:
        #     # print("idx {}, curr_vid_idx {}, last_vid_idx {}".format(i, curr_vid_idx, last_vid_idx))
        #     if curr_vid_idx == last_vid_idx:
        #         continue
        #     else:
        #         ct = 0
        #         total_ct += 1
        # if total_ct == 10:
        #     break
        if curr_vid_idx != last_vid_idx or seq_idx % seq_len == 0:
            heatmap = np.zeros((time_se_channel, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
            last_vid_idx = curr_vid_idx
            seq_idx = 1
            # ct += 1

        input = input.to(device, non_blocking=True)
        target_weight = target_weight.float().to(device, non_blocking=True)

        # compute output
        # print("model device: ", model.device)
        heatmap_tensor = torch.from_numpy(heatmap).to(device)
        input[:, -time_se_channel:, ...] = heatmap_tensor

        # input_vis = input.clone().detach().cpu().numpy()
        # hm0 = input_vis[0, -3, ...]
        # hm1 = input_vis[0, -2, ...]
        # hm2 = input_vis[0, -1, ...]
        # cv2.imshow("heatmap0", hm0)
        # cv2.imshow("heatmap1", hm1)
        # cv2.imshow("heatmap2", hm2)
        # cv2.waitKey()

        outputs = model(input)
        # target = meta['joints'].float().cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        if use_regress:
            target_regress = meta['joints'].float().to(device, non_blocking=True)
            if use_head_pose:
                target_head_pose = meta['head_pose'].float().to(device, non_blocking=True)
            if use_multi_eye:
                target_eye = meta['eyes'].to(device, non_blocking=True)

            output = outputs['regress']
            output_hw = output.size(-1)
            if use_negative_example:
                if len(config.GPUS) == 1:
                    target_negative_example = meta['negative_example'].float().to(device, non_blocking=True)
                else:
                    target_negative_example = meta['negative_example'].float().cuda(non_blocking=True)
                negative_example = output[:, -1]
                output = output[:, :-1]
                loss_negative_example = F.binary_cross_entropy_with_logits(negative_example, target_negative_example)
                loss_negative_example = loss_negative_example * 0.001
                # if loss_negative_example > 0.00125:
                loss += loss_negative_example

            if use_dense_regression:
                output_stack = output.view(output.size(0), num_joints, 2, -1)
                target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                # target_weight (bs, num_joints, 1)
                target_weight_stack = target_weight.unsqueeze(-1)
                output_for_loss = output_stack * target_weight_stack
                target_regress_for_loss = target_regress * target_weight_stack
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                # NME: Now NME spend too much time so we
                # do not use it in training process
                hw_mid = output_hw // 2
                output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                output_center = output_center.mean([2, 3])
                output_center = output_center.view(output_center.size(0), -1, 2)
                preds = output_center.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds  # * 4
            else:
                if use_head_pose:
                    output_head_pose = outputs['head_pose']
                    loss_head_pose = criterion_rg(output_head_pose, target_head_pose.squeeze())
                    loss += loss_head_pose * 0.1
                if use_multi_eye:
                    # output_eye_s3 = outputs['s3_regress']
                    output_eye_s4 = outputs['s4_regress']
                    # loss_eye_s3 = criterion_rg(output_eye_s3, target_eye.squeeze())
                    loss_eye_s4 = criterion_rg(output_eye_s4, target_eye.squeeze())
                    # loss += loss_eye_s3
                    loss += loss_eye_s4 * 0.2
                output = output.view(output.size(0), -1, 2)
                target_regress = target_regress.view(target_regress.size(0), -1, 2)
                output_for_loss = output * target_weight
                target_regress_for_loss = target_regress * target_weight
                loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                # NME: Now NME spend too much time so we
                # do not use it in training process
                preds = output_for_loss.data.cpu()
                preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                # preds = preds.reshape(preds.shape[0], -1, 2)
                preds_rg = preds  # * 4

            # meta['pts'] = meta['pts'] * target_weight.cpu()
            # # Transform back
            # for j in range(preds.size(0)):
            #     preds[j] = transform_preds(preds[j],
            #                                meta['center'][j], meta['scale'][j],
            #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
            # # print('preds, meta: ', preds, meta['pts'])
            # nme_batch = compute_nme(preds, meta)
            # nme_batch_sum_rg = nme_batch_sum_rg + np.sum(nme_batch)
            # nme_count_rg = nme_count_rg + preds.size(0)

        if use_heatmap:
            if use_boundary_map:
                target = torch.cat([target, meta['boundary_map']], axis=1).to(device, non_blocking=True)
            else:
                target = target.to(device, non_blocking=True)

            output = outputs['heatmap']

            if use_weighted_loss:
                target_mask = meta['weight_mask'].to(device, non_blocking=True)
                loss_hm = criterion_hm(output, target, target_weight, target_mask)
            else:
                loss_hm = criterion_hm(output, target, target_weight)
            loss += loss_hm * config.LOSS.LOSS_HM_RATIO

            if use_aux_head:
                output_aux = outputs['heatmap_aux']
                loss_hm_aux = criterion_hm(output_aux, target, target_weight)
                loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
            # NME
            # score_map = output.data.cpu()
            if use_boundary_map:
                output_hm = output[:, :-1, ...]
                output_bd = output[:, -1, ...]
            else:
                output_hm = output
            if config.MODEL.HEATMAP_DM:
                output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
            preds, preds_hm, maxvals = get_final_preds(
                  config, output_hm.detach().cpu().numpy(), meta)
            preds = preds.squeeze(0)
            preds_hm *= heatmap_stride

            heatmap[:-1] = heatmap[1:]
            heatmap[-1] = 0.
            if use_boundary_map:
                output_bd = cv2.resize(output_bd.detach().squeeze().cpu().numpy(), (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))
                heatmap[-1] = output_bd
            else:
                heatmap[-1] = draw_circle_map(heatmap[-1], preds_hm.squeeze(0))
            center, scale = pts2cs(preds, pixel_std=200.0)
            scale = scale * 1.25
            if tracking_face:
                train_loader.dataset.last_center = center
                train_loader.dataset.last_scale = scale

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if use_regress and use_heatmap:
            losses_rg.update(loss_regress.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
        if use_aux_head:
            losses_aux.update(loss_hm_aux.item(), input.size(0))
        if use_negative_example:
            losses_neg.update(loss_negative_example.item(), input.size(0))
        if use_head_pose:
            losses_head_pose.update(loss_head_pose.item(), input.size(0))
        if use_multi_eye:
            losses_s4_eye.update(loss_eye_s4.item(), input.size(0))
            # losses_s3_eye.update(loss_eye_s3.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        seq_idx += 1

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)

            if use_regress and use_heatmap:
                msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                           '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                       loss_rg=losses_rg, loss_hm=losses_hm
                                   )
                msg += msg_loss

            if use_aux_head:
                msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                    loss_aux=losses_aux
                )
                msg += msg_loss

            if use_negative_example:
                msg_loss = '\tNegative Example BCE Loss {loss_neg.val:.5f} ({loss_neg.avg:.5f})'.format(
                    loss_neg=losses_neg
                )
                msg += msg_loss

            if use_head_pose:
                msg_loss = '\tHead Pose Loss {loss_hp.val:.5f} ({loss_hp.avg:.5f})'.format(
                    loss_hp=losses_head_pose
                )
                msg += msg_loss
            if use_multi_eye:
                msg_loss = '\tEye Branch Loss {loss_s4.val:.5f} ({loss_s4.avg:.5f})'.format(
                    loss_s4=losses_s4_eye
                )
                msg += msg_loss

            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if use_regress:
                prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                tpts = meta['joints']
                tpts = tpts.reshape(tpts.size(0), -1, 2) * target_weight.cpu()
                tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]

                meta['joints'] = tpts
                output_vis = preds_rg
                # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                debug_reg_config = config
                debug_reg_config.defrost()
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                  prefix)
                debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

            if use_heatmap:
                prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'train'), i)
                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                save_debug_images(config, input, meta, target, preds_hm, output,
                                  prefix)

                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = True

    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
    # if use_heatmap:
    #     nme = nme_batch_sum_hm / nme_count_hm
    #     msg += ' nme_hm:{:.4f}'.format(nme)
    # if use_regress:
    #     nme = nme_batch_sum_rg / nme_count_rg
    #     msg += ' nme_rg:{:.4f}'.format(nme)
    logger.info(msg)

    return losses.avg


def validate_face_time_se(config, validate_loader, model, criterions, epoch,
                          output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_weighted_loss = config.LOSS.USE_WEIGHTED_LOSS
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    num_joints = config.MODEL.NUM_FACE_JOINTS
    tracking_face = config.MODEL.EXTRA.TRACKING_FACE
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))
    time_se_channel = config.MODEL.EXTRA.IMG_CHANNEL - 3

    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.eval()
    if use_heatmap:
        criterion_hm = criterions['heatmap']
        nme_batch_sum_hm = 0
        nme_count_hm = 0
    if use_regress:
        criterion_rg = criterions['regress']
        nme_batch_sum_rg = 0
        nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_negative_example:
        losses_neg = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()
    if use_multi_eye:
        # losses_s3_eye = AverageMeter()
        losses_s4_eye = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    heatmap = np.zeros((time_se_channel, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
    last_vid_idx = -1
    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(validate_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            loss = 0
            # print('cuda:{}'.format(config.GPUS[0]))

            curr_vid_idx = int(meta['vid_idx'][0])
            if curr_vid_idx != last_vid_idx:
                heatmap = np.zeros((time_se_channel, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
                last_vid_idx = curr_vid_idx
            input = input.to(device, non_blocking=True)
            target_weight = target_weight.float().to(device, non_blocking=True)

            # compute output
            # print("model device: ", model.device)
            heatmap_tensor = torch.from_numpy(heatmap).to(device)
            input[:, -time_se_channel:, ...] = heatmap_tensor

            # input_vis = input.clone().detach().cpu().numpy()
            # hm0 = input_vis[0, -3, ...]
            # hm1 = input_vis[0, -2, ...]
            # hm2 = input_vis[0, -1, ...]
            # cv2.imshow("heatmap0", hm0)
            # cv2.imshow("heatmap1", hm1)
            # cv2.imshow("heatmap2", hm2)
            # cv2.waitKey()

            outputs = model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            if use_regress:
                target_regress = meta['joints'].float().to(device, non_blocking=True)
                if use_head_pose:
                    target_head_pose = meta['head_pose'].float().to(device, non_blocking=True)
                if use_multi_eye:
                    target_eye = meta['eyes'].to(device, non_blocking=True)

                output = outputs['regress']
                output_hw = output.size(-1)
                if use_negative_example:
                    if len(config.GPUS) == 1:
                        target_negative_example = meta['negative_example'].float().to(device, non_blocking=True)
                    else:
                        target_negative_example = meta['negative_example'].float().cuda(non_blocking=True)
                    negative_example = output[:, -1]
                    output = output[:, :-1]
                    loss_negative_example = F.binary_cross_entropy_with_logits(negative_example, target_negative_example)
                    loss_negative_example = loss_negative_example * 0.001
                    # if loss_negative_example > 0.00125:
                    loss += loss_negative_example

                if use_dense_regression:
                    output_stack = output.view(output.size(0), num_joints, 2, -1)
                    target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                    target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                    # target_weight (bs, num_joints, 1)
                    target_weight_stack = target_weight.unsqueeze(-1)
                    output_for_loss = output_stack * target_weight_stack
                    target_regress_for_loss = target_regress * target_weight_stack
                    loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                    loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                    # NME: Now NME spend too much time so we
                    # do not use it in training process
                    hw_mid = output_hw // 2
                    output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                    output_center = output_center.mean([2, 3])
                    output_center = output_center.view(output_center.size(0), -1, 2)
                    preds = output_center.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds  # * 4
                else:
                    if use_head_pose:
                        output_head_pose = outputs['head_pose']
                        loss_head_pose = criterion_rg(output_head_pose, target_head_pose.squeeze())
                        loss += loss_head_pose * 0.1
                    if use_multi_eye:
                        # output_eye_s3 = outputs['s3_regress']
                        output_eye_s4 = outputs['s4_regress']
                        # loss_eye_s3 = criterion_rg(output_eye_s3, target_eye.squeeze())
                        loss_eye_s4 = criterion_rg(output_eye_s4, target_eye.squeeze())
                        # loss += loss_eye_s3
                        loss += loss_eye_s4 * 0.2
                    output = output.view(output.size(0), -1, 2)
                    target_regress = target_regress.view(target_regress.size(0), -1, 2)
                    output_for_loss = output * target_weight
                    target_regress_for_loss = target_regress * target_weight
                    loss_regress = criterion_rg(output_for_loss, target_regress_for_loss)
                    loss += loss_regress * config.LOSS.LOSS_REG_RATIO

                    # NME: Now NME spend too much time so we
                    # do not use it in training process
                    preds = output_for_loss.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds  # * 4

                # meta['pts'] = meta['pts'] * target_weight.cpu()
                # # Transform back
                # for j in range(preds.size(0)):
                #     preds[j] = transform_preds(preds[j],
                #                                meta['center'][j], meta['scale'][j],
                #                                [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]])
                # # print('preds, meta: ', preds, meta['pts'])
                # nme_batch = compute_nme(preds, meta)
                # nme_batch_sum_rg = nme_batch_sum_rg + np.sum(nme_batch)
                # nme_count_rg = nme_count_rg + preds.size(0)

            if use_heatmap:
                if use_boundary_map:
                    target = torch.cat([target, meta['boundary_map']], axis=1).to(device, non_blocking=True)
                else:
                    target = target.to(device, non_blocking=True)

                output = outputs['heatmap']

                if use_weighted_loss:
                    target_mask = meta['weight_mask'].to(device, non_blocking=True)
                    loss_hm = criterion_hm(output, target, target_weight, target_mask)
                else:
                    loss_hm = criterion_hm(output, target, target_weight)
                loss += loss_hm * config.LOSS.LOSS_HM_RATIO

                if use_aux_head:
                    output_aux = outputs['heatmap_aux']
                    loss_hm_aux = criterion_hm(output_aux, target, target_weight)
                    loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
                # NME
                # score_map = output.data.cpu()
                if use_boundary_map:
                    output_hm = output[:, :-1, ...]
                    output_bd = output[:, -1, ...]
                else:
                    output_hm = output
                if config.MODEL.HEATMAP_DM:
                    output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
                preds, preds_hm, maxvals = get_final_preds(
                      config, output_hm.detach().cpu().numpy(), meta)
                preds = preds.squeeze(0)
                preds_hm *= heatmap_stride

                heatmap[:-1] = heatmap[1:]
                heatmap[-1] = 0.
                if use_boundary_map:
                    output_bd = cv2.resize(output_bd.detach().squeeze().cpu().numpy(), (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))
                    heatmap[-1] = output_bd
                else:
                    heatmap[-1] = draw_circle_map(heatmap[-1], preds_hm.squeeze(0))
                center, scale = pts2cs(preds, pixel_std=200.0)
                scale = scale * 1.25
                if tracking_face:
                    validate_loader.dataset.last_center = center
                    validate_loader.dataset.last_scale = scale

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            if use_regress and use_heatmap:
                losses_rg.update(loss_regress.item(), input.size(0))
                losses_hm.update(loss_hm.item(), input.size(0))
            if use_aux_head:
                losses_aux.update(loss_hm_aux.item(), input.size(0))
            if use_negative_example:
                losses_neg.update(loss_negative_example.item(), input.size(0))
            if use_head_pose:
                losses_head_pose.update(loss_head_pose.item(), input.size(0))
            if use_multi_eye:
                losses_s4_eye.update(loss_eye_s4.item(), input.size(0))
                # losses_s3_eye.update(loss_eye_s3.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          epoch, i, len(validate_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)

                if use_regress and use_heatmap:
                    msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                               '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                           loss_rg=losses_rg, loss_hm=losses_hm
                                       )
                    msg += msg_loss

                if use_aux_head:
                    msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                        loss_aux=losses_aux
                    )
                    msg += msg_loss

                if use_negative_example:
                    msg_loss = '\tNegative Example BCE Loss {loss_neg.val:.5f} ({loss_neg.avg:.5f})'.format(
                        loss_neg=losses_neg
                    )
                    msg += msg_loss

                if use_head_pose:
                    msg_loss = '\tHead Pose Loss {loss_hp.val:.5f} ({loss_hp.avg:.5f})'.format(
                        loss_hp=losses_head_pose
                    )
                    msg += msg_loss
                if use_multi_eye:
                    msg_loss = '\tEye Branch Loss {loss_s4.val:.5f} ({loss_s4.avg:.5f})'.format(
                        loss_s4=losses_s4_eye
                    )
                    msg += msg_loss

                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                    tpts = meta['joints']
                    tpts = tpts.reshape(tpts.size(0), -1, 2) * target_weight.cpu()
                    tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]

                    meta['joints'] = tpts
                    output_vis = preds_rg
                    # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                    debug_reg_config = config
                    debug_reg_config.defrost()
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                    save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                      prefix)
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'test'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                    save_debug_images(config, input, meta, target, preds_hm, output,
                                      prefix)

                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

        msg = 'Test Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
        # if use_heatmap:
        #     nme = nme_batch_sum_hm / nme_count_hm
        #     msg += ' nme_hm:{:.4f}'.format(nme)
        # if use_regress:
        #     nme = nme_batch_sum_rg / nme_count_rg
        #     msg += ' nme_rg:{:.4f}'.format(nme)
        logger.info(msg)

        return losses.avg


def inference_face(config, test_loader, face_model, output_dir):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    num_joints = config.MODEL.NUM_FACE_JOINTS
    tracking_face = config.MODEL.EXTRA.TRACKING_FACE
    output_pkl_path = os.path.join(output_dir, "predict_keypoints")
    if not os.path.exists(output_pkl_path):
        os.makedirs(output_pkl_path)
    device = torch.device('cuda:{}'.format(config.GPUS[0]))
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]

    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    face_model.eval()

    with torch.no_grad():
        end = time.time()
        vid_idx_curr = '001'
        for i, (input, target, target_weight, meta) in enumerate(test_loader):
            vid_idx = meta['vid_idx'][0]
            frame_idx = meta['frame_idx'][0]

            output_vis_path = os.path.join(output_pkl_path, vid_idx)
            if not os.path.exists(output_vis_path):
                os.makedirs(output_vis_path)

            # measure data loading time
            data_time.update(time.time() - end)
            # print('cuda:{}'.format(config.GPUS[0]))

            input = input.to(device, non_blocking=True)
            # compute output
            outputs = face_model(input)
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            if use_regress:
                output = outputs['regress']
                output_hw = output.size(-1)

                if use_dense_regression:
                    output_stack = output.view(output.size(0), num_joints, 2, -1)
                    target_regress = target_regress.view(target_regress.size(0), -1, 2).unsqueeze(-1)
                    target_regress = target_regress.repeat(1, 1, 1, output_hw * output_hw)

                    # target_weight (bs, num_joints, 1)
                    target_weight_stack = target_weight.unsqueeze(-1)
                    output_for_loss = output_stack * target_weight_stack
                    target_regress_for_loss = target_regress * target_weight_stack

                    # NME: Now NME spend too much time so we
                    # do not use it in training process
                    hw_mid = output_hw // 2
                    output_center = output[:, :, hw_mid:hw_mid+2, hw_mid:hw_mid+2]
                    output_center = output_center.mean([2, 3])
                    output_center = output_center.view(output_center.size(0), -1, 2)
                    preds = output_center.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds  # * 4
                else:
                    output = output.view(output.size(0), -1, 2)

                    preds = output.data.cpu()
                    preds[..., 0] = preds[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    preds[..., 1] = preds[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # preds = preds.reshape(preds.shape[0], -1, 2)
                    preds_rg = preds  # * 4

            if use_heatmap:
                output = outputs['heatmap']
                if use_boundary_map:
                    output_hm = output[:, :-1, ...]
                    output_bd = output[:, -1, ...]
                else:
                    output_hm = output

                if config.MODEL.HEATMAP_DM:
                    output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
                preds, preds_hm, maxvals = get_final_preds(
                      config, output_hm.detach().cpu().numpy(), meta)
                preds = preds.squeeze(0)

                # img_show = meta['img_ori'].cpu().numpy().squeeze(0)
                # img_show = draw_face(img_show, preds)
                # cv2.imshow("img_show", img_show.astype(np.uint8))
                # cv2.waitKey()
                preds_hm *= heatmap_stride
                center, scale_ori = pts2cs(preds, pixel_std=200.0)
                scale = scale_ori * 1.25
                if tracking_face:
                    test_loader.dataset.last_center = center
                    test_loader.dataset.last_scale = scale
                    # print("test_loader.dataset.last_center, last_scale: ", test_loader.dataset.last_center, test_loader.dataset.last_scale)
                    # trans_face_inv = get_affine_transform(center, scale, rot=0,
                    #                                       output_size=(WIDTH, HEIGHT), inv=1)
                output_dict = {"keypoints": preds,
                               "maxval": maxvals,
                               "center": center,
                               "scale": scale_ori}
                with open(os.path.join(output_vis_path, frame_idx + ".pkl"), "wb") as f:
                    pkl.dump(output_dict, f)
                # output_arr = np.zeros((input.shape[0], preds.shape[1], 2))
                # output_arr[..., :2] = preds
                # output_arr[..., -1] = maxvals.squeeze()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Iter: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)'.format(
                          i, len(test_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time)

                logger.info(msg)
                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'test'), i)
                    tpts = meta['joints']
                    tpts = tpts.reshape(tpts.size(0), -1, 2)
                    tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]

                    meta['joints'] = tpts
                    output_vis = preds_rg
                    # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                    debug_reg_config = config
                    debug_reg_config.defrost()
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                    save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                      prefix)
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'test'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                    save_debug_images(config, input, meta, target, preds_hm, output,
                                      prefix)

                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

    logger.info(msg)


def train_face_u2net(config, train_loader, model, criterions, optimizer, epoch,
                     output_dir, tb_log_dir, writer_dict):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_weighted_loss = config.LOSS.USE_WEIGHTED_LOSS
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    num_joints = config.MODEL.NUM_FACE_JOINTS
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))

    assert use_regress or use_heatmap, 'Either regress or heatmap branch must enable.'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    losses_d0 = AverageMeter()
    losses_d1 = AverageMeter()
    losses_d2 = AverageMeter()
    losses_d3 = AverageMeter()
    losses_d4 = AverageMeter()
    losses_d5 = AverageMeter()
    # switch to train mode
    model.train()
    if use_heatmap:
        criterion_hm = criterions['heatmap']
        nme_batch_sum_hm = 0
        nme_count_hm = 0
    if use_regress:
        criterion_rg = criterions['regress']
        nme_batch_sum_rg = 0
        nme_count_rg = 0
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_negative_example:
        losses_neg = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()
    # bs = config.TRAIN.BATCH_SIZE_PER_GPU

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        # print('cuda:{}'.format(config.GPUS[0]))

        input = input.to(device, non_blocking=True)
        target_weight = target_weight.float().to(device, non_blocking=True)

        # compute output
        # print("model device: ", model.device)
        d0, d1, d2, d3, d4, d5 = model(input)
        output = d0
        # target = meta['joints'].float().cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        if use_boundary_map:
            target = torch.cat([target, meta['boundary_map']], axis=1).to(device, non_blocking=True)
        else:
            target = target.to(device, non_blocking=True)

        # if use_weighted_loss:
        #     target_mask = meta['weight_mask'].to(device, non_blocking=True)
        #     loss_hm = criterion_hm(output, target, target_weight, target_mask)
        # else:
        loss_d0 = criterion_hm(d0, target, target_weight)
        loss_d1 = criterion_hm(d1, target, target_weight)
        loss_d2 = criterion_hm(d2, target, target_weight)
        loss_d3 = criterion_hm(d3, target, target_weight)
        loss_d4 = criterion_hm(d4, target, target_weight)
        loss_d5 = criterion_hm(d5, target, target_weight)
        loss += loss_d0 + loss_d1 + loss_d2 + loss_d3 + loss_d4 + loss_d5

        if use_aux_head:
            output_aux = outputs['heatmap_aux']
            loss_hm_aux = criterion_hm(output_aux, target, target_weight)
            loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO
        # NME
        # score_map = output.data.cpu()
        # if use_boundary_map:
        #     output_hm = output[:, :-1, ...]
        #     output_bd = output[:, -1, ...]
        # else:
        #     output_hm = output
        output_hm = d0
        if config.MODEL.HEATMAP_DM:
            output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
        preds, preds_hm, maxvals = get_final_preds(
              config, output_hm.detach().cpu().numpy(), meta)

        # preds, preds_hm = decode_face_preds(score_map, meta,
        #                                     [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]],
        #                                     [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
        #                                     heatmap_stride)

        # nme_batch = compute_nme(preds, meta)
        # nme_batch_sum_hm = nme_batch_sum_hm + np.sum(nme_batch)
        # nme_count_hm = nme_count_hm + preds.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses_d0.update(loss_d0.item(), input.size(0))
        losses_d1.update(loss_d1.item(), input.size(0))
        losses_d2.update(loss_d2.item(), input.size(0))
        losses_d3.update(loss_d3.item(), input.size(0))
        losses_d4.update(loss_d4.item(), input.size(0))
        losses_d5.update(loss_d5.item(), input.size(0))
        if use_regress and use_heatmap:
            losses_rg.update(loss_regress.item(), input.size(0))
            losses_hm.update(loss_hm.item(), input.size(0))
        if use_aux_head:
            losses_aux.update(loss_hm_aux.item(), input.size(0))
        if use_negative_example:
            losses_neg.update(loss_negative_example.item(), input.size(0))
        if use_head_pose:
            losses_head_pose.update(loss_head_pose.item(), input.size(0))
        if use_multi_eye:
            losses_s4_eye.update(loss_eye_s4.item(), input.size(0))
            # losses_s3_eye.update(loss_eye_s3.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)

            msg_loss = '\tLoss d0 {loss_d0.val:.5f} ({loss_d0.avg:.5f})\t' \
                       'Loss d1 {loss_d1.val:.5f} ({loss_d1.avg:.5f})\t' \
                       'Loss d2 {loss_d2.val:.5f} ({loss_d2.avg:.5f})\t' \
                       'Loss d3 {loss_d3.val:.5f} ({loss_d3.avg:.5f})\t' \
                       'Loss d4 {loss_d4.val:.5f} ({loss_d4.avg:.5f})\t' \
                       'Loss d5 {loss_d5.val:.5f} ({loss_d5.avg:.5f})'.format(
                          loss_d0=losses_d0,
                          loss_d1=losses_d1,
                          loss_d2=losses_d2,
                          loss_d3=losses_d3,
                          loss_d4=losses_d4,
                          loss_d5=losses_d5)
            msg += msg_loss
            if use_aux_head:
                msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                    loss_aux=losses_aux
                )
                msg += msg_loss

            if use_head_pose:
                msg_loss = '\tHead Pose Loss {loss_hp.val:.5f} ({loss_hp.avg:.5f})'.format(
                    loss_hp=losses_head_pose
                )
                msg += msg_loss

            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if use_regress:
                prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'train'), i)
                tpts = meta['joints']
                tpts = tpts.reshape(tpts.size(0), -1, 2) * target_weight.cpu()
                tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]

                meta['joints'] = tpts
                output_vis = preds_rg
                # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                debug_reg_config = config
                debug_reg_config.defrost()
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                  prefix)
                debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

            if use_heatmap:
                prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'train'), i)
                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                save_debug_images(config, input, meta, target, preds_hm * heatmap_stride, output,
                                  prefix)

                if use_regress:
                    config.DEBUG.SAVE_BATCH_IMAGES_GT = True

    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
    # if use_heatmap:
    #     nme = nme_batch_sum_hm / nme_count_hm
    #     msg += ' nme_hm:{:.4f}'.format(nme)
    # if use_regress:
    #     nme = nme_batch_sum_rg / nme_count_rg
    #     msg += ' nme_rg:{:.4f}'.format(nme)
    logger.info(msg)

    return losses.avg


def validate_face_u2net(config, val_loader, val_dataset, model, criterions, epoch, output_dir,
                  tb_log_dir, writer_dict=None):
    use_regress = config.MODEL.EXTRA.USE_REGRESS_BRANCH
    use_heatmap = config.MODEL.EXTRA.USE_HEATMAP_BRANCH
    use_aux_head = config.MODEL.EXTRA.USE_AUX_HEAD
    use_negative_example = config.FACE_DATASET.NEGATIVE_EXAMPLE
    use_dense_regression = config.MODEL.EXTRA.DENSE_REGRESSION if "DENSE_REGRESSION" in config.MODEL.EXTRA else None
    use_head_pose = config.MODEL.EXTRA.USE_HEAD_POSE if "USE_HEAD_POSE" in config.MODEL.EXTRA else False
    use_weighted_loss = config.LOSS.USE_WEIGHTED_LOSS
    num_joints = config.MODEL.NUM_FACE_JOINTS
    use_multi_eye = config.MODEL.EXTRA.USE_EYE_BRANCH if "USE_EYE_BRANCH" in config.MODEL.EXTRA else False
    use_boundary_map = config.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in config.MODEL.EXTRA else False
    assert config.MODEL.IMAGE_SIZE[0] == config.MODEL.IMAGE_SIZE[1], 'img size must equal'
    assert config.MODEL.HEATMAP_SIZE[0] == config.MODEL.HEATMAP_SIZE[1], 'img size must equal'
    heatmap_stride = config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEATMAP_SIZE[0]
    device = torch.device('cuda:{}'.format(config.GPUS[0]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_d0 = AverageMeter()
    losses_d1 = AverageMeter()
    losses_d2 = AverageMeter()
    losses_d3 = AverageMeter()
    losses_d4 = AverageMeter()
    losses_d5 = AverageMeter()
    acc = AverageMeter()
    sigma = config.MODEL.SIGMA
    # switch to evaluate mode
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    DM = config.MODEL.HEATMAP_DM
    model.eval()

    nme_count_rg = 0
    nme_count_hm = 0
    nme_batch_sum_rg = 0
    nme_batch_sum_hm = 0
    count_failure_008_rg = 0
    count_failure_010_rg = 0
    count_failure_008_hm = 0
    count_failure_010_hm = 0
    end = time.time()

    if use_heatmap:
        criterion_hm = criterions['heatmap']
    if use_regress:
        criterion_rg = criterions['regress']
    if use_regress and use_heatmap:
        losses_rg = AverageMeter()
        losses_hm = AverageMeter()
    if use_aux_head:
        losses_aux = AverageMeter()
    if use_head_pose:
        losses_head_pose = AverageMeter()
    if use_multi_eye:
        # losses_s3_eye = AverageMeter()
        losses_s4_eye = AverageMeter()

    with torch.no_grad():
        # bs = config.TRAIN.BATCH_SIZE_PER_GPU

        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            loss = 0

            device = torch.device('cuda:{}'.format(config.GPUS[0]))
            input = input.to(device, non_blocking=True)
            target_weight = target_weight.float().to(device, non_blocking=True)

            # compute output
            d0, d1, d2, d3, d4, d5 = model(input)
            output = d0
            # target = meta['joints'].float().cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.float().cuda(non_blocking=True)

            if use_heatmap:
                if use_boundary_map:
                    target = torch.cat([target, meta['boundary_map']], axis=1).to(device, non_blocking=True)
                else:
                    target = target.float().to(device, non_blocking=True)

                # output = outputs['heatmap']
                if use_weighted_loss:
                    target_mask = meta['weight_mask'].to(device, non_blocking=True)
                    loss_hm = criterion_hm(output, target, target_weight, target_mask)

                if use_aux_head:
                    output_aux = outputs['heatmap_aux']
                    loss_hm_aux = criterion_hm(output_aux, target, target_weight)
                    loss += loss_hm_aux * config.LOSS.LOSS_HM_AUX_RATIO

                loss_d0 = criterion_hm(d0, target, target_weight)
                loss_d1 = criterion_hm(d1, target, target_weight)
                loss_d2 = criterion_hm(d2, target, target_weight)
                loss_d3 = criterion_hm(d3, target, target_weight)
                loss_d4 = criterion_hm(d4, target, target_weight)
                loss_d5 = criterion_hm(d5, target, target_weight)
                loss += loss_d0 + loss_d1 + loss_d2 + loss_d3 + loss_d4 + loss_d5

                # NME
                if use_boundary_map:
                    output_hm = output[:, :-1, ...]
                    output_bd = output[:, -1, ...]
                else:
                    output_hm = output
                if config.MODEL.HEATMAP_DM:
                    output_hm = gaussian_modulation_torch(output_hm, config.MODEL.FACE_SIGMA)
                preds, preds_hm, maxvals = get_final_preds(
                      config, output_hm.detach().cpu().numpy(), meta)

                preds = torch.from_numpy(preds)
                nme_temp = compute_nme(preds, meta)
                # Failure Rate under different threshold
                failure_008_hm = (nme_temp > 0.08).sum()
                failure_010_hm = (nme_temp > 0.10).sum()
                count_failure_008_hm += failure_008_hm
                count_failure_010_hm += failure_010_hm

                nme_batch_sum_hm += np.sum(nme_temp)
                nme_count_hm = nme_count_hm + preds.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            losses_d0.update(loss_d0.item(), input.size(0))
            losses_d1.update(loss_d1.item(), input.size(0))
            losses_d2.update(loss_d2.item(), input.size(0))
            losses_d3.update(loss_d3.item(), input.size(0))
            losses_d4.update(loss_d4.item(), input.size(0))
            losses_d5.update(loss_d5.item(), input.size(0))
            if use_regress and use_heatmap:
                losses_rg.update(loss_regress.item(), input.size(0))
                losses_hm.update(loss_hm.item(), input.size(0))
            if use_aux_head:
                losses_aux.update(loss_hm_aux.item(), input.size(0))
            if use_head_pose:
                losses_head_pose.update(loss_head_pose.item(), input.size(0))
            if use_multi_eye:
                losses_s4_eye.update(loss_eye_s4.item(), input.size(0))
                # losses_s3_eye.update(loss_eye_s3.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)

                msg_loss = '\tLoss d0 {loss_d0.val:.5f} ({loss_d0.avg:.5f})\t' \
                           'Loss d1 {loss_d1.val:.5f} ({loss_d1.avg:.5f})\t' \
                           'Loss d2 {loss_d2.val:.5f} ({loss_d2.avg:.5f})\t' \
                           'Loss d3 {loss_d3.val:.5f} ({loss_d3.avg:.5f})\t' \
                           'Loss d4 {loss_d4.val:.5f} ({loss_d4.avg:.5f})\t' \
                           'Loss d5 {loss_d5.val:.5f} ({loss_d5.avg:.5f})'.format(
                              loss_d0=losses_d0,
                              loss_d1=losses_d1,
                              loss_d2=losses_d2,
                              loss_d3=losses_d3,
                              loss_d4=losses_d4,
                              loss_d5=losses_d5)
                msg += msg_loss

                if use_regress and use_heatmap:
                    msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                               '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                           loss_rg=losses_rg, loss_hm=losses_hm
                                       )
                    msg += msg_loss

                if use_aux_head:
                    msg_loss = '\tAux Loss {loss_aux.val:.5f} ({loss_aux.avg:.5f})'.format(
                        loss_aux=losses_aux
                    )
                    msg += msg_loss

                if use_head_pose:
                    msg_loss = '\tHead Pose Loss {loss_hp.val:.5f} ({loss_hp.avg:.5f})'.format(
                        loss_hp=losses_head_pose
                    )
                    msg += msg_loss
                if use_multi_eye:
                    msg_loss = '\tEye Branch Loss {loss_s4.val:.5f} ({loss_s4.avg:.5f})'.format(
                        loss_s4=losses_s4_eye
                    )
                    msg += msg_loss
                logger.info(msg)

                if use_regress:
                    prefix = '{}_{}_regress'.format(os.path.join(output_dir, 'val'), i)
                    tpts = meta['joints']
                    tpts = tpts.reshape(tpts.size(0), -1, 2) * target_weight.cpu()
                    tpts[..., 0] = tpts[..., 0] * config.MODEL.IMAGE_SIZE[0]
                    tpts[..., 1] = tpts[..., 1] * config.MODEL.IMAGE_SIZE[1]
                    # print("meta['joints'] aft denormal: ", tpts)
                    meta['joints'] = tpts
                    output_vis = preds_rg
                    # output_vis = output_vis.view(output_vis.size(0), -1, 2) * config.MODEL.IMAGE_SIZE[0]
                    debug_reg_config = config
                    debug_reg_config.defrost()
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_GT:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = False
                    if debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED:
                        debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = False
                    save_debug_images(debug_reg_config, input, meta, target, output_vis, output,
                                      prefix)
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_GT = True
                    debug_reg_config.DEBUG.SAVE_HEATMAPS_PRED = True

                if use_heatmap:
                    prefix = '{}_{}_heatmap'.format(os.path.join(output_dir, 'val'), i)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = False

                    save_debug_images(config, input, meta, target, preds_hm * heatmap_stride, output,  #  * heatmap_stride
                                      prefix)
                    if use_regress:
                        config.DEBUG.SAVE_BATCH_IMAGES_GT = True

        msg = 'Test Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
        if use_regress and use_heatmap:
            msg_loss = '\tRegress Loss {loss_rg.val:.5f} ({loss_rg.avg:.5f})' \
                       '\tHeatmap Loss {loss_hm.val:.5f} ({loss_hm.avg:.5f})'.format(
                                   loss_rg=losses_rg, loss_hm=losses_hm
                               )
            msg += msg_loss
        if use_regress:
            nme_rg = nme_batch_sum_rg / nme_count_rg
            failure_008_rate = count_failure_008_rg / nme_count_rg
            failure_010_rate = count_failure_010_rg / nme_count_rg
            msg += ' nme_rg:{:.4f} [008]:{:.4f} ' \
                   '[010]:{:.4f}'.format(nme_rg, failure_008_rate, failure_010_rate)

        if use_heatmap:
            nme_hm = nme_batch_sum_hm / nme_count_hm
            failure_008_rate = count_failure_008_hm / nme_count_hm
            failure_010_rate = count_failure_010_hm / nme_count_hm
            msg += ' nme_hm:{:.4f} [008]:{:.4f} ' \
                   '[010]:{:.4f}'.format(nme_hm, failure_008_rate, failure_010_rate)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            if use_regress and use_heatmap:
                writer.add_scalar('valid_rg_loss', losses_rg.avg, global_steps)
                writer.add_scalar('valid_hm_loss', losses_hm.avg, global_steps)
            if use_regress:
                writer.add_scalar('valid_nme_rg', nme_rg, global_steps)
            if use_heatmap:
                writer.add_scalar('valid_nme_hm', nme_hm, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        if use_heatmap:
            nme = nme_hm
        if use_regress:
            nme = nme_rg

        return nme, losses.avg


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
