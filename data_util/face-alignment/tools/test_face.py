# --------------------------------------------------------
# Licensed under The MIT License
# Written by Shunyu Yao (ysy at sjtu.edu.cn)
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import matplotlib as mpl
mpl.use('Agg')

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
from core.function import test_face_model
from utils.utils import create_logger, ToTensorTest

import dataset
from models import face_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
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

    HG_BLOCKS = cfg.FACE_MODEL.HG_BLOCKS
    END_RELU = False if cfg.FACE_MODEL.END_RELU == 'False' else True
    NUM_LANDMARKS = cfg.FACE_MODEL.NUM_LANDMARKS
    GRAY_SCALE = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            ToTensorTest(),
            # normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    use_gpu = torch.cuda.is_available()
    face_save_dir = os.path.join(final_output_dir, 'save_landmarks')
    wrong_files_dir = os.path.join(final_output_dir, 'wrong_files')
    landmark_coords_dir = os.path.join(final_output_dir, 'landmark_coords')
    if not os.path.exists(face_save_dir):
        os.makedirs(face_save_dir)
    if not os.path.exists(wrong_files_dir):
        os.makedirs(wrong_files_dir)
    if not os.path.exists(landmark_coords_dir):
        os.makedirs(landmark_coords_dir)
    # evaluate on validation set
    # validate(cfg, valid_loader, valid_dataset, model, criterion,
    #          final_output_dir, tb_log_dir)
    test_face_model(cfg, valid_loader, valid_dataset,
                    model, device, use_gpu=use_gpu,
                    output_dir=final_output_dir, num_landmarks=NUM_LANDMARKS)


if __name__ == '__main__':
    main()
