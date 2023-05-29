
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.PERF_INDICATOR = 'low'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.LOAD_MODEL = ''
_C.MODEL.NUM_JOINTS = 115
_C.MODEL.NUM_POSE_JOINTS = 12
_C.MODEL.NUM_FACE_JOINTS = 98
_C.MODEL.NUM_EYE_JOINTS = 9
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2.0
_C.MODEL.FACE_SIGMA = 2.0
_C.MODEL.HEATMAP_EN =True
_C.MODEL.HEATMAP_DE = False
_C.MODEL.HEATMAP_DM = False
_C.MODEL.FACE_POSE_COMBINE = True
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.FACE_MODEL = CN()
_C.FACE_MODEL.HG_BLOCKS = 4
_C.FACE_MODEL.END_RELU = False
_C.FACE_MODEL.NUM_LANDMARKS = 98
_C.FACE_MODEL.TEST_HP_THRE = 0.3

_C.FACE_DATASET = CN()
_C.FACE_DATASET.DATASET = 'face_wlfw'
_C.FACE_DATASET.TRAINSET = ''
_C.FACE_DATASET.TESTSET = ''
_C.FACE_DATASET.ROOT = ''
_C.FACE_DATASET.SCALE_FACTOR = 0.45
_C.FACE_DATASET.ROT_FACTOR = 30
_C.FACE_DATASET.SHIFT_FACTOR = 0.0
_C.FACE_DATASET.BRIGHTNESS_FACTOR = 0.0
_C.FACE_DATASET.CONTRAST_FACTOR = 0.0
_C.FACE_DATASET.FLIP = False
_C.FACE_DATASET.USE_WFLW = False
_C.FACE_DATASET.NEGATIVE_EXAMPLE = False
_C.FACE_DATASET.REGRESS_MINI_FOR_OCCLUSSION = False
_C.FACE_DATASET.TRANSFER_98_TO_68 = False
_C.FACE_DATASET.FLIP_EYE = True
_C.FACE_DATASET.FILTE_CLOSED_EYE = False
_C.FACE_DATASET.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.CRITERION = 'mse'
_C.LOSS.CRITERION_HEATMAP = 'mse'
_C.LOSS.CRITERION_REGRESS = 'mse'
_C.LOSS.LOSS_REG_RATIO = 1.0
_C.LOSS.LOSS_HM_RATIO = 1.0
_C.LOSS.LOSS_HM_AUX_RATIO = 0.2
_C.LOSS.USE_OHKM = False
_C.LOSS.USE_WEIGHTED_LOSS = True
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.POSE_TARGET_WEIGHT_RATIO = 1.0
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = False
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

_C.DATASET.TRAIN_FACE_HM_THRE = 0.5
_C.DATASET.ONLY_USE_FACE = False

# train
_C.TRAIN = CN()

_C.TRAIN.USE_LR_SCHEDULER = True
_C.TRAIN.SCHEDULER = 'MultiStepLR'
_C.TRAIN.SCHEDULER_PATIENCE = 15
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.POSE_BRANCH_LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BETA1 = 0.9
_C.TRAIN.BETA2 = 0.999

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False
_C.DEBUG.SAVE_FACE_LANDMARKS_PIC = False
_C.DEBUG.SAVE_FACE_LANDMARKS_JSON = False

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    if 'lr_range_test' in args:
        cfg.LR_RANGE_TEST = args.lr_range_test
    # if args.lr_range_test:
    #     cfg.LR_RANGE_TEST = args.lr_range_test

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
