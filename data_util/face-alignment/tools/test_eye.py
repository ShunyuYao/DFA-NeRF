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
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, AdaptiveWingLoss, AWingLoss, WingLoss
from core.function import test_eye, inference_eye
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from torch.nn.modules import SmoothL1Loss, MSELoss

import dataset
import models


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

    # philly
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
    parser.add_argument('--load_last_epoch',
                        help='load last epoch',
                        action='store_true')
    parser.add_argument('--image_path',
                        type=str,
                        default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_eye_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    dump_input = torch.rand(
        (1, 1, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    if cfg.MODEL.LOAD_MODEL:
        checkpoint_file = cfg.MODEL.LOAD_MODEL
        assert os.path.exists(checkpoint_file), 'Model ckpt do not exists.'
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        # begin_epoch = checkpoint['epoch']
        state_dict = torch.load(checkpoint_file)
        if not args.load_last_epoch:
            print('Load the best performance model weights.')
            print('Best performance: ', state_dict['best_perf'])
            model.load_state_dict(state_dict['best_state_dict'])
        else:
            print('Load the last epoch model weights.')
            state_dict = state_dict['state_dict']
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

    if len(cfg.GPUS) == 1:
        device = torch.device('cuda:{}'.format(cfg.GPUS[0]))
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = {}
    if cfg.LOSS.CRITERION_REGRESS == 'mse':
        criterion['regress'] = MSELoss()
    elif cfg.LOSS.CRITERION_REGRESS == 'smoothl1':
        criterion['regress'] = SmoothL1Loss()
    elif cfg.LOSS.CRITERION_REGRESS == 'wing':
        criterion['regress'] = WingLoss()
    elif cfg.LOSS.CRITERION_REGRESS == 'ada_wing':
        criterion['regress'] = AWingLoss()
    else:
        raise Exception('The criterion not implemented.')

    if cfg.LOSS.CRITERION_HEATMAP == 'mse':
        criterion['heatmap'] = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif cfg.LOSS.CRITERION_HEATMAP == 'ada_wing':
        criterion['heatmap'] = AdaptiveWingLoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
            use_weighted_loss=cfg.LOSS.USE_WEIGHTED_LOSS
        ).cuda()
    else:
        raise Exception('The criterion not implemented.')
    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    # train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    if not args.image_path:
        test_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
            cfg, True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

    best_model = False
    last_epoch = -1
    model_params = model
    if cfg.PERF_INDICATOR == 'low':
        best_perf = 100000.0
        lr_scheduler_mode = 'min'
    elif cfg.PERF_INDICATOR == 'high':
        best_perf = 0.0
        lr_scheduler_mode = 'max'
    else:
        raise Exception('cfg.PERF_INDICATOR must be low or high')

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    if args.image_path:
        inference_eye(cfg, model, args.image_path)
    else:
        # evaluate on validation set
        test_eye(cfg, test_loader, test_dataset, model, final_output_dir)

    # logger.info('=> performance: {}'.format(
    #     perf_indicator)
    # )


if __name__ == '__main__':
    main()
