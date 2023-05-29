# --------------------------------------------------------
# Licensed under The MIT License
# Written by Shunyu Yao (ysy at sjtu.edu.cn)
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
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
from core.function import train_face
from core.function import validate_face
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.output_csv_log import output_csv_log
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

    model = eval('models.'+cfg.MODEL.NAME+'.get_face_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))
    # summary, gflops = get_model_summary(model, dump_input)
    # logger.info(summary)

    if cfg.MODEL.LOAD_MODEL:
        checkpoint_file = cfg.MODEL.LOAD_MODEL
        assert os.path.exists(checkpoint_file), 'Model ckpt do not exists.'
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        # begin_epoch = checkpoint['epoch']
        if 'best_perf' in checkpoint.keys():
            best_perf = checkpoint['best_perf']
            logger.info("performance of the checkpoint: {}".format(best_perf))
        # last_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer_ckpt = checkpoint['optimizer']

        model.load_state_dict(checkpoint['best_state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    # if len(cfg.GPUS) == 1:
    device = torch.device('cuda:{}'.format(cfg.GPUS[0]))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(device)
    # else:
    #     model = torch.nn.DataParallel(model, device_ids=cfg.GPUS, output_device=cfg.GPUS[0])  # .cuda()
    #     print("GPUS: ", cfg.GPUS)

    # define loss function (criterion) and optimizer
    criterion = {}
    if cfg.LOSS.CRITERION_REGRESS == 'mse':
        criterion['regress'] = MSELoss()
    elif cfg.LOSS.CRITERION_REGRESS == 'smoothl1':
        criterion['regress'] = SmoothL1Loss()
    elif cfg.LOSS.CRITERION_REGRESS == 'wing':
        criterion['regress'] = WingLoss(omega=15, epsilon=3)
    elif cfg.LOSS.CRITERION_REGRESS == 'ada_wing':
        criterion['regress'] = AWingLoss()
    else:
        raise Exception('The criterion not implemented.')

    if cfg.LOSS.CRITERION_HEATMAP == 'mse':
        criterion['heatmap'] = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        )  # .cuda()
    elif cfg.LOSS.CRITERION_HEATMAP == 'ada_wing':
        criterion['heatmap'] = AdaptiveWingLoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
            use_weighted_loss=cfg.LOSS.USE_WEIGHTED_LOSS
        )  # .cuda()
    else:
        raise Exception('The criterion not implemented.')

    valid_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
        cfg, False
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=cfg.TRAIN.SHUFFLE,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )

    best_model = False
    last_epoch = -1
    model_params = model
    best_perf = 100000.0
    # model_params = [{'params': model.module.conv1.parameters()},
    #                 {'params': model.module.bn1.parameters()},
    #                 {'params': model.module.layer1.parameters()},
    #                 {'params': model.module.layer2.parameters()},
    #                 {'params': model.module.layer3.parameters()},
    #                 {'params': model.module.layer4.parameters()},
    #                 {'params': model.module.face_deconv_layers.parameters()},
    #                 {'params': model.module.face_final_layer.parameters()},
    #                 {'params': model.module.deconv_layers.parameters(),
    #                  'lr': cfg.TRAIN.POSE_BRANCH_LR},
    #                 {'params': model.module.final_layer.parameters(),
    #                  'lr': cfg.TRAIN.POSE_BRANCH_LR}
    #                 ]
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        if 'best_perf' in checkpoint.keys():
            best_perf = checkpoint['best_perf']
        elif 'perf' in checkpoint.keys():
            best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    # evaluate on validation set
    perf_indicator, test_loss = validate_face(
        cfg, valid_loader, valid_dataset, model, criterion, 0,
        final_output_dir, tb_log_dir, writer_dict
    )

    logger.info('=> performance: {} NME, test loss: {}'.format(
        perf_indicator, test_loss)
    )




if __name__ == '__main__':
    main()
