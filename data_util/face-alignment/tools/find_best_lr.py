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
from core.function import train_face
from core.function import validate_face
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.adaWing_lr_finder import AdaWing_LRFinder
from torch.nn.modules import SmoothL1Loss, MSELoss

import dataset
import models

from torch_lr_finder import LRFinder
import pickle as pkl
import os

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

    # lr finder
    parser.add_argument('--end_lr',
                        help='end lr',
                        type=float,
                        default=1.)

    parser.add_argument('--num_iters',
                        help='num iter',
                        type=int,
                        default=100)

    parser.add_argument('--lr_range_type',
                        help='exponential manner or linear for lr range test',
                        type=str,
                        default='exp',
                        choices=['exp', 'linear'])

    parser.add_argument('--batch_range',
                        help='use different batch sizes to lr test',
                        type=str,
                        default='')

    parser.add_argument('--wd_range',
                        help='use different weight_decay to lr test',
                        type=str,
                        default='')

    parser.add_argument('--batch_size_step',
                        help='batch size step',
                        type=int,
                        default=2)

    parser.add_argument('--wd_size_step',
                        help='weight decay size step',
                        type=float,
                        default=10.0)

    parser.add_argument('--no_plot',
                        help='do not plot the lr range test',
                        action='store_true')

    parser.add_argument('--save_history',
                        help='save lr range test history',
                        action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.lr_range_test = True
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if 'eye' in cfg.MODEL.NAME:
        model = eval('models.'+cfg.MODEL.NAME+'.get_eye_net')(
        cfg, is_train=True
    )

    elif 'face' in cfg.MODEL.NAME:
        model = eval('models.'+cfg.MODEL.NAME+'.get_face_net')(
            cfg, is_train=True
        )
    elif 'pose' in cfg.MODEL.NAME:
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )
    else:
        raise Exception('There is no such model: {}'.format(cfg.MODEL.NAME))

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

    #writer_dict['writer'].add_graph(model, (dump_input, ))

    #logger.info(get_model_summary(model, dump_input))

    if cfg.MODEL.LOAD_MODEL:
        checkpoint_file = cfg.MODEL.LOAD_MODEL
        assert os.path.exists(checkpoint_file), 'Model ckpt do not exists.'
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        if 'perf' in checkpoint.keys():
            best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer_ckpt = checkpoint['optimizer']

        model.load_state_dict(checkpoint['best_state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
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
        criterion['regress'] = WingLoss(omega=15, epsilon=3)
    elif cfg.LOSS.CRITERION_REGRESS == 'ada_wing':
        criterion['regress'] = AWingLoss()
    else:
        raise Exception('The criterion not implemented.')

    if cfg.LOSS.CRITERION_HEATMAP == 'mse':
        criterion['heatmap'] = MSELoss()  # nn.MSELoss(reduction='mean')
        # JointsMSELoss(
        #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        # ).cuda()
    elif cfg.LOSS.CRITERION_HEATMAP == 'ada_wing':
        criterion['heatmap'] = AdaptiveWingLoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
            use_weighted_loss=cfg.LOSS.USE_WEIGHTED_LOSS
        ).cuda()
    else:
        raise Exception('The criterion not implemented.')
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
        cfg, True
    )

    valid_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
        cfg, False
    )


    best_perf = 100000.0
    best_model = False
    last_epoch = -1
    model_params = model

    optimizer = get_optimizer(cfg, model_params, list_params=False)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        if 'best_perf' in checkpoint.keys():
            best_perf = checkpoint['best_perf']
        elif 'perf' in checkpoint.keys():
            best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    batch_range = args.batch_range
    if batch_range:
        batch_range = [int(i) for i in batch_range.split(',')]
        assert len(batch_range) == 2, 'batch range should equal to 2'

    wd_range = args.wd_range
    if wd_range:
        wd_range = [float(i) for i in wd_range.split(',')]
        assert len(wd_range) == 2, 'weight decay range should equal to 2'

    if cfg.MODEL.EXTRA.USE_REGRESS_BRANCH:
        criterion_for_lr = criterion['regress']
    elif cfg.MODEL.EXTRA.USE_HEATMAP_BRANCH:
        criterion_for_lr = criterion['heatmap']
    else:
        raise Exception("MODEL.EXTRA.USE_REGRESS_BRANCH or MODEL.EXTRA.USE_REGRESS_BRANCH must be True")
    if cfg.LOSS.CRITERION_HEATMAP == 'ada_wing':
        LRFinder_use = AdaWing_LRFinder
        lr_finder = AdaWing_LRFinder(model, optimizer, criterion_for_lr, device="cuda")
    else:
        LRFinder_use = LRFinder
        lr_finder = LRFinder(model, optimizer, criterion_for_lr, device="cuda")

    lr_range_history = {}
    if args.lr_range_type == 'exp':
        if batch_range:
            start_bs = batch_range[0]
            bs = start_bs
            while bs <= batch_range[1]:
                # train_loader.batch_size = bs
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=bs,
                    shuffle=cfg.TRAIN.SHUFFLE,
                    num_workers=cfg.WORKERS,
                    pin_memory=cfg.PIN_MEMORY
                )

                lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iters)
                lr_range_history[bs] = lr_finder.history

                bs = bs * args.batch_size_step
                lr_finder.reset()

        elif wd_range:
            start_wd = wd_range[0]
            wd = start_wd
            cfg.defrost()
            while wd <= wd_range[1]:
                cfg.TRAIN.WD = wd
                optimizer = get_optimizer(cfg, model_params, list_params=False)
                lr_finder = LRFinder_use(model, optimizer, criterion_for_lr, device="cuda")

                lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iters)
                lr_range_history[wd] = lr_finder.history

                wd = wd * args.wd_size_step
                lr_finder.reset()

        else:
            bs = cfg.TRAIN.BATCH_SIZE_PER_GPU
            lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iters)
            lr_range_history[bs] = lr_finder.history
            lr_finder.reset()
    elif args.lr_range_type == 'linear':
        if batch_range:
            start_bs = batch_range[0]
            bs = start_bs
            while bs <= batch_range[1]:
                # train_loader.batch_size = bs
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=bs,
                    shuffle=cfg.TRAIN.SHUFFLE,
                    num_workers=cfg.WORKERS,
                    pin_memory=cfg.PIN_MEMORY
                )
                lr_finder.range_test(train_loader, val_loader=valid_loader, end_lr=args.end_lr, num_iter=args.num_iters, step_mode="linear")
                lr_range_history[bs] = lr_finder.history

                bs = bs * args.batch_size_step
                lr_finder.reset()

        elif wd_range:
            start_wd = wd_range[0]
            wd = start_wd
            while wd <= wd_range[1]:
                cfg.MODEL.TRAIN.WD = wd
                optimizer = get_optimizer(cfg, model_params, list_params=False)
                lr_finder = LRFinder_use(model, optimizer, criterion_for_lr, device="cuda")

                lr_finder.range_test(train_loader, val_loader=valid_loader, end_lr=args.end_lr, num_iter=args.num_iters, step_mode="linear")
                lr_range_history[wd] = lr_finder.history

                wd = wd * args.wd_size_step
                lr_finder.reset()

        else:
            bs = cfg.TRAIN.BATCH_SIZE_PER_GPU
            lr_finder.range_test(train_loader, val_loader=valid_loader, end_lr=args.end_lr, num_iter=args.num_iters, step_mode="linear")
            lr_range_history[bs] = lr_finder.history
            lr_finder.reset()

    if args.save_history:
        save_path = os.path.join(final_output_dir, 'lr_range_test_history.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(lr_range_history, f)
            logger.info("dump the lr range history successfully.")



if __name__ == '__main__':
    main()
