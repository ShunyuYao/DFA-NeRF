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
from core.loss import JointsMSELoss, AdaptiveWingLoss
from core.function import train, train_pose_with_wflw
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

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

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
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
    #writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    if cfg.MODEL.LOAD_MODEL:
        checkpoint_file = cfg.MODEL.LOAD_MODEL
        assert os.path.exists(checkpoint_file), 'Model ckpt do not exists.'
        checkpoint_file_sd = os.path.join(os.path.dirname(checkpoint_file), 'checkpoint_final.pth')
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        if 'perf' in checkpoint.keys():
            best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])

        # optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_ckpt = checkpoint['optimizer']

        checkpoint = torch.load(checkpoint_file_sd)
        model.load_state_dict(checkpoint)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if cfg.LOSS.CRITERION == 'mse':
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif cfg.LOSS.CRITERION == 'ada_wing':
        criterion = AdaptiveWingLoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
            use_weighted_loss=cfg.LOSS.USE_WEIGHTED_LOSS
        ).cuda()
    else:
        raise Exception('The criterion not implemented.')

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    if cfg.FACE_DATASET.USE_WFLW:
        train_face_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
            cfg, True
        )
        train_face_loader = torch.utils.data.DataLoader(
            train_face_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

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

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    model_params = model
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
    optimizer = get_optimizer(cfg, model_params, list_params=False)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    optimizer.load_state_dict(optimizer_ckpt)

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        if 'perf' in checkpoint.keys():
            best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if torch.__version__ < '1.1.0' and cfg.TRAIN.USE_LR_SCHEDULER:
            lr_scheduler.step()

        # train for one epoch
        if cfg.FACE_DATASET.USE_WFLW:
            train_pose_with_wflw(cfg, train_loader, train_face_loader,
                                 model, criterion, optimizer, lr_scheduler, epoch,
                                 final_output_dir, tb_log_dir, writer_dict)
        else:
            train(cfg, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
                  final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        # perf_indicator = validate(
        #     cfg, valid_loader, valid_dataset, model, criterion,
        #     final_output_dir, tb_log_dir, writer_dict
        # )

        # if perf_indicator >= best_perf:
        #     best_perf = perf_indicator
        #     best_model = True
        # else:
        #     best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
