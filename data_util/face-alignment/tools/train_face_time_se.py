# --------------------------------------------------------
# Licensed under The MIT License
# Written by Shunyu Yao (ysy at sjtu.edu.cn)
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
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
from core.function import train_face_time_se, validate_face_time_se
# from core.function import validate_face
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

    input_dim = cfg.MODEL.EXTRA.IMG_CHANNEL if "IMG_CHANNEL" in cfg.MODEL.EXTRA else 3
    dump_input = torch.rand(
        (1, input_dim, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))
    summary, gflops = get_model_summary(model, dump_input)
    logger.info(summary)

    if cfg.MODEL.LOAD_MODEL:
        checkpoint_file = cfg.MODEL.LOAD_MODEL
        assert os.path.exists(checkpoint_file), 'Model ckpt do not exists.'
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        # begin_epoch = checkpoint['epoch']
        if 'best_perf' in checkpoint.keys():
            best_perf = checkpoint['best_perf']
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
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
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

    train_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
        cfg, True
    )

    valid_dataset = eval('dataset.'+cfg.FACE_DATASET.DATASET)(
        cfg, False
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

    if cfg.TRAIN.USE_LR_SCHEDULER:
        if cfg.TRAIN.SCHEDULER == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
                last_epoch=last_epoch
            )
        elif cfg.TRAIN.SCHEDULER == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      mode='min',
                                                                      factor=0.1,
                                                                      patience=cfg.TRAIN.SCHEDULER_PATIENCE,
                                                                      verbose=True,
                                                                      threshold=0.0001,
                                                                      )
        else:
            raise Exception('lr scheduler not supported')

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if torch.__version__ < '1.1.0' and cfg.TRAIN.USE_LR_SCHEDULER:
            if cfg.TRAIN.SCHEDULER == 'ReduceLROnPlateau':
                lr_scheduler.step(best_perf)
            else:
                lr_scheduler.step()

        train_loss = train_face_time_se(cfg, train_loader, model, criterion, optimizer, epoch,
                                        final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        test_loss = validate_face_time_se(
            cfg, valid_loader, model, criterion, epoch,
            final_output_dir, tb_log_dir, writer_dict
        )

        # if perf_indicator < best_perf:
        #     best_perf = perf_indicator
        #     best_model = True
        # else:
        #     best_model = False

        if torch.__version__ >= '1.1.0' and cfg.TRAIN.USE_LR_SCHEDULER:
            if cfg.TRAIN.SCHEDULER == 'ReduceLROnPlateau':
                lr_scheduler.step(best_perf)
            else:
                lr_scheduler.step()

        logger.info('=> best performance so far: {} NME'.format(
            best_perf)
        )
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'best_perf': best_perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )

    logger.info('=> best performance: {} NME'.format(
        best_perf)
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    output_csv_log(cfg, final_output_dir, train_loss, test_loss, best_perf, epoch + 1, gflops)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
