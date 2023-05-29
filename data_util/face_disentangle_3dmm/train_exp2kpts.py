import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataloader import DataLoader
from network import MouthExp2KptsNet
from utils.util import *
from utils.HyperSaver import HyperSaver
# from face_model import Face_3DMM
from data_loader import FaceExp2KptsDataset
import argparse
import os
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict


class Average_data(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

mouthIdx = range(31,51)
otherIdx = range(0,31)


def parse_args():
    """
    Create python script parameters.
    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train face 3dmm disentangle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help='description of train task'
    )
    parser.add_argument(
        "--exp_train_path",
        type=str,
        default='./dataset/exp_m_dim8_train_0725.npy'
    )
    parser.add_argument(
        "--kpts_train_path",
        type=str,
        default='./dataset/face3dmmAlignKpts_train_0725.npy'
    )
    parser.add_argument(
        "--exp_val_path",
        type=str,
        default='./dataset/exp_m_dim8_val_0725.npy'
    )
    parser.add_argument(
        "--kpts_val_path",
        type=str,
        default='./dataset/face3dmmAlignKpts_val_0725.npy'
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=256,
        help="batch size")
    parser.add_argument(
        "--exp_dim",
        type=int,
        default=8,
        help="disentangle exp dimension")
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=1,
        help="num of hidden layers"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        default=64,
        help="hidden dims"
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=300,
        help="epochs")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="num workers")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate")
    parser.add_argument(
        "--lr_step",
        type=str,
        default='250',
        help="learning rate decay step")
    parser.add_argument(
        "--coord_dim",
        type=int,
        default=2,
        help="face coordnate dimension, 2 or 3",
        choices=[2, 3])
    parser.add_argument(
        '--resume_ckpt',
        type=str,
        default=''
    )

    args = parser.parse_args()
    return args


def train(epoch, dataloader, model, device,
          optimizer, scheduler, loss_func):

    loss_avg = Average_data()
    tqdm_loader = tqdm(dataloader)
    for exp_data, kpts_gt in tqdm_loader:
        bs = kpts_gt.shape[0]
        kpts_gt = kpts_gt[:, mouthIdx, :2]
        kpts_gt = kpts_gt.view(bs, -1)

        exp_data = exp_data.to(device)
        kpts_gt = kpts_gt.to(device)

        out_kpts = model(exp_data)
        loss = loss_func(out_kpts, kpts_gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item(), 1)
        
        tqdm_loader.set_postfix(OrderedDict(stage="train", epoch=epoch, loss_avg=loss_avg.avg))

    scheduler.step()

    return loss_avg


def test(epoch, dataloader, model, device,
         loss_func, best_loss, save_path):

    loss_avg = Average_data()
    tqdm_loader = tqdm(dataloader)

    model.eval()
    with torch.no_grad():
        for exp_data, kpts_gt in tqdm_loader:
            bs = kpts_gt.shape[0]
            kpts_gt = kpts_gt[:, mouthIdx, :2]
            kpts_gt = kpts_gt.view(bs, -1)

            exp_data = exp_data.to(device)
            kpts_gt = kpts_gt.to(device)
            out_kpts = model(exp_data)
            loss = loss_func(out_kpts, kpts_gt)
            
            loss_avg.update(loss.item(), 1)
            
            tqdm_loader.set_postfix(OrderedDict(stage="test", epoch=epoch, loss_avg=loss_avg.avg))

        if best_loss > loss_avg.avg:
            best_loss = loss_avg.avg
            print('Saving..')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            state_dis = {
                'net': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dis, os.path.join(save_path, 'modelMouthExp2Kpts.pth'))

    return loss_avg


def main():
    # configuration
    args = parse_args()
    hyperSaver = HyperSaver(
        init_template='', set_id_by_time=True)
    hyperSaver.get_config_from_class(args, match_template=False)
    hyperSaver._show_serialized_json()
    log_path = os.path.join("logs", "{}_{}".format(args.exp_name, hyperSaver.time_str))
    print("log_path: ", log_path)

    BS = args.bs
    NUM_WORKERS = args.num_workers
    train_epoch = args.eps
    TRAIN_LR = args.lr # 0.001
    TRAIN_MOMENTUM = 0.9
    TRAIN_WEIGHT_DECAY = 5e-4
    TRAIN_LR_DECAY_STEP = args.lr_step.split(',') # 250
    TRAIN_LR_DECAY_STEP = [int(i) for i in TRAIN_LR_DECAY_STEP]
    print("TRAIN_LR_DECAY_STEP: ", TRAIN_LR_DECAY_STEP)
    TRAIN_LR_DECAY_RATE = 0.1
    start_epoch = 0
    lands_info = np.loadtxt('./3DMM/lands_info.txt', dtype=np.int32)
    lands_info = torch.as_tensor(lands_info).cuda()
    
    # id, focal = data_loader.load_id(os.path.join(path,'static_params.json'))
    model = MouthExp2KptsNet(input_dims=args.exp_dim, hidden_dims=args.hidden_dims, 
        num_hidden_layers=args.num_hidden_layers)
    if args.resume_ckpt:
        model_path = os.path.join(args.resume_ckpt, 'model.pth')
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
    device = torch.device("cuda:0")
    model = model.to(device)
    # id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    # face_proj = Face_3DMM('./3DMM',id_dim, exp_dim, tex_dim, point_num)
    
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_LR)
    scheduler = MultiStepLR(optimizer, TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE) # , verbose=True

    best_loss = 100
    loss_func = nn.MSELoss()
    # load data
    train_dataset = FaceExp2KptsDataset(args.exp_train_path, args.kpts_train_path)
    test_dataset = FaceExp2KptsDataset(args.exp_val_path, args.kpts_val_path)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BS, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

    loss_func = nn.L1Loss()
    for epoch in range(start_epoch, train_epoch):
        print('Epoch: %d.'%(epoch))
        train_losse = train(epoch, train_dataloader, model, device,
              optimizer, scheduler, loss_func)
        with torch.no_grad():
            test_loss = test(epoch, test_dataloader, model, device, loss_func, best_loss, log_path)
            if test_loss.avg < best_loss:
                best_loss = test_loss.avg
    # def train(epoch, dataloader, model, device,
    #       optimizer, scheduler, loss_func):
    model_perf = {
        'train_losses': {
            'loss': train_losse.avg,
        },
        'test_losses': {
            'loss_total': best_loss,
        }
    }
    hyperSaver.set_config(model_perf, match_template=False)
    hyperSaver._show_serialized_json()
    hyperSaver.save_all_configs_to_json(os.path.join(log_path, './results.json'))


if __name__ == '__main__':
    main()
