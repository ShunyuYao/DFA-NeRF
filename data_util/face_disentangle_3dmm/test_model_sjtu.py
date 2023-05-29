import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataloader import DataLoader
import network 
from utils.util import *
from face_model import Face_3DMM
from data_loader import FaceSJTUDataset
import argparse
import os
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict


def parse_args():
    """
    Create python script parameters.
    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Test face 3dmm disentangle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train_LSR2_data0725_dim20_8_202107251748",
        help="checkpoint path")
    parser.add_argument(
        "--bs",
        type=int,
        default=256,
        help="batch size")
    parser.add_argument(
        "--dim_o",
        type=int,
        default=64,
        help="disentangle others dimension")
    parser.add_argument(
        "--dim_m",
        type=int,
        default=32,
        help="disentangle mouth dimension")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="num workers")
    # parser.add_argument(
    #     "--coord_dim",
    #     type=int,
    #     default=2,
    #     help="face coordnate dimension, 2 or 3",
    #     choices=[2, 3])


    args = parser.parse_args()
    return args


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

def test(dataloader, exp_dis_net, exp_cat_net, face_proj, lands_info):

    loss_data = Average_data()
    id = torch.zeros(1, 100).cuda()
    out_m_list = []
    out_o_list = []
    with tqdm(dataloader) as _tqdm:
        for exp_data, data_path in _tqdm:
            # print("data_path: ", data_path)
            bs = exp_data.shape[0]
            exp_data = exp_data.squeeze(0)
            exp_para = exp_data.cuda()

            out_o, out_m = exp_dis_net(exp_para)

            exp_out = exp_cat_net(out_o, out_m)

            geometry_out = face_proj.forward_geo_sub(
                id, exp_out, lands_info[-51:].long())
            geometry_ori = face_proj.forward_geo_sub(
                id, exp_para, lands_info[-51:].long())

            loss_lan = cal_lan_loss(
                geometry_out, geometry_ori)
            loss_data.update(loss_lan.item(), 1)
            _tqdm.set_postfix(OrderedDict(stage="test", loss=loss_data.avg))
    
            # print("out_o.shape: ", out_o.shape)
            # print("out_m.shape: ", out_m.shape)
            output_dict = {
                "exp_o": out_o.cpu(),
                "exp_m": out_m.cpu()
            }

            torch.save(output_dict, os.path.join(data_path[0], "face_params_dis_32_16.pt"))
    
    return output_dict

def test_LSR2_dataset(dataloader, exp_dis_net, exp_cat_net, dim_o, dim_m):

    loss_data = Average_data()
    id = torch.zeros(1, 100).cuda()

    with tqdm(dataloader) as _tqdm:
        for exp_data, exp_path in _tqdm:
            exp_para = exp_data.cuda()
            exp_para = exp_para.squeeze(0)
            # print("exp_para.shape: ", exp_para.shape)
            out_o, out_m = exp_dis_net(exp_para)
            out_o = out_o.cpu()
            out_m = out_m.cpu()
            # print("out_m.shape: ", out_m.shape)
            # print("exp_num: ", len(exp_num))
            output_dict = {
                "exp_o": out_o,
                "exp_m": out_m
            }

            # print("exp_num[i]: ", exp_num[i])
            # print("exp_path[0]: ", exp_path[0])
            torch.save(output_dict, os.path.join(exp_path[0], 'face_params_dis_{}_{}.pt'.format(dim_o, dim_m)))
    
    return True

# def test_LSR2_dataset(dataloader, exp_dis_net, exp_cat_net):

#     loss_data = Average_data()
#     id = torch.zeros(1, 100).cuda()

#     with tqdm(dataloader) as _tqdm:
#         for exp_data, exp_path, exp_num in _tqdm:
#             bs = exp_data.shape[0]
#             exp_para = exp_data.cuda()

#             out_o, out_m = exp_dis_net(exp_para)
#             out_o = out_o.cpu()
#             out_m = out_m.cpu()
#             # print("exp_num: ", len(exp_num))
#             cnt = 0
#             exp_path_processed = []
#             for i in range(bs):
#                 if exp_path[i] not in exp_path_processed:
#                     exp_path_processed.append(exp_path[i])
#                     idx = exp_num[i]
#                     print("cnt: ", cnt)
#                     print("cnt+idx: ", cnt+idx)
#                     print("out_m.shape: ", out_m.shape)
#                     output_dict = {
#                         "exp_o": out_o[cnt: cnt+idx],
#                         "exp_m": out_m[cnt: cnt+idx]
#                     }
#                     cnt += idx
#                     # print("exp_num[i]: ", exp_num[i])
#                     # print("exp_path[i]: ", exp_path[i])
#                     torch.save(output_dict, os.path.join(exp_path[i], 'face_params_dis.pt'))
    
#     return True


def main():
    # configuration
    args = parse_args()
    print("args: ", args)
    BS = 1
    NUM_WORKERS = 4
    # log_path = 'logs/train_all_withCycleVec_woVecLoss_202107161639'
    log_path = args.ckpt_path
    exp_dis_net_path = os.path.join(log_path, 'dis_ckpt.pth')
    exp_cat_net_path = os.path.join(log_path, 'cat_ckpt.pth')

    dis_ckpt = torch.load(exp_dis_net_path)
    cat_ckpt = torch.load(exp_cat_net_path)
    lands_info = np.loadtxt('./3DMM/lands_info.txt', dtype=np.int32)
    lands_info = torch.as_tensor(lands_info).cuda()
    
    # id, focal = data_loader.load_id(os.path.join(path,'static_params.json'))
    exp_dis_net = network.Distangler(79,args.dim_o,args.dim_m)
    exp_cat_net = network.Concatenater(args.dim_o,args.dim_m,79)
    device = 'cuda'
    exp_dis_net = exp_dis_net.to(device)
    exp_cat_net = exp_cat_net.to(device)
    exp_dis_net.load_state_dict(dis_ckpt['net'])
    exp_cat_net.load_state_dict(cat_ckpt['net'])
    exp_dis_net.eval()
    exp_cat_net.eval()

    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    face_proj = Face_3DMM('./3DMM',id_dim, exp_dim, tex_dim, point_num)
    best_loss = 100

    # load data
    # train_paths = ['./face_3dmm_params/obama/']
    test_paths = "/mnt/Diske/zhongruizhe/data/head_VAE/train_exp_params" # ['./face_3dmm_params/zhang_driven/']  # Mark_Zuck_old ysy_sjtu_talk JackMa_head
    test_dataset = FaceSJTUDataset(test_paths)
    # test_train_dataset = FaceLSR2DatasetTest('./dataset', 'train_0725', is_train=False)
    # test_val_dataset = FaceLSR2DatasetTest('./dataset', 'val_0725', is_train=False)
    # test_dataset = FaceLSR2DatasetTest('./dataset', 'val_0714', is_train=False)
    # test_dataset = FaceLSR2DatasetTest('./dataset', 'val_exp_0710', is_train=False)

    test_dataloader = DataLoader(
        test_dataset, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    # test_train_dataloader = DataLoader(
    #     test_train_dataset, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    # test_val_dataloader = DataLoader(
    #     test_val_dataset, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    with torch.no_grad():
        outputs = test(test_dataloader, exp_dis_net, exp_cat_net, face_proj, lands_info)
        # test_LSR2_dataset(test_train_dataloader, exp_dis_net, exp_cat_net, args.dim_o, args.dim_m)
        # test_LSR2_dataset(test_val_dataloader, exp_dis_net, exp_cat_net, args.dim_o, args.dim_m)
    # torch.save(outputs, os.path.join(log_path, "zhang_driven_disentangle_new_32_16.pt"))


if __name__ == '__main__':
    main()
