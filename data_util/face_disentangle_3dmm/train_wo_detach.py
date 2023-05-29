import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataloader import DataLoader
import network 
from util import *
from face_model import Face_3DMM
from data_loader import Face3dmmDataset
import argparse
import os
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
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
def train(epoch, dataloader, exp_dis_net, exp_cat_net, face_proj, 
          lands_info, optimizer_dis, optimizer_cat, scheduler_dis, scheduler_cat):

    loss_data = Average_data()
    id = torch.zeros(1, 100).cuda()
    tqdm_switch_mouth = tqdm(dataloader)
    for exp_data in tqdm_switch_mouth:
        bs = exp_data.shape[0]
        assert bs % 2 == 0, "Batch size must not be odd."
        half_bs = bs // 2
        exp_para_1 = exp_data[:half_bs].cuda()
        exp_para_2 = exp_data[half_bs:].cuda()

        # Without Detach
        # Swap mouth
        out_1_o, out_1_m = exp_dis_net(exp_para_1)
        out_2_o, out_2_m = exp_dis_net(exp_para_2)
        exp_out = exp_cat_net(out_1_o, out_2_m)
        # exp_out += exp_para_1
        geometry_mouth_swap_1 = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())
        with torch.no_grad():
            geometry_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
            geometry_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())

        loss_lan_others_ms1 = cal_lan_loss(
            geometry_mouth_swap_1[:, otherIdx, :2], geometry_1[:, otherIdx, :2])
        loss_lan_mouth_ms1 = cal_lan_loss(
            geometry_mouth_swap_1[:, mouthIdx, :2], geometry_2[:, mouthIdx, :2])
        
        exp_out = exp_cat_net(out_2_o, out_1_m)
        # exp_out += exp_para_2
        geometry_mouth_swap_2 = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())

        loss_lan_others_ms2 = cal_lan_loss(
            geometry_mouth_swap_2[:, otherIdx, :2], geometry_2[:, otherIdx, :2])
        loss_lan_mouth_ms2 = cal_lan_loss(
            geometry_mouth_swap_2[:, mouthIdx, :2], geometry_1[:, mouthIdx, :2])

        
        loss_lan = loss_lan_others_ms1 + loss_lan_mouth_ms1 + \
            loss_lan_others_ms2 + loss_lan_mouth_ms2
        loss_data.update(loss_lan.item(), 1)
        tqdm_switch_mouth.set_postfix(OrderedDict(stage="train_switch_mouth", epoch=epoch, loss=loss_data.avg))
        optimizer_dis.zero_grad()
        optimizer_cat.zero_grad()
        loss_lan.backward()
        optimizer_dis.step()
        optimizer_cat.step()

    loss_data = Average_data()
    id = torch.zeros(1, 100).cuda()
    tqdm_switch_others = tqdm(dataloader)
    for exp_data in tqdm_switch_others:
        bs = exp_data.shape[0]
        assert bs % 2 == 0, "Batch size must not be odd."
        half_bs = bs // 2
        exp_para_1 = exp_data[:half_bs].cuda()
        exp_para_2 = exp_data[half_bs:].cuda()

        # Swap others
        out_1_o, out_1_m = exp_dis_net(exp_para_1)
        out_2_o, out_2_m = exp_dis_net(exp_para_2)
        exp_out = exp_cat_net(out_2_o, out_1_m)
        # exp_out += exp_para_1
        geometry_mouth_swap_1 = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())
        with torch.no_grad():
            geometry_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
            geometry_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())

        loss_lan_others_ms1 = cal_lan_loss(
            geometry_mouth_swap_1[:, otherIdx, :2], geometry_2[:, otherIdx, :2])
        loss_lan_mouth_ms1 = cal_lan_loss(
            geometry_mouth_swap_1[:, mouthIdx, :2], geometry_1[:, mouthIdx, :2])
        
        exp_out = exp_cat_net(out_1_o, out_2_m)
        # exp_out += exp_para_2
        geometry_mouth_swap_2 = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())

        loss_lan_others_ms2 = cal_lan_loss(
            geometry_mouth_swap_2[:, otherIdx, :2], geometry_1[:, otherIdx, :2])
        loss_lan_mouth_ms2 = cal_lan_loss(
            geometry_mouth_swap_2[:, mouthIdx, :2], geometry_2[:, mouthIdx, :2])

        
        loss_lan = loss_lan_others_ms1 + loss_lan_mouth_ms1 + \
            loss_lan_others_ms2 + loss_lan_mouth_ms2
        loss_data.update(loss_lan.item(), 1)
        tqdm_switch_others.set_postfix(OrderedDict(stage="train_switch_others", epoch=epoch, loss=loss_data.avg))
        optimizer_dis.zero_grad()
        optimizer_cat.zero_grad()
        loss_lan.backward()
        optimizer_dis.step()
        optimizer_cat.step()

    scheduler_cat.step()
    scheduler_dis.step()
    return exp_dis_net, exp_cat_net

def test(epoch, dataloader, exp_dis_net, exp_cat_net, face_proj, lands_info, best_loss):

    loss_data = Average_data()
    id = torch.zeros(1, 100).cuda()
    with tqdm(dataloader) as _tqdm:
        for exp_data in _tqdm:
            bs = exp_data.shape[0]
            assert bs % 2 == 0, "Batch size must not be odd."
            half_bs = bs // 2
            exp_para_1 = exp_data[:half_bs].cuda()
            exp_para_2 = exp_data[half_bs:].cuda()

            out_1_o, out_1_m = exp_dis_net(exp_para_1)
            out_2_o, out_2_m = exp_dis_net(exp_para_2)
            exp_out_mouth_swap_1 = exp_cat_net(out_1_o, out_2_m)
            exp_out_mouth_swap_2 = exp_cat_net(out_2_o, out_1_m)
            # exp_out_mouth_swap_1 += exp_para_1
            # exp_out_mouth_swap_2 += exp_para_2
            geometry_mouth_swap_1 = face_proj.forward_geo_sub(
                id, exp_out_mouth_swap_1, lands_info[-51:].long())
            geometry_mouth_swap_2 = face_proj.forward_geo_sub(
                id, exp_out_mouth_swap_2, lands_info[-51:].long())
            geometry_1 = face_proj.forward_geo_sub(
                id, exp_para_1, lands_info[-51:].long())
            geometry_2 = face_proj.forward_geo_sub(
                id, exp_para_2, lands_info[-51:].long())

            loss_lan_others_ms1 = cal_lan_loss(
                geometry_mouth_swap_1[:, otherIdx, :2], geometry_1[:, otherIdx, :2])
            loss_lan_mouth_ms1 = cal_lan_loss(
                geometry_mouth_swap_1[:, mouthIdx, :2], geometry_2[:, mouthIdx, :2])
            loss_lan_others_ms2 = cal_lan_loss(
                geometry_mouth_swap_2[:, otherIdx, :2], geometry_2[:, otherIdx, :2])
            loss_lan_mouth_ms2 = cal_lan_loss(
                geometry_mouth_swap_2[:, mouthIdx, :2], geometry_1[:, mouthIdx, :2])
            loss_lan = loss_lan_others_ms1 + loss_lan_mouth_ms1 + \
                loss_lan_others_ms2 + loss_lan_mouth_ms2
            loss_data.update(loss_lan.item(), 1)
            _tqdm.set_postfix(OrderedDict(stage="test", epoch=epoch, loss=loss_data.avg)) 
    print('In the epoch %d, the average loss is %f.'%(epoch,loss_data.avg))
    if best_loss > loss_data.avg:
        best_loss = loss_data.avg
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state_dis = {
            'net': exp_dis_net.state_dict(),
            'epoch': epoch
        }
        torch.save(state_dis, './checkpoint/dis_ckpt.pth')
        state_cat = {
            'net': exp_cat_net.state_dict(),
            'epoch': epoch
        }
        torch.save(state_cat, './checkpoint/cat_ckpt.pth')
    
    return best_loss


def main():
    # configuration
    BS = 256
    NUM_WORKERS = 4
    train_epoch = 300
    TRAIN_LR = 0.001
    TRAIN_MOMENTUM = 0.9
    TRAIN_WEIGHT_DECAY = 5e-4
    TRAIN_LR_DECAY_STEP = 250
    TRAIN_LR_DECAY_RATE = 0.1
    lands_info = np.loadtxt('./3DMM/lands_info.txt', dtype=np.int32)
    lands_info = torch.as_tensor(lands_info).cuda()
    
    # id, focal = data_loader.load_id(os.path.join(path,'static_params.json'))
    exp_dis_net = network.Distangler(79,128,64)
    exp_cat_net = network.Concatenater(128,64,79)
    device = 'cuda'
    exp_dis_net = exp_dis_net.to(device)
    exp_cat_net = exp_cat_net.to(device)
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    face_proj = Face_3DMM('./3DMM',id_dim, exp_dim, tex_dim, point_num)
    optimizer_dis = optim.SGD(exp_dis_net.parameters(), lr=TRAIN_LR,momentum=TRAIN_MOMENTUM,weight_decay=TRAIN_WEIGHT_DECAY)
    optimizer_cat = optim.SGD(exp_cat_net.parameters(), lr=TRAIN_LR,momentum=TRAIN_MOMENTUM,weight_decay=TRAIN_WEIGHT_DECAY)
    scheduler_dis = StepLR(optimizer_dis, step_size=TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE)
    scheduler_cat = StepLR(optimizer_cat, step_size=TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE)
    best_loss = 100

    # load data
    train_paths = ['./face_3dmm_params/Mark_Zuck_old/', './face_3dmm_params/obama/', './face_3dmm_params/ysy_sjtu_talk']
    test_paths = ['./face_3dmm_params/obama/']
    train_dataset = Face3dmmDataset(train_paths)
    test_dataset = Face3dmmDataset(test_paths)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BS, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

    for epoch in range(0, train_epoch):
        print('Epoch: %d.'%(epoch))
        train(epoch, train_dataloader, exp_dis_net, exp_cat_net, face_proj,
              lands_info, optimizer_dis, optimizer_cat, scheduler_dis, scheduler_cat)
        with torch.no_grad():
            best_loss = test(epoch, test_dataloader, exp_dis_net, exp_cat_net, face_proj, 
                lands_info, best_loss)


if __name__ == '__main__':
    main()
