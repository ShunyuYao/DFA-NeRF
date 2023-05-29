import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import network 
from util import *
from face_model import Face_3DMM
import data_loader 
import argparse
import os
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
import pandas as pd
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
def train(epoch,id,focal,exp_list,exp_dis,exp_cat,face_proj,lands_info,optimizer_dis,optimizer_cat,scheduler_dis,scheduler_cat):
    rounds = 1000
    length = len(exp_list)
    loss_data = Average_data()
    id = id.cuda()
    focal = torch.as_tensor(focal).cuda()
    with tqdm(range(rounds)) as _tqdm:
        for round in _tqdm:
            cho_1 = random.randint(0,length-1)
            cho_2 = random.randint(0,length-1)
            while cho_1 == cho_2:
                cho_2 = random.randint(0,length-1)
            exp_para_1 = torch.as_tensor(exp_list[cho_1]).reshape((1,79)).cuda()
            exp_para_2 = torch.as_tensor(exp_list[cho_2]).reshape((1,79)).cuda()
            # detach m1 side
            out_1_1,out_1_2 = exp_dis(exp_para_1)
            out_1_2.detach()
            with torch.no_grad():
                _,out_2 = exp_dis(exp_para_2)
            exp_out = exp_cat(out_1_1,out_2)
            #exp_out += exp_para_1
            geometry_with_expNet_1 = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())
            with torch.no_grad():
                geometry_without_expNet_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
            loss_lan_others = cal_lan_loss(geometry_with_expNet_1[:, otherIdx, :2], geometry_without_expNet_1[:, otherIdx, :2])
            # detach o1 side
            out_2_1,out_2_2 = exp_dis(exp_para_2)
            out_2_1.detach()
            with torch.no_grad():
                out_1,_ = exp_dis(exp_para_1)
            exp_out = exp_cat(out_1,out_2_2)
            #exp_out += exp_para_1
            geometry_with_expNet_2 = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())
            with torch.no_grad():
                geometry_without_expNet_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())
            loss_lan_mouth = cal_lan_loss(geometry_with_expNet_2[:, mouthIdx, :2], geometry_without_expNet_2[:, mouthIdx, :2])
            loss_lan = loss_lan_mouth+loss_lan_others
            loss_data.update(loss_lan.item(), 1)
            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_data.avg),sample_num=round+1) 
            optimizer_dis.zero_grad()
            optimizer_cat.zero_grad()
            loss_lan.backward()
            optimizer_dis.step()
            optimizer_cat.step()
    scheduler_cat.step()
    scheduler_dis.step()

def test(epoch,id,focal,exp_list,exp_dis,exp_cat,face_proj,lands_info,best_loss):
    rounds = 200
    length = len(exp_list)
    loss_data = Average_data()
    id = id.cuda()
    focal = torch.as_tensor(focal).cuda()
    with torch.no_grad():
        with tqdm(range(rounds)) as _tqdm:
            for round in _tqdm:
                cho_1 = random.randint(0,length-1)
                cho_2 = random.randint(0,length-1)
                while cho_1 == cho_2:
                    cho_2 = random.randint(0,length-1)
                exp_para_1 = torch.as_tensor(exp_list[cho_1]).reshape((1,79)).cuda()
                exp_para_2 = torch.as_tensor(exp_list[cho_2]).reshape((1,79)).cuda()
                out_1,_ = exp_dis(exp_para_1)
                _,out_2 = exp_dis(exp_para_2)
                exp_out = exp_cat(out_1,out_2)
                #exp_out += exp_para_1
                geometry_with_expNet = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())
                geometry_without_expNet_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
                geometry_without_expNet_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())
                loss_lan_mouth = cal_lan_loss(geometry_with_expNet[:, mouthIdx, :2], geometry_without_expNet_2[:, mouthIdx, :2])
                loss_lan_others = cal_lan_loss(geometry_with_expNet[:, otherIdx, :2], geometry_without_expNet_1[:, otherIdx, :2])
                loss_lan = loss_lan_mouth + loss_lan_others
                loss_data.update(loss_lan.item(), 1)
                _tqdm.set_postfix(OrderedDict(stage="test", epoch=epoch, loss=loss_data.avg),sample_num=round+1) 
    print('In the epoch %d, the average loss is %f.'%(epoch,loss_data.avg))
    if best_loss>loss_data.avg:
        best_loss = loss_data.avg
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state_dis = {
            'net': exp_dis.state_dict(),
            'epoch': epoch
        }
        torch.save(state_dis, './checkpoint/dis_with_detach_ckpt.pth')
        state_cat = {
            'net': exp_cat.state_dict(),
            'epoch': epoch
        }
        torch.save(state_cat, './checkpoint/cat_with_detach_ckpt.pth')

        


def main():
    # configuration
    train_epoch = 50
    TRAIN_LR = 0.05
    TRAIN_MOMENTUM = 0.9
    TRAIN_WEIGHT_DECAY = 5e-4
    TRAIN_LR_DECAY_STEP = 20
    TRAIN_LR_DECAY_RATE = 0.1
    lands_info = np.loadtxt('./3DMM/lands_info.txt', dtype=np.int32)
    lands_info = torch.as_tensor(lands_info).cuda()
    # load data
    path = './face_3dmm_params/obama/'
    _,focal = data_loader.load_id(os.path.join(path,'static_params.json'))
    exp_list,_,_ = data_loader.load_exp(path)
    exp_dis = network.Distangler(79,128,64)
    exp_cat = network.Concatenater(128,64,79)
    device = 'cuda'
    exp_dis = exp_dis.to(device)
    exp_cat = exp_cat.to(device)
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    id = torch.zeros([1, id_dim],dtype = torch.float)
    face_proj = Face_3DMM('./3DMM',id_dim, exp_dim, tex_dim, point_num)
    optimizer_dis = optim.SGD(exp_dis.parameters(), lr=TRAIN_LR,momentum=TRAIN_MOMENTUM,weight_decay=TRAIN_WEIGHT_DECAY)
    optimizer_cat = optim.SGD(exp_cat.parameters(), lr=TRAIN_LR,momentum=TRAIN_MOMENTUM,weight_decay=TRAIN_WEIGHT_DECAY)
    scheduler_dis = StepLR(optimizer_dis, step_size=TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE)
    scheduler_cat = StepLR(optimizer_cat, step_size=TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE)
    best_loss = 100
    for epoch in range(1,train_epoch+1):
        print('Epoch: %d.'%(epoch))
        train(epoch,id,focal,exp_list,exp_dis,exp_cat,face_proj,lands_info,optimizer_dis,optimizer_cat,scheduler_dis,scheduler_cat)
        test(epoch,id,focal,exp_list,exp_dis,exp_cat,face_proj,lands_info,best_loss)


if __name__ == '__main__':
    main()
