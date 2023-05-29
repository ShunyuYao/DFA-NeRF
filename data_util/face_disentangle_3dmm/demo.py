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
import cv2

def main():
    lands_info = np.loadtxt('./3DMM/lands_info.txt', dtype=np.int32)
    lands_info = torch.as_tensor(lands_info).cuda()
    # load data
    path = './face_3dmm_params/obama/'
    id,focal = data_loader.load_id(os.path.join(path,'static_params.json'))
    exp_list,_,_ = data_loader.load_exp(path)
    exp_dis = network.Distangler(79,128,64)
    exp_cat = network.Concatenater(128,64,79)
    device = 'cuda'
    exp_dis = exp_dis.to(device)
    exp_cat = exp_cat.to(device)
    checkpoint_dis = torch.load('./checkpoint/dis_ckpt.pth')
    checkpoint_cat = torch.load('./checkpoint/cat_ckpt.pth')
    exp_dis.load_state_dict(checkpoint_dis['net'])
    exp_cat.load_state_dict(checkpoint_cat['net'])
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    face_proj = Face_3DMM('./3DMM',id_dim, exp_dim, tex_dim, point_num)
    color = (0,0,255)
    radius = 5
    thickness = 5
    length = len(exp_list)
    id = torch.as_tensor(id).cuda()
    focal = torch.as_tensor(focal).cuda()
    if not os.path.isdir('demo_results'):
        os.mkdir('demo_results')
    with torch.no_grad():
        cho_1 = random.randint(0,length-1)
        cho_2 = random.randint(0,length-1)
        while cho_1 == cho_2:
            cho_2 = random.randint(0,length-1)
        exp_para_1 = torch.as_tensor(exp_list[cho_1]).reshape((1,79)).cuda()
        exp_para_2 = torch.as_tensor(exp_list[cho_2]).reshape((1,79)).cuda()
        out_1_1,out_1_2 = exp_dis(exp_para_1)
        out_1_2.detach()
        _,out_2 = exp_dis(exp_para_2)
        exp_out = exp_cat(out_1_1,out_2)
        exp_out += exp_para_1
        geometry_with_expNet = face_proj.forward_geo_sub(id, exp_out, lands_info[-51:].long())
        geometry_without_expNet_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
        geometry_without_expNet_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())
        geometry_with_expNet_list = geometry_with_expNet[:,:,:2].view(51,2).tolist()
        img_with_expNet = np.zeros((2000,2000),np.uint8)
        img_with_expNet.fill(255)
        for point in geometry_with_expNet_list:
            point[0] += 100
            point[1] += 100
            point[0] *= 10
            point[1] *= 10
            point[0] = round(point[0])
            point[1] = round(point[1])
            point = tuple(point)
            cv2.circle(img_with_expNet,point,radius,color,thickness)
        cv2.imwrite('./demo_results/img_with_expNet.png',img_with_expNet)
        geometry_without_expNet_1_list = geometry_without_expNet_1[:,:,:2].view(51,2).tolist()
        img_without_expNet_1 = np.zeros((2000,2000),np.uint8)
        img_without_expNet_1.fill(255)
        for point in geometry_without_expNet_1_list:
            point[0] += 100
            point[1] += 100
            point[0] *= 10
            point[1] *= 10
            point[0] = round(point[0])
            point[1] = round(point[1])
            point = tuple(point)
            cv2.circle(img_without_expNet_1,point,radius,color,thickness)
        cv2.imwrite('./demo_results/img_without_expNet_1.png',img_without_expNet_1)
        geometry_without_expNet_2_list = geometry_without_expNet_2[:,:,:2].view(51,2).tolist()
        img_without_expNet_2 = np.zeros((2000,2000),np.uint8)
        img_without_expNet_2.fill(255)
        for point in geometry_without_expNet_2_list:
            point[0] += 100
            point[1] += 100
            point[0] *= 10
            point[1] *= 10
            point[0] = round(point[0])
            point[1] = round(point[1])
            point = tuple(point)
            cv2.circle(img_without_expNet_2,point,radius,color,thickness)
        cv2.imwrite('./demo_results/img_without_expNet_2.png',img_without_expNet_2)


if __name__ == '__main__':
    main()
