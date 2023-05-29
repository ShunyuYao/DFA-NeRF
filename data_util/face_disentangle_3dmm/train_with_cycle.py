import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataloader import DataLoader
import network
from utils.util import *
from utils.HyperSaver import HyperSaver
from face_model import Face_3DMM
from data_loader import Face3dmmDataset, FaceLSR2Dataset
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
        "--use_exp_loss",
        action='store_true',
        help='use expression code for loss'
    )
    parser.add_argument(
        "--use_vec_loss",
        action='store_true',
        help='use vector consistence for loss'
    )
    parser.add_argument(
        "--not_use_cycle_loss",
        action='store_true',
        help='do not use cycle consistence for loss'
    )
    parser.add_argument(
        "--self_rec_rate",
        type=float,
        default=0.,
        help='self reconstruction method'
    )

    args = parser.parse_args()
    return args


def train(epoch, args, dataloader, exp_dis_net, exp_cat_net, face_proj, 
          lands_info, optimizer_dis, optimizer_cat, scheduler_dis, scheduler_cat):

    loss_data = Average_data()
    loss_exchange = Average_data()
    loss_back = Average_data()
    loss_exp = Average_data()
    loss_vec = Average_data()
    id = torch.zeros(1, 100).cuda()
    tqdm_switch_mouth = tqdm(dataloader)
    coord_dim = args.coord_dim
    loss_exps = 0
    loss_l1 = nn.L1Loss()
    for exp_data in tqdm_switch_mouth:
        bs = exp_data.shape[0]
        assert bs % 2 == 0, "Batch size must not be odd."
        if args.self_rec_rate > 0 and random.random() < args.self_rec_rate:
            # print("train self reconstruction")
            exp_para_1 = exp_data.cuda()
            exp_para_2 = exp_data.clone().cuda()
        else:
            half_bs = bs // 2
            exp_para_1 = exp_data[:half_bs].cuda()
            exp_para_2 = exp_data[half_bs:].cuda()

        # Without Detach
        # Swap mouth
        out_1_o, out_1_m = exp_dis_net(exp_para_1)
        out_2_o, out_2_m = exp_dis_net(exp_para_2)
        exp_out_1o_2m = exp_cat_net(out_1_o, out_2_m)
        # exp_out += exp_para_1
        geometry_mouth_swap_1 = face_proj.forward_geo_sub(id, exp_out_1o_2m, lands_info[-51:].long())
        with torch.no_grad():
            geometry_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
            geometry_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())

        loss_lan_others_ms1 = cal_lan_loss(
            geometry_mouth_swap_1[:, otherIdx, :coord_dim], geometry_1[:, otherIdx, :coord_dim])
        loss_lan_mouth_ms1 = cal_lan_loss(
            geometry_mouth_swap_1[:, mouthIdx, :coord_dim], geometry_2[:, mouthIdx, :coord_dim])

        exp_out_2o_1m = exp_cat_net(out_2_o, out_1_m)
        # exp_out += exp_para_2
        geometry_mouth_swap_2 = face_proj.forward_geo_sub(id, exp_out_2o_1m, lands_info[-51:].long())

        loss_lan_others_ms2 = cal_lan_loss(
            geometry_mouth_swap_2[:, otherIdx, :coord_dim], geometry_2[:, otherIdx, :coord_dim])
        loss_lan_mouth_ms2 = cal_lan_loss(
            geometry_mouth_swap_2[:, mouthIdx, :coord_dim], geometry_1[:, mouthIdx, :coord_dim])
        
        loss_lan = loss_lan_others_ms1 + loss_lan_mouth_ms1 + \
            loss_lan_others_ms2 + loss_lan_mouth_ms2
            
        loss_exchange.update(loss_lan.item(), 1)
        optimizer_dis.zero_grad()
        optimizer_cat.zero_grad()
        # loss_lan.backward(retain_graph=True)
        optimizer_dis.step()
        optimizer_cat.step()
        
        # cycle back
        loss_cycle = 0
        loss_total = 0
        if not args.not_use_cycle_loss:
            exp_out_1o, exp_out_2m = exp_dis_net(exp_out_1o_2m)
            exp_out_2o, exp_out_1m = exp_dis_net(exp_out_2o_1m)

            exp_back_1 = exp_cat_net(exp_out_1o, exp_out_1m)
            exp_back_2 = exp_cat_net(exp_out_2o, exp_out_2m)
            
            geometry_exp_back_1 = face_proj.forward_geo_sub(id, exp_back_1, lands_info[-51:].long())
            geometry_exp_back_2 = face_proj.forward_geo_sub(id, exp_back_2, lands_info[-51:].long())
            loss_lan_back_exp_1 = cal_lan_loss(
                geometry_exp_back_1[:, :, :coord_dim], geometry_1[:, :, :coord_dim])
            loss_lan_back_exp_2 = cal_lan_loss(
                geometry_exp_back_2[:, :, :coord_dim], geometry_2[:, :, :coord_dim])
        
            if args.use_vec_loss:
                loss_vec_1o = loss_l1(exp_out_1o, out_1_o)
                loss_vec_2o = loss_l1(exp_out_2o, out_2_o)
                loss_vec_1m = loss_l1(exp_out_1m, out_1_m)
                loss_vec_2m = loss_l1(exp_out_2m, out_2_m)
                loss_vecs = loss_vec_1o + loss_vec_2o + loss_vec_1m + loss_vec_2m
                loss_cycle += loss_vecs
                loss_vec.update(loss_vecs.item(), 1)

            if args.use_exp_loss:
                loss_exp1 = cal_lan_loss(exp_para_1, exp_back_1)
                loss_exp2 = cal_lan_loss(exp_para_2, exp_back_2)
                loss_exps = loss_exp1 + loss_exp2
                loss_cycle += loss_exps
                loss_exp.update(loss_exps.item(), 1)

            loss_cycle += loss_lan_back_exp_1 + loss_lan_back_exp_2

            loss_back.update(loss_cycle.item(), 1)
            optimizer_dis.zero_grad()
            optimizer_cat.zero_grad()
            loss_cycle.backward()
            optimizer_dis.step()
            optimizer_cat.step()

        loss_total += loss_lan + loss_cycle
        loss_data.update(loss_total.item(), 1)
        
        if args.use_exp_loss:
            tqdm_switch_mouth.set_postfix(OrderedDict(stage="train_switch_mouth", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg, loss_exp=loss_exp.avg))
        elif args.use_vec_loss:
            tqdm_switch_mouth.set_postfix(OrderedDict(stage="test", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg, loss_vec=loss_vec.avg))
        else:
            tqdm_switch_mouth.set_postfix(OrderedDict(stage="train_switch_mouth", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg))

    scheduler_cat.step()
    scheduler_dis.step()
    losses = [loss_data, loss_exchange, loss_back, loss_exp, loss_vec]
    return losses


# def train_others_exchange(epoch, args, dataloader, exp_dis_net, exp_cat_net, face_proj, 
#           lands_info, optimizer_dis, optimizer_cat, scheduler_dis, scheduler_cat):
#     loss_data = Average_data()
#     loss_exchange = Average_data()
#     loss_back = Average_data()
#     if args.use_exp_loss:
#         loss_exp = Average_data()
#     id = torch.zeros(1, 100).cuda()
#     tqdm_switch_mouth = tqdm(dataloader)
#     coord_dim = args.coord_dim
#     for exp_data in tqdm_switch_mouth:
#         bs = exp_data.shape[0]
#         assert bs % 2 == 0, "Batch size must not be odd."
#         half_bs = bs // 2
#         exp_para_1 = exp_data[:half_bs].cuda()
#         exp_para_2 = exp_data[half_bs:].cuda()

#         # Without Detach
#         # Swap mouth
#         out_1_o, out_1_m = exp_dis_net(exp_para_1)
#         out_2_o, out_2_m = exp_dis_net(exp_para_2)
#         exp_out_2o_1m = exp_cat_net(out_2_o, out_1_m)
#         # exp_out += exp_para_1
#         geometry_mouth_swap_1 = face_proj.forward_geo_sub(id, exp_out_2o_1m, lands_info[-51:].long())
#         with torch.no_grad():
#             geometry_1 = face_proj.forward_geo_sub(id, exp_para_1, lands_info[-51:].long())
#             geometry_2 = face_proj.forward_geo_sub(id, exp_para_2, lands_info[-51:].long())

#         loss_lan_others_ms1 = cal_lan_loss(
#             geometry_mouth_swap_1[:, otherIdx, :coord_dim], geometry_2[:, otherIdx, :coord_dim])
#         loss_lan_mouth_ms1 = cal_lan_loss(
#             geometry_mouth_swap_1[:, mouthIdx, :coord_dim], geometry_1[:, mouthIdx, :coord_dim])

#         exp_out_1o_2m = exp_cat_net(out_1_o, out_2_m)
#         # exp_out += exp_para_2
#         geometry_mouth_swap_2 = face_proj.forward_geo_sub(id, exp_out_1o_2m, lands_info[-51:].long())

#         loss_lan_others_ms2 = cal_lan_loss(
#             geometry_mouth_swap_2[:, otherIdx, :coord_dim], geometry_1[:, otherIdx, :coord_dim])
#         loss_lan_mouth_ms2 = cal_lan_loss(
#             geometry_mouth_swap_2[:, mouthIdx, :coord_dim], geometry_2[:, mouthIdx, :coord_dim])
        
#         loss_lan = loss_lan_others_ms1 + loss_lan_mouth_ms1 + \
#             loss_lan_others_ms2 + loss_lan_mouth_ms2
            
#         loss_exchange.update(loss_lan.item(), 1)
#         optimizer_dis.zero_grad()
#         optimizer_cat.zero_grad()
#         loss_lan.backward(retain_graph=True)
#         optimizer_dis.step()
#         optimizer_cat.step()
        
#         # cycle back
#         loss_cycle = 0
#         exp_out_1o, exp_out_2m = exp_dis_net(exp_out_1o_2m)
#         exp_out_2o, exp_out_1m = exp_dis_net(exp_out_2o_1m)
#         exp_back_1 = exp_cat_net(exp_out_1o, exp_out_1m)
#         exp_back_2 = exp_cat_net(exp_out_2o, exp_out_2m)
        
#         geometry_exp_back_1 = face_proj.forward_geo_sub(id, exp_back_1, lands_info[-51:].long())
#         geometry_exp_back_2 = face_proj.forward_geo_sub(id, exp_back_2, lands_info[-51:].long())
#         loss_lan_back_exp_1 = cal_lan_loss(
#             geometry_exp_back_1[:, :, :coord_dim], geometry_1[:, :, :coord_dim])
#         loss_lan_back_exp_2 = cal_lan_loss(
#             geometry_exp_back_2[:, :, :coord_dim], geometry_2[:, :, :coord_dim])
        
#         if args.use_exp_loss:
#             loss_exp1 = cal_lan_loss(exp_para_1, exp_back_1)
#             loss_exp2 = cal_lan_loss(exp_para_2, exp_back_2)
#             loss_exps = loss_exp1 + loss_exp2
#             loss_cycle += loss_exps
#             loss_exp.update(loss_exps.item(), 1)

#         loss_cycle += loss_lan_back_exp_1 + loss_lan_back_exp_2

#         loss_back.update(loss_cycle.item(), 1)
#         optimizer_dis.zero_grad()
#         optimizer_cat.zero_grad()
#         loss_cycle.backward()
#         optimizer_dis.step()
#         optimizer_cat.step()

#         loss_total = loss_lan + loss_cycle
#         loss_data.update(loss_total.item(), 1)
        
#         if args.use_exp_loss:
#             tqdm_switch_mouth.set_postfix(OrderedDict(stage="train_switch_mouth", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg, loss_exp=loss_exps.avg))
#         else:
#             tqdm_switch_mouth.set_postfix(OrderedDict(stage="train_switch_mouth", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg))

#     scheduler_cat.step()
#     scheduler_dis.step()
#     losses = [loss_data, loss_exchange, loss_back]
#     if args.use_exp_loss:
#         losses = losses + [loss_exps]
#     return losses

def test(epoch, args, save_path, dataloader, exp_dis_net, exp_cat_net, face_proj, lands_info, best_loss):

    loss_data = Average_data()
    loss_exchange = Average_data()
    loss_back = Average_data()
    loss_exp = Average_data()
    loss_vec = Average_data()
    id = torch.zeros(1, 100).cuda()
    _tqdm = tqdm(dataloader)
    coord_dim = args.coord_dim
    loss_exps = 0
    loss_l1 = nn.L1Loss()
    # with tqdm(dataloader) as _tqdm:
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
        
        loss_exchange.update(loss_lan.item(), 1)

        # cycle back
        loss_cycle = 0
        loss_total = 0
        if not args.not_use_cycle_loss:
            exp_out_1o, exp_out_2m = exp_dis_net(exp_out_mouth_swap_1)
            exp_out_2o, exp_out_1m = exp_dis_net(exp_out_mouth_swap_2)
            exp_back_1 = exp_cat_net(exp_out_1o, exp_out_1m)
            exp_back_2 = exp_cat_net(exp_out_2o, exp_out_2m)
            
            geometry_exp_back_1 = face_proj.forward_geo_sub(id, exp_back_1, lands_info[-51:].long())
            geometry_exp_back_2 = face_proj.forward_geo_sub(id, exp_back_2, lands_info[-51:].long())
            loss_lan_back_exp_1 = cal_lan_loss(
                geometry_exp_back_1[:, :, :coord_dim], geometry_1[:, :, :coord_dim])
            loss_lan_back_exp_2 = cal_lan_loss(
                geometry_exp_back_2[:, :, :coord_dim], geometry_2[:, :, :coord_dim])
            
            loss_cycle += loss_lan_back_exp_1 + loss_lan_back_exp_2
            loss_back.update(loss_cycle.item(), 1)
            
            if args.use_vec_loss:
                loss_vec_1o = loss_l1(exp_out_1o, out_1_o)
                loss_vec_2o = loss_l1(exp_out_2o, out_2_o)
                loss_vec_1m = loss_l1(exp_out_1m, out_1_m)
                loss_vec_2m = loss_l1(exp_out_2m, out_2_m)
                loss_vecs = loss_vec_1o + loss_vec_2o + loss_vec_1m + loss_vec_2m
                loss_cycle += loss_vecs
                loss_vec.update(loss_vecs.item(), 1)
            
            if args.use_exp_loss:
                loss_exp1 = cal_lan_loss(exp_para_1, exp_back_1)
                loss_exp2 = cal_lan_loss(exp_para_2, exp_back_2)
                loss_exps = loss_exp1 + loss_exp2
                loss_cycle += loss_exps
                loss_total += loss_exps
                loss_exp.update(loss_exps.item(), 1)
        loss_total += loss_lan + loss_cycle
        loss_data.update(loss_total.item(), 1)
    
        if args.use_exp_loss:
            _tqdm.set_postfix(OrderedDict(stage="test", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg, loss_exp=loss_exp.avg))
        elif args.use_vec_loss:
            _tqdm.set_postfix(OrderedDict(stage="test", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg, loss_vec=loss_vec.avg))
        else:
            _tqdm.set_postfix(OrderedDict(stage="test", epoch=epoch, loss_all=loss_data.avg, loss_exchange=loss_exchange.avg, loss_back=loss_back.avg))


    print('In the epoch %d, the average loss is %f.'%(epoch,loss_data.avg))
    if best_loss > loss_data.avg:
        best_loss = loss_data.avg
        print('Saving..')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        state_dis = {
            'net': exp_dis_net.state_dict(),
            'epoch': epoch
        }
        torch.save(state_dis, os.path.join(save_path, 'dis_ckpt.pth'))
        state_cat = {
            'net': exp_cat_net.state_dict(),
            'epoch': epoch
        }
        torch.save(state_cat, os.path.join(save_path, 'cat_ckpt.pth'))
        losses = [loss_data, loss_exchange, loss_back, loss_exp, loss_vec, True]
    else:
        losses = [loss_data, loss_exchange, loss_back, loss_exp, loss_vec, False]
    return losses


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
    lands_info = np.loadtxt('./3DMM/lands_info.txt', dtype=np.int32)
    lands_info = torch.as_tensor(lands_info).cuda()
    
    # id, focal = data_loader.load_id(os.path.join(path,'static_params.json'))
    exp_dis_net = network.Distangler(79, args.dim_o, args.dim_m) 
    exp_cat_net = network.Concatenater(args.dim_o, args.dim_m, 79)
    device = 'cuda'
    exp_dis_net = exp_dis_net.to(device)
    exp_cat_net = exp_cat_net.to(device)
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    face_proj = Face_3DMM('./3DMM',id_dim, exp_dim, tex_dim, point_num)
    optimizer_dis = optim.SGD(exp_dis_net.parameters(), lr=TRAIN_LR,momentum=TRAIN_MOMENTUM,weight_decay=TRAIN_WEIGHT_DECAY)
    optimizer_cat = optim.SGD(exp_cat_net.parameters(), lr=TRAIN_LR,momentum=TRAIN_MOMENTUM,weight_decay=TRAIN_WEIGHT_DECAY)
    scheduler_dis = MultiStepLR(optimizer_dis, TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE) # , verbose=True
    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, threshold=0.001)
    # StepLR(optimizer_dis, step_size=TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE)
    scheduler_cat = MultiStepLR(optimizer_cat, TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE) # , verbose=True
    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, threshold=0.001)
    # StepLR(optimizer_cat, step_size=TRAIN_LR_DECAY_STEP, gamma=TRAIN_LR_DECAY_RATE)
    best_loss = 100

    # load data
    # train_paths = ['./face_3dmm_params/Mark_Zuck_old/', './face_3dmm_params/ysy_sjtu_talk', './face_3dmm_params/obama/'] # './face_3dmm_params/obama/'
    # test_paths = ['./face_3dmm_params/obama/']
    # train_dataset = Face3dmmDataset(train_paths)
    # test_dataset = Face3dmmDataset(test_paths)
    train_dataset = FaceLSR2Dataset('./dataset', 'train_0714')
    test_dataset = FaceLSR2Dataset('./dataset', 'val_0714')

    train_dataloader = DataLoader(
        train_dataset, batch_size=BS, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

    for epoch in range(0, train_epoch):
        print('Epoch: %d.'%(epoch))
        with torch.autograd.set_detect_anomaly(True):
            train_losses = train(epoch, args, train_dataloader, exp_dis_net, exp_cat_net, face_proj,
                lands_info, optimizer_dis, optimizer_cat, scheduler_dis, scheduler_cat)
        with torch.no_grad():
            test_loss = test(epoch, args, log_path, test_dataloader, exp_dis_net, exp_cat_net, face_proj, 
                lands_info, best_loss)
            if test_loss[-1] is True:
                best_loss = test_loss[0].avg
                save_loss = test_loss
    
    model_perf = {
        'train_losses': {
            'loss_total': train_losses[0].avg,
            'loss_exchange': train_losses[1].avg,
            'loss_back': train_losses[2].avg,
            'loss_exp': train_losses[3].avg,
            'loss_vec': train_losses[4].avg
        },
        'test_losses': {
            'loss_total': save_loss[0].avg,
            'loss_exchange': save_loss[1].avg,
            'loss_back': save_loss[2].avg,
            'loss_exp': save_loss[3].avg,
            'loss_vec': save_loss[4].avg
        }
    }
    hyperSaver.set_config(model_perf, match_template=False)
    hyperSaver._show_serialized_json()
    hyperSaver.save_all_configs_to_json(os.path.join(log_path, './results.json'))


if __name__ == '__main__':
    main()
