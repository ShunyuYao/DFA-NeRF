import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'core'))
from pathlib import Path
from utils_core import flow_viz
from utils import load_flow
import cv2
import struct
import argparse
from data_test_flow import *
from models.network_test_flow import NeuralNRT
from options_test_flow import TrainOptions, TestOptions, ValOptions
import torch
import numpy as np



def save_flow(filename, flow_input):
    flow = np.copy(flow_input)

    # Flow is stored row-wise in order [channels, height, width].
    assert len(flow.shape) == 3

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', flow.shape[2]))
        fout.write(struct.pack('I', flow.shape[1]))
        fout.write(struct.pack('I', flow.shape[0]))
        fout.write(struct.pack('={}f'.format(flow.size), *flow.flatten("C")))


def save_flow_numpy(filename, flow_input):
    np.save(filename, flow_input)


def triangle(img):
    H, W, _ = img.shape
    interval_x = W // 50
    interval_y = H // 50
    pos_v = []
    for i in range(0, H-1, interval_y):
        for j in range(0, W-1, interval_x):
            pos_v.append([i, j])
        pos_v.append([i, W-1])
    for j in range(0, W-1, interval_x):
        pos_v.append([H-1, j])
    pos_v.append([H-1, W-1])
    new_H = len(list(range(0, H-1, interval_y))) + 1
    new_W = len(list(range(0, W-1, interval_x))) + 1
    return pos_v, new_H, new_W


def viz(img, img2, save_path_fw, flo, mask):
    img = img.permute(1, 2, 0).cpu().numpy()
    H, W, C = img.shape
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    flo = flo.transpose(1, 2, 0)
    mask = (mask != 0)[0, :, :, None].cpu().numpy()
    print(mask.shape)
    align_img1 = np.zeros(img.shape, np.uint8)
    # if pos_v is None:
    #     pos_v, new_H, new_W = triangle(img)
    img_copy = img.copy()
    img2_copy = img2.copy()

    new_pos = []
    for i in range(flo.shape[0]):
        for j in range(flo.shape[1]):
            x = int(j + np.floor(flo[i, j, 0]))
            y = int(i + np.floor(flo[i, j, 1]))
            if not mask[i, j, 0]:
                continue

            if x >= W or x < 0 or y >= H or y < 0:
                continue
            align_img1[y, x, :] = img[i, j, :]

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # cv2.imwrite("flow1.png", flo)
    img_flo = np.concatenate([img, flo], axis=0)

    warp_img_tar = np.concatenate([align_img1, img2], axis=0)
    img_cat = np.concatenate([img_flo, warp_img_tar], axis=1)
    cv2.imwrite(save_path_fw, img_cat[:, :, [2, 1, 0]])


def predict(data):
    with torch.no_grad():
        model.eval()
        path_flow = data["path_flow"]
        src_crop_im = data["src_crop_color"].cuda()
        tar_crop_im = data["tar_crop_color"].cuda()
        src_im = data["src_color"].cuda()
        tar_im = data["tar_color"].cuda()
        src_mask = data["src_mask"].cuda()
        tar_mask = data["tar_mask"].cuda()
        crop_param = data["Crop_param"].cuda()
        B = src_mask.shape[0]
        flow = model(src_crop_im, tar_crop_im, src_im, tar_im, crop_param)
        for i in range(B):
            flow_tmp = flow[i].cpu().numpy() * src_mask[i].cpu().numpy()
            save_flow_numpy(os.path.join(save_path, os.path.basename(
                path_flow[i])[:-6]+".npy"), flow_tmp)


if __name__ == "__main__":
    width = 272
    height = 480

    test_opts = TestOptions().parse()
    test_opts.pretrain_model_path = os.path.join(
        dir_path, 'pretrain_model/raft-small.pth')
    data_loader = CreateDataLoader(test_opts)
    testloader = data_loader.load_data()
    model_path = os.path.join(dir_path, 'sgd_NNRT_model_epoch19008_50000.pth')
    model = NeuralNRT(test_opts, os.path.join(
        dir_path, 'pretrain_model/raft-small.pth'))
    state_dict = torch.load(model_path)

    model.CorresPred.load_state_dict(state_dict["net_C"])
    model.ImportanceW.load_state_dict(state_dict["net_W"])

    model = model.cuda()

    save_path = test_opts.savepath
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for batch_idx, data in enumerate(testloader):
        predict(data)
        if(batch_idx % 100 == 0):
            print('estimated flow', batch_idx)
