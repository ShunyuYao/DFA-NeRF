import json
import torch
import numpy as np
import os
import argparse
import cv2
import shutil
from glob import glob
# from scipy.ndimage import gaussian_filter1d
from PIL import Image

BG_LABEL = 255
HEAD_LABEL = 29
NECK_LABEL = 150
BODY_LABEL = 76

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', type=str,
                    default='', help='input_img_path')
parser.add_argument('--input_val_img_path', type=str,
                    default='', help='input_val_img_path')
parser.add_argument('--gt_img_path', type=str,
                    default='', help='gt_img_path')
parser.add_argument('--parse_path', type=str,
                    default='', help='parse_path')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--val_idx', type=int, default=7111)
parser.add_argument('--input_size', type=int, default=512)
args = parser.parse_args()


img_input_dir = args.input_img_path
img_val_input_dir = args.input_val_img_path
gt_img_dir = args.gt_img_path
save_dir = args.save_path
if args.mode == 'train_val':
    input_train_dst = os.path.join(save_dir, 'trainsets', 'input')
    gt_train_dst = os.path.join(save_dir, 'trainsets', 'gt')
    parse_train_dst = os.path.join(save_dir, "seg_map_train")
    os.makedirs(input_train_dst, exist_ok=True)
    os.makedirs(gt_train_dst, exist_ok=True)
    os.makedirs(parse_train_dst, exist_ok=True)
    
    input_val_dst = os.path.join(save_dir, 'valsets', 'input')
    gt_val_dst = os.path.join(save_dir, 'valsets', 'gt')
    parse_val_dst = os.path.join(save_dir, "seg_map_val")
    os.makedirs(input_val_dst, exist_ok=True)
    os.makedirs(gt_val_dst, exist_ok=True)
    os.makedirs(parse_val_dst, exist_ok=True)
else:
    os.makedirs(save_dir, exist_ok=True)

input_paths = sorted(os.listdir(img_input_dir))
gt_img_paths = sorted(glob(os.path.join(gt_img_dir, "*.jpg")))
val_idx = args.val_idx

cnt = 0
print("len(gt_img_paths): ", len(gt_img_paths))
if args.mode == 'train_val':
    for i, path in enumerate(gt_img_paths):
        print("idx: ", i)
        if i < val_idx:
            continue
        input_dir_src = os.path.join(img_input_dir, "{:06d}.jpg".format(i))
        parse_dir_src = os.path.join(args.parse_path, "{:06d}.png".format(i))
        
        gt_dir_src = os.path.join(gt_img_dir, "{:06d}.jpg".format(i))

        img_parse = Image.open(parse_dir_src).convert('L')
        img_parse_arr = np.array(img_parse)

        img_tmp = np.zeros((args.input_size, args.input_size, 1)).astype(np.uint8)
        idx = np.argwhere(img_parse_arr == BODY_LABEL)
        for idx_i in range(idx.shape[0]):
            idxes = idx[idx_i]
            img_tmp[idxes[0], idxes[1]] = np.array([255, ])
        
        if i < val_idx:
            input_dst = input_train_dst
            gt_dst = gt_train_dst
            parse_dst = os.path.join(parse_train_dst, "{:06d}.png".format(i))
        else:
            input_dst = input_val_dst
            gt_dst = os.path.join(gt_val_dst, "{:06d}.jpg".format(cnt))
            parse_dst =  os.path.join(parse_val_dst, "{:06d}.png".format(cnt))
            input_dir_src = os.path.join(img_val_input_dir, "{:06d}.jpg".format(cnt))
            cnt += 1
        
        shutil.copy(input_dir_src, input_dst)
        shutil.copy(gt_dir_src, gt_dst)    
        cv2.imwrite(parse_dst, img_tmp)
else:
    for i, path in enumerate(input_paths):
        input_src = os.path.join(img_input_dir, "{:06d}.jpg".format(i))
        input_dst = save_dir
        
        shutil.copy(input_src, input_dst)
