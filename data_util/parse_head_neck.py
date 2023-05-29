import cv2
import numpy as np
import face_alignment
from skimage import io
import torch
import torch.nn.functional as F
import json
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
                    default='obama', help='identity of target person')

args = parser.parse_args()
id = args.id
id_dir = os.path.join('dataset', id)
Path(id_dir).mkdir(parents=True, exist_ok=True)
ori_imgs_dir = os.path.join('dataset', id, 'ori_imgs')
head_imgs_dir = os.path.join('dataset', id, 'head_torso_imgs') # torso_imgs head_neck_imgs
parsing_dir = os.path.join(id_dir, 'parsing')
Path(head_imgs_dir).mkdir(parents=True, exist_ok=True)

max_frame_num = 200000
valid_img_ids = []
for i in range(max_frame_num):
    if os.path.isfile(os.path.join(ori_imgs_dir, str(i) + '.lms')):
        valid_img_ids.append(i)
valid_img_num = len(valid_img_ids)
tmp_img = cv2.resize(cv2.imread(os.path.join(ori_imgs_dir, str(valid_img_ids[0])+'.jpg')),(512,512))
h, w = tmp_img.shape[0], tmp_img.shape[1]

bc_img = cv2.imread(os.path.join(id_dir, 'bc.jpg'))

# head_neck
# for i in valid_img_ids:
#     print(i)
#     parsing_img = cv2.imread(os.path.join(parsing_dir, str(i) + '.png'))
#     head_part = (parsing_img[:, :, 0] == 255) & (
#         parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 0) | (parsing_img[:, :, 0] == 0) & (
#         parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 0)
#     bc_part = (parsing_img[:, :, 0] == 255) & (
#         parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
#     img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.jpg'))
#     img[bc_part] = bc_img[bc_part]
#     # cv2.imwrite(os.path.join(com_imgs_dir, str(i) + '.jpg'), img)
#     img[~head_part] = bc_img[~head_part]
#     cv2.imwrite(os.path.join(head_imgs_dir, str(i) + '.jpg'), img)

# torso
# for i in valid_img_ids:
#     print(i)
#     parsing_img = cv2.imread(os.path.join(parsing_dir, str(i) + '.png'))
#     head_part = (parsing_img[:, :, 0] == 0) & (
#         parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 255)
#     bc_part = (parsing_img[:, :, 0] == 255) & (
#         parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
#     img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.jpg'))
#     img[bc_part] = bc_img[bc_part]
#     # cv2.imwrite(os.path.join(com_imgs_dir, str(i) + '.jpg'), img)
#     img[~head_part] = bc_img[~head_part]
#     cv2.imwrite(os.path.join(head_imgs_dir, str(i) + '.jpg'), img)

# torso neck
for i in valid_img_ids:
    print(i)
    parsing_img = cv2.imread(os.path.join(parsing_dir, str(i) + '.png'))
    head_part = (parsing_img[:, :, 0] == 0) & (
        parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 255) | (parsing_img[:, :, 0] == 0) & (
        parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 0)
    bc_part = (parsing_img[:, :, 0] == 255) & (
        parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
    img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.jpg'))
    img[bc_part] = bc_img[bc_part]
    # cv2.imwrite(os.path.join(com_imgs_dir, str(i) + '.jpg'), img)
    img[~head_part] = bc_img[~head_part]
    cv2.imwrite(os.path.join(head_imgs_dir, str(i) + '.jpg'), img)