import torch
import json
from glob import glob
import numpy as np
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str,
                    default='', help='input track params')
parser.add_argument('--save_path', type=str,
                    default='', help='save path')
parser.add_argument('--aud_path', type=str,
                    default='', help='input track params')
parser.add_argument('--seq_num', type=int, default=3, help='seq num')
parser.add_argument('--param_scale', type=float, default=0.5)
args = parser.parse_args()


aud_feat = torch.load(args.aud_path)
with open(args.json_path, "r") as f:
    transform = json.load(f)

json_path_dirname = os.path.dirname(args.json_path)
last_dirname = json_path_dirname.split("/")[-1]
transform_len = len(transform['frames'])
if not args.save_path:
    save_path = json_path_dirname
else:
    save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

aud_len = aud_feat.shape[0]
data_len = transform_len

transform_list = []
for i in range(data_len):
    transform_list.append(
        np.array(transform['frames'][i]['transform_matrix'], dtype=np.float32)
                    )

transform_arr = np.array(transform_list, dtype=np.float32)
print("transform_arr.shape: ", transform_arr.shape)
transform_arr_diff = transform_arr[1:] - transform_arr[:-1]
transform_arr_diff = transform_arr_diff * args.param_scale
transform_curr = transform_arr[0]
for i in range(data_len - 1):
    transform_arr[i + 1] = transform_curr + transform_arr_diff[i]
    transform_curr = transform_arr[i + 1]
sample_len = data_len - aud_len

for seq_i in range(args.seq_num):
    transform_copy = transform.copy()
    transform_copy['frames'] = []

    start_idx = random.randint(0, sample_len)
    print("start_idx: ", start_idx)
    for idx, i in enumerate(range(start_idx, start_idx + aud_len)):
        # offset_i = i + offset
        tmp = transform['frames'][i].copy()
        # transform_matrix_np = np.array(tmp['transform_matrix'], dtype=np.float32)
        # transform_matrix_np[:2, 3] = transform_matrix_np[:2, 3]
        tmp['transform_matrix'] = transform_arr[i].tolist()
        tmp['img_id'] = idx
        tmp['aud_id'] = idx
        transform_copy['frames'].append(tmp)

    save_name = 'transform_val_hps{}.json'.format(seq_i)

    with open(os.path.join(save_path, save_name), "w") as f:
        json.dump(transform_copy, f)
