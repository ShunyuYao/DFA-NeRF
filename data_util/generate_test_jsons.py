import torch
import json
from glob import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str,
                    default='', help='input track params')
parser.add_argument('--input_auds_path', type=str,
                    default='', help='input track params')
parser.add_argument('--save_path', type=str,
                    default='', help='save path')
parser.add_argument('--param_scale', type=float, default=0.5)
parser.add_argument('--static_pose_idx', type=int, default=0)
parser.add_argument('--no_output_static', action='store_true',
                    help='do not output static params')
args = parser.parse_args()


with open(args.json_path, "r") as f:
    transform = json.load(f)

json_path_dirname = os.path.dirname(args.json_path)
last_dirname = json_path_dirname.split("/")[-1]
transform_len = len(transform['frames'])
aud_paths = glob(os.path.join(args.input_auds_path, "*.pt"))
aud_feats = []
if not args.save_path:
    save_path = json_path_dirname
else:
    save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
for aud_path in aud_paths:
    aud_basename = os.path.basename(aud_path).split('.')[0]
    if last_dirname in aud_basename:
        continue
    aud_feat = torch.load(aud_path)
    
    transform_copy = transform.copy()
    transform_static_copy = transform.copy()
    aud_len = aud_feat.shape[0]
    transform_copy['frames'] = []
    transform_static_copy['frames'] = []
    data_len = aud_len
    if aud_len > transform_len:
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
    for i in range(data_len):
        # offset_i = i + offset
        tmp = transform['frames'][i].copy()
        # transform_matrix_np = np.array(tmp['transform_matrix'], dtype=np.float32)
        # transform_matrix_np[:2, 3] = transform_matrix_np[:2, 3]
        tmp['transform_matrix'] = transform_arr[i].tolist()
        tmp['img_id'] = i
        tmp['aud_id'] = i
        transform_copy['frames'].append(tmp)
        if not args.no_output_static:
            tmp_static = transform['frames'][args.static_pose_idx].copy()
            tmp_static['img_id'] = i
            tmp_static['aud_id'] = i
            transform_static_copy['frames'].append(tmp_static)

    save_name = 'transform_val_{}.json'.format(aud_basename)
    save_static_name = 'transform_val_static_{}.json'.format(aud_basename)
    with open(os.path.join(save_path, save_name), "w") as f:
        json.dump(transform_copy, f)
    with open(os.path.join(save_path, save_static_name), "w") as f:
        json.dump(transform_static_copy, f)