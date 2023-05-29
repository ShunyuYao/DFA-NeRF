import json
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str,
                    default='', help='input track params')
parser.add_argument('--save_path', type=str,
                    default='', help='save path')
args = parser.parse_args()

param = torch.load(args.input_path)
exp_list = param['exp'].tolist()
save_path = args.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)

param_list = {}
for key, value in param.items():
    param_list[key] = value.tolist()
    
for i in range(len(param_list['exp'])):
    save_dict = { 
            'exp': param_list['exp'][i],
            'euler': param_list['euler'][i],
            'trans': param_list['trans'][i]
        }
    with open(os.path.join(save_path, "{:08d}.json".format(i)), 'w') as f:
        json.dump(save_dict, f)