# --------------------------------------------------------
# Licensed under The MIT License
# Written by Shunyu Yao (ysy at sjtu.edu.cn)
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.onnx
import _init_paths
from config import cfg
from config import update_config
import models

from thop import profile
from thop import clever_format


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--LoadModelPath',
                        help='load model path',
                        type=str,
                        required=True)

    parser.add_argument('--ExportModelPath',
                        help='export model path',
                        type=str,
                        required=True)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--export_onnx_model',
                        default='eye',
                        choices=['face', 'eye'])
    parser.add_argument('--export_last_epoch',
                        action='store_true')
    parser.add_argument('--parameter_count',
                        action='store_true')

    args = parser.parse_args()

    return args


args = parse_args()
args.lr_range_test = True
update_config(cfg, args)

if args.export_onnx_model == 'eye':
    model = eval('models.'+cfg.MODEL.NAME+'.get_eye_net')(
        cfg, is_train=False
    )
elif args.export_onnx_model == 'face':
    model = eval('models.'+cfg.MODEL.NAME+'.get_face_net')(
        cfg, is_train=False
    )
else:
    raise Exception('Model do not support now.')

model_path = args.LoadModelPath
state_dict = torch.load(model_path)
if not args.export_last_epoch:
    print('Export the best performance model weights.')
    print('Best performance: ', state_dict['best_perf'])
    model.load_state_dict(state_dict['best_state_dict'])
else:
    print('Export the last epoch model weights.')
    state_dict = state_dict['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


WIDTH = cfg.MODEL.IMAGE_SIZE[0]
HEIGHT = cfg.MODEL.IMAGE_SIZE[1]
if args.export_onnx_model == 'eye':
    batch_size = 2
    if cfg.DATASET.COLOR_RGB:
        input_c = 3
    else:
        input_c = 1
elif args.export_onnx_model == 'face':
    batch_size = 1
    input_c = 3
input_h, input_w = HEIGHT, WIDTH

export_params = True
opset_version = 9
do_constant_folding = True
model.cpu()
model.eval()

# Input to the model
x = torch.randn(batch_size, input_c, input_h, input_w) #, requires_grad=True)
torch_out = model(x)

# Parameter count
# if args.parameter_count:
#     from torchscan import summary
#
#     model_scan = model.cuda()
#     print(summary(model, (input_c, input_h, input_w), max_depth=2))
#     exit()

# macs, params = profile(model, inputs=(x, ))
# macs, params = clever_format([macs, params], "%.3f")
# print("macs: ", macs)
# print("params: ", params)

# Export the model
torch.onnx.export(model,                          # model being run
                  x,                                  # model input (or a tuple for multiple inputs)
                  args.ExportModelPath,                 # where to save the model (can be a file or file-like object)
                  export_params=export_params,        # store the trained parameter weights inside the model file
                  opset_version=opset_version,          # the ONNX version to export the model to
                  do_constant_folding=do_constant_folding,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['max_val', 'max_pos'], # the model's output names 'heatmap',
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                  #             'output' : {0 : 'batch_size'}}
                 )

print("Onnx model successfullt export to path {}".format(args.ExportModelPath))
