import json
#import trt_pose.coco
import numpy as np
import argparse
import os
import time
import pprint
import cv2
import sys
import matplotlib.pyplot as plt
import pickle as pkl

import math
from math import sqrt
from glob import glob

import torch
import torch.backends.cudnn as cudnn

import _init_paths_demo
from config import cfg
from config_new import cfg_eye
# from config import cfg as cfg_eye
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from core.inference import demo_preds_function
from core.inference import gaussian_modulation_torch
from utils.transforms import flip_back, MATCHED_PARTS
from utils.transforms import get_affine_transform, LTRB_to_xywh, affine_transform
from utils.transforms import crop, fliplr_eye_joints
from utils.utils import parse_roi_box_from_landmark, draw_circle_map, find_tensor_peak_batch
from collections import deque

dir_path = os.path.dirname(os.path.realpath(__file__))
# from utils.transforms_face import crop
import models
#from save_pose import SaveObjects, save_kp2d_to_json

sys.path.append("./third_party/BlazeFace-PyTorch")
from blazeface import BlazeFace
BlazeFacePath = "./third_party/BlazeFace-PyTorch"

sys.path.append("./third_party/head-pose-estimation")
from utils.head_pose_estimation import PoseEstimator
from stabilizer import Stabilizer
# sys.path.append(r"E:\projects\smooth_network")
# from lib.model.model_smooth import TemporalModel

sys.path.append("./third_party/useful_codes")
from filters.common_filters import OneEuroFilter
from pose_utils.draw_keypoints import draw_circle, face_kpts_98_to_68
from pose_utils.transform import pts2cs, bbox2cs

class KpFilter():
    def __init__(self, temp_filter, num_kps, filter_settings):
        self.filters = []
        for i in range(num_kps):
            self.filters.append(temp_filter(**filter_settings))
            # setattr(self, 'filter_{}'.format(i), temp_filter)
    def __getitem__(self, index):
        return self.filters[index]

miu = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

filter_settings = {
    'freq': 30,
    'mincutoff': 0.7,
    'beta': 0.007,
    'dcutoff': 1
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--cfg_eye',
                        help='eye experiment configure file name',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--testModelPath',
                        help='test model path',
                        type=str,
                        default='./models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--testEyeModelPath',
                        help='test model path',
                        type=str,
                        default='./models/eye_ghostnet_40x35_reg.pth')
    parser.add_argument('--testMode',
                        help='test mode',
                        type=str,
                        choices=['video', 'camera', 'filepath'],
                        default='camera')
    parser.add_argument('--inputPath',
                        help='input path, video or file directory',
                        type=str,
                        default='./demo/kunkun_cut.mp4')
    parser.add_argument('--outputVidPath',
                        help='output path',
                        type=str,
                        default='')
    parser.add_argument('--outputSavePath',
                        help='output save path',
                        type=str,
                        default='')
    parser.add_argument('--outputCropPath',
                        help='output crop path',
                        type=str,
                        default='')           
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
    parser.add_argument('--filter_type',
                        help='input video path',
                        default=None,
                        choices=['OneEuro', None])
    parser.add_argument('--only_upper_body',
                        help='only upper body',
                        action='store_true')
    parser.add_argument('--thresh',
                        help='thresh for kps show',
                        default=0.3,
                        type=float)
    parser.add_argument('--cpu',
                        help='use cpu to inference',
                        action='store_true')
    parser.add_argument('--num_joints',
                        help='number of joints to pred',
                        default=98,
                        type=int)
    parser.add_argument('--face_detect_inputSize',
                        help='the input size of face detection',
                        default=128,
                        type=int)
    parser.add_argument('--dst_size',
                        help='the input size of frame',
                        default=0,
                        type=int)
    parser.add_argument('--face_bbox_scale',
                        help='the scale factor of face bbox',
                        default=1.25*0.3,
                        type=float)
    parser.add_argument('--not_use_imagenet_normalize',
                        help='not use imagenet normalize',
                        action='store_true')
    parser.add_argument('--flip_eye',
                        action='store_true')
    parser.add_argument('--face_type',
                        help='input video path',
                        default='WLFW',
                        choices=['WLFW', '300W'])
    parser.add_argument('--export_onnx',
                        default=None,
                        choices=['face', 'eye'])
    parser.add_argument('--eye_heatmap_decode',
                        action='store_true')
    parser.add_argument('--multi_eye_branch',
                        action='store_true')
    parser.add_argument('--smoother',
                        action='store_true')
    parser.add_argument('--only_smooth_eye',
                        action='store_true')
    parser.add_argument('--use_sbr_model',
                        action='store_true')
    parser.add_argument('--use_optical_flow',
                        action='store_true')
    parser.add_argument('--draw_pose_annot',
                        action='store_true')

    args = parser.parse_args()
    return args


def preprocess_face_detect(image, output_size, trans_color=True):
    if frameSize[0] > frameSize[1]:
        mid_pt = frameSize[0] // 2
        square_len = frameSize[1]
        short_side = mid_pt - square_len // 2
        long_side = mid_pt + square_len // 2
        img_crop = image[:, int(short_side):int(long_side)]
    elif frameSize[0] < frameSize[1]:
        mid_pt = frameSize[1] // 2
        square_len = frameSize[0]
        short_side = mid_pt - square_len // 2
        long_side = mid_pt + square_len // 2
        img_crop = image[int(short_side):int(long_side), :]
    else:
        img_crop = image
        # short_side = image.shape[0]

    image_resize = cv2.resize(img_crop,
                              (output_size[0], output_size[1]))
    scale_ratio = img_crop.shape[0] / output_size[0]
    # print("short_side: ", img_crop.shape[0], output_size[0])
    if cfg.DATASET.COLOR_RGB and trans_color:
        image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    return image_resize, img_crop, scale_ratio

def preprocess_direct_resize(image):
    image_resize = cv2.resize(image, (WIDTH, HEIGHT))
    return image_resize, image

def cal_points_dist(pts1, pts2):
    assert len(pts1.shape == 2) and len(pts2.shape == 2), "pts shape must equal to 2"
    pts_delta = (pts1 - pts2) * (pts1 - pts2)



def get_img_np_nchw(image):
    # image_cv = image # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_cv = cv2.resize(image_cv, (128, 96))

    # img_np = np.array(image_cv, dtype=float) / 255.0
    # r = (img_np[:, :, 0] - miu[0]) / std[0]
    # g = (img_np[:, :, 1] - miu[1]) / std[1]
    # b = (img_np[:, :, 2] - miu[2]) / std[2]
    # img_np_t = np.array([r, g, b])
    if not args.not_use_imagenet_normalize:
        image = (image / 255.0 - miu) / std
    image = image.transpose((2, 0, 1))
    image_np_nchw = np.expand_dims(image, axis=0)
    return image_np_nchw

def save_kp2d_to_json(kp_2d, dump_dir, scores=None):
    json_out = {"pose_keypoints_2d": kp_2d.tolist()}
    json_out["pose_scores_2d"] = scores.tolist()
    with open(dump_dir, 'w', encoding='utf-8') as json_file:
        json.dump(json_out, json_file)

def decode_face_detections(detections, img_shape):
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img_shape[0]
        xmin = detections[i, 1] * img_shape[1]
        ymax = detections[i, 2] * img_shape[0]
        xmax = detections[i, 3] * img_shape[1]

    return [xmin, ymin, xmax, ymax]


def transform_eye_imgs(img, pts, dataset='WLFW'):
    if dataset == 'WLFW':
        pts_left_eye = pts[68:76]
        pts_right_eye = pts[60:68]
    elif dataset == '300W':
        pts_left_eye = pts[42:48]
        pts_right_eye = pts[36:42]
    elif dataset == '300W_eye':
        print("pts[42:48] shape: ", pts[42:48].shape)
        print("pts[22:27] shape: ", pts[22:27].shape)
        pts_left_eye = np.concatenate([pts[42:48], pts[22:27]], axis=0)
        pts_right_eye = np.concatenate([pts[42:48], pts[17:22]], axis=0)
    else:
        raise Exception('no dataset found')
    # pts_left_pupil = pts[97].reshape(-1, 2)
    # pts_right_pupil = pts[96].reshape(-1, 2)

    # pts_left_eye = np.vstack([pts_left_eye, pts_left_pupil])
    # pts_right_eye = np.vstack([pts_right_eye, pts_right_pupil])

    l_eye_center, l_eye_scale = pts2cs(pts_left_eye)
    r_eye_center, r_eye_scale = pts2cs(pts_right_eye)

    l_eye_scale = l_eye_scale * 1.35
    r_eye_scale = r_eye_scale * 1.35

    # nparts = pts_left_eye.shape[0]

    # if cfg.DATASET.COLOR_RGB:
    #     print("COLOR_BGR2RGB")
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_l_eye = crop(img, l_eye_center, l_eye_scale, [EYE_WIDTH, EYE_HEIGHT], rot=0)
    img_r_eye = crop(img, r_eye_center, r_eye_scale, [EYE_WIDTH, EYE_HEIGHT], rot=0)

    # flip left img right
    if args.flip_eye:
        img_l_eye = np.fliplr(img_l_eye)

    trans_l = get_affine_transform(l_eye_center, l_eye_scale, 0, [EYE_WIDTH, EYE_HEIGHT], inv=1)
    trans_r = get_affine_transform(r_eye_center, r_eye_scale, 0, [EYE_WIDTH, EYE_HEIGHT], inv=1)

    # img_l_eye = cv2.cvtColor(img_l_eye, cv2.COLOR_BGR2RGB)
    # img_r_eye = cv2.cvtColor(img_r_eye, cv2.COLOR_BGR2RGB)
    img_l_eye_gray_show = cv2.cvtColor(img_l_eye, cv2.COLOR_RGB2BGR)
    img_r_eye_gray_show = cv2.cvtColor(img_r_eye, cv2.COLOR_RGB2BGR)

    img_l_eye = cv2.cvtColor(img_l_eye_gray_show, cv2.COLOR_BGR2GRAY)
    img_r_eye = cv2.cvtColor(img_r_eye_gray_show, cv2.COLOR_BGR2GRAY)
    # img_l_eye_gray_show = img_l_eye.copy()
    # img_r_eye_gray_show = img_r_eye.copy()
    img_l_eye_gray = img_l_eye.astype(np.float32) / 255.0
    img_r_eye_gray = img_r_eye.astype(np.float32) / 255.0
    # img_l_eye_gray = img_l_eye_gray.transpose(2, 0, 1)
    # img_r_eye_gray = img_r_eye_gray.transpose(2, 0, 1)
    # img_l_eye_gray = img_l_eye_gray[np.newaxis, ...]
    # img_r_eye_gray = img_r_eye_gray[np.newaxis, ...]
    img_l_eye_gray = img_l_eye_gray[np.newaxis, np.newaxis, ...]
    img_r_eye_gray = img_r_eye_gray[np.newaxis, np.newaxis, ...]

    img_eyes_gray = np.vstack([img_l_eye_gray, img_r_eye_gray])
    return img_eyes_gray, [img_l_eye_gray_show, img_r_eye_gray_show], [trans_l, trans_r]

def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def trans_back(vertex, roi_bbox, size=128):
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    vertex[:, 0] = vertex[:, 0] * scale_x + sx
    vertex[:, 1] = vertex[:, 1] * scale_y + sy

    return vertex

args = parse_args()
if args.export_onnx == 'face':
    args.lr_range_test = True
args_eye = parse_args()
args_eye.cfg = args_eye.cfg_eye
update_config(cfg, args)
update_config(cfg_eye, args_eye)

num_joints = cfg.MODEL.NUM_FACE_JOINTS
num_eye_points = cfg_eye.MODEL.NUM_EYE_JOINTS
RECEPTIVE_FIELD = 27
if args.smoother:
    dq = deque()

if not args.cpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

logger, final_output_dir, tb_log_dir = create_logger(
    cfg, args.cfg, 'valid')

if args.use_optical_flow:
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(8, 8),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     20, 0.03))
# input size
WIDTH = cfg.MODEL.IMAGE_SIZE[0] # 96
HEIGHT = cfg.MODEL.IMAGE_SIZE[1] # 128
FACE_DETECTION_WIDTH = args.face_detect_inputSize
FACE_DETECTIOM_HEIGHT = args.face_detect_inputSize
USE_HEATMAP = cfg.MODEL.EXTRA.USE_HEATMAP_BRANCH
USE_BOUNDARY_MAP = cfg.MODEL.EXTRA.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in cfg.MODEL.EXTRA else False
low_score_idxes = []
# input eye size
EYE_WIDTH = cfg_eye.MODEL.IMAGE_SIZE[0]
EYE_HEIGHT = cfg_eye.MODEL.IMAGE_SIZE[1]
time_se_channel = cfg.MODEL.EXTRA.IMG_CHANNEL - 3 if "IMG_CHANNEL" in cfg.MODEL.EXTRA else 0
USE_TIME_SE = False
if time_se_channel > 0:
    USE_TIME_SE = True
    heatmap_se = np.zeros((time_se_channel, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
    input = torch.zeros(1, cfg.MODEL.EXTRA.IMG_CHANNEL, HEIGHT, WIDTH, dtype=torch.float32).to(device)

if args.filter_type == 'OneEuro':
    kp_filters = KpFilter(OneEuroFilter, 6, filter_settings)


model_path = args.testModelPath
eye_model_path = args.testEyeModelPath
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = eval('models.'+cfg.MODEL.NAME+'.get_face_net')(
    cfg, is_train=False
)

eye_model = eval('models.'+cfg_eye.MODEL.NAME+'.get_eye_net')(
    cfg_eye, is_train=False
)

face_model = BlazeFace()
if not args.cpu:
    model.to(device)
    eye_model.to(device)
    face_model.to(device)

logger.info('=> loading model from {}'.format(model_path))
logger.info('=> loading eye model from {}'.format(eye_model_path))

print("model_path: ", model_path)
state_dict = torch.load(model_path)  # map_location={'cuda:2':'cuda:0'}
print(state_dict['best_perf'])
model.load_state_dict(state_dict['best_state_dict'])

print("eye_model_path: ", eye_model_path)
pretrained_eye_model = torch.load(eye_model_path)
if 'best_state_dict' in pretrained_eye_model.keys():
    print('eye state_dict')
    pretrained_eye_model = pretrained_eye_model['best_state_dict']

eye_model.load_state_dict(pretrained_eye_model, strict=True)
# eye_model.load_state_dict(new_state_dict, strict=True)

face_model.load_weights(os.path.join(BlazeFacePath, "blazeface.pth"))
face_model.load_anchors(os.path.join(BlazeFacePath, "anchors.npy"))

face_model.min_score_thresh = 0.3
face_model.min_suppression_threshold = 0.3

if args.smoother:
    model_smooth = TemporalModel(num_joints, 2, num_joints, filter_widths=[3, 3, 3], causal=True, channels=256).to(device)
    checkpoint = torch.load(r"E:\projects\smooth_network\checkpoint\epoch_100.bin")
    state_dict = checkpoint['model_pos']
    model_smooth.load_state_dict(state_dict)

logger.info('model already loaded!')

'''
export eye onnx model
'''
if args.export_onnx == 'eye':
    import torch.onnx

    batch_size = 2
    input_c, input_h, input_w = 1, EYE_HEIGHT, EYE_WIDTH

    export_params = True
    opset_version = 10
    do_constant_folding = True
    eye_model.cpu()
    eye_model.eval()

    # Input to the model
    x = torch.randn(batch_size, input_c, input_h, input_w) #, requires_grad=True)
    torch_out = eye_model(x)

    # Export the model
    torch.onnx.export(eye_model,                          # model being run
                      x,                                  # model input (or a tuple for multiple inputs)
                      './models/eye_model_v0.5.1.onnx',                 # where to save the model (can be a file or file-like object)
                      export_params=export_params,        # store the trained parameter weights inside the model file
                      opset_version=opset_version,          # the ONNX version to export the model to
                      do_constant_folding=do_constant_folding,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                      #             'output' : {0 : 'batch_size'}}
                     )
    exit()

'''
export face onnx model
'''
if args.export_onnx == 'face':
    import torch.onnx

    batch_size = 1
    input_c, input_h, input_w = 3, HEIGHT, WIDTH

    export_params = True
    opset_version = 10
    do_constant_folding = True
    model.cpu()
    model.eval()

    # Input to the model
    x = torch.randn(batch_size, input_c, input_h, input_w) #, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,                                          # model being run
                      x,                                              # model input (or a tuple for multiple inputs)
                      './models/facelandmark_v2.2.onnx',                       # where to save the model (can be a file or file-like object)
                      export_params=export_params,                    # store the trained parameter weights inside the model file
                      opset_version=opset_version,                    # the ONNX version to export the model to
                      do_constant_folding=do_constant_folding,        # whether to execute constant folding for optimization
                      input_names = ['input'],                        # the model's input names
                      output_names = ['output'],                      # the model's output names
                      #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                      #             'output' : {0 : 'batch_size'}}
                     )
    exit()

'''
capture and crop photos
'''
print("args.testMode: ", args.testMode)
if args.testMode == 'camera':
    cap = cv2.VideoCapture(1)
    # hasFrame, frame = cap.read()
    # print("hasFrame: ", hasFrame)
    # while hasFrame:
    #     hasFrame, frame = cap.read()
    #     cv2.imshow("frame", frame)
    #     cv2.waitKey(1)
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
elif args.testMode == 'video':
    cap = cv2.VideoCapture(args.inputPath)
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
elif args.testMode == 'filepath':
    # pic_exts = ["*.jpg", "*.png", "*.jpeg"]
    # pic_paths = []
    # for ext in pic_exts:
    #     pic_paths.extend(glob(os.path.join(args.inputPath, ext)))
    # pic_paths = sorted(pic_paths)
    # pics_num = len(glob(os.path.join(args.inputPath, "*.jpg")))
    # os.listdir(args.inputPath) # len() // 2
    pic_paths = sorted(glob(os.path.join(args.inputPath, "*.jpg")))
    # [os.path.join(args.inputPath, "{}.jpg".format(i)) for i in range(pics_num)]
    print("pic_paths: ", pic_paths)
    tmp = cv2.imread(pic_paths[0])
    frameSize = (int(tmp.shape[0]),
            int(tmp.shape[1]))
else:
    raise NotImplementedError('only video or camera input supported')




'''
model test demo
'''
# flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
flip_pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
flip_pairs = np.array(flip_pairs) - 5
sigma = cfg.MODEL.SIGMA

if args.outputVidPath:
    # JSON_OUTPUT_DIR = os.path.join(args.outputVidPath, 'json')
    # OUTPUT_VIDEO_DIR = os.path.join(args.outputVidPath, 'video') # './demo_out/video/output_demo.mp4'
    OUTPUT_VIDEO_PATH = os.path.join(args.outputVidPath, 'output_demo.mp4')

    # if not os.path.exists(JSON_OUTPUT_DIR):
    #     os.makedirs(JSON_OUTPUT_DIR)
    if not os.path.exists(args.outputVidPath):
        os.makedirs(args.outputVidPath)
    # Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    crop_size = frameSize[0] if frameSize[0] < frameSize[1] else frameSize[1]
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(20), (crop_size, crop_size))
    
# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.3,
    cov_measure=18.1) for _ in range(6)]

with torch.no_grad():
    model.eval()
    face_model.eval()
    eye_model.eval()
    if args.smoother:
        model_smooth.eval()

    infer_times = []
    last_frame_pts = []
    counter = 0
    if args.testMode != 'filepath':
        hasFrame, frame = cap.read()
    else:
        if len(pic_paths) > 0:
            frame_path = pic_paths.pop(0)
            frame = cv2.imread(frame_path)
            hasFrame = True
        else:
            frame = None
            hasFrame = False
    print("hasFrame: ", hasFrame)
    if args.dst_size and hasFrame:
        frame, _, _ = preprocess_face_detect(frame, (args.dst_size, args.dst_size), trans_color=False)
    img_crop, img_crop_origin, scale_ratio = preprocess_face_detect(frame, (FACE_DETECTION_WIDTH, FACE_DETECTIOM_HEIGHT))
    frame_height, frame_width = img_crop_origin.shape[:2]
    pose_estimator = PoseEstimator(img_size=(frame_height, frame_width))

    if args.use_optical_flow:
        old_gray = cv2.cvtColor(img_crop_origin, cv2.COLOR_BGR2GRAY)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(img_crop_origin)
        frame_counter = 0
    # while True:
    while hasFrame:
        # hasFrame, frame = cap.read()
        # if not hasFrame:
        #     cv2.destroyAllWindows()
        #     break
        img_crop, img_crop_origin, scale_ratio = preprocess_face_detect(frame, (FACE_DETECTION_WIDTH, FACE_DETECTIOM_HEIGHT))
        
        frame_height, frame_width = img_crop_origin.shape[:2]
        # pose_estimator = PoseEstimator(img_size=(128, 128))

        # face detection
        face_detections = face_model.predict_on_image(img_crop)
        if face_detections.size(0) == 0:

            if args.outputVidPath:
                # print("img_crop_origin shape: ", img_crop_origin.shape)
                out_vid.write(img_crop_origin)

            if args.outputSavePath:
                # save_dict = {
                #     'pose_mat': pose_mat,
                #     'pose_np': pose_np,
                #     'steady_pose': steady_pose,
                #     'face_landmarks_show': face_landmarks_show
                # }
                
                out_lms_path = os.path.join(args.outputSavePath, 'lms')
                # out_pkl_path = os.path.join(args.outputSavePath, 'pkl')
                if not os.path.exists(out_lms_path):
                    os.makedirs(out_lms_path)
                # if not os.path.exists(out_pkl_path):
                #     os.makedirs(out_pkl_path)
                lands = np.zeros((68, 2), dtype=np.float32)
                # lands = lmk.copy() # preds[0].reshape(-1, 2)
                np.savetxt(os.path.join(out_lms_path, str(counter) + '.lms'), lands, '%f')
                counter += 1
                # with open(os.path.join(out_pkl_path, str(counter) + '.pkl'), 'wb') as f:
                #     pkl.dump(save_dict, f)

                # raise Exception('frame idx: {} no face.'.format(counter))
                if args.testMode != 'filepath':
                    hasFrame, frame = cap.read()
                else:
                    if len(pic_paths) > 0:
                        frame_path = pic_paths.pop(0)
                        frame = cv2.imread(frame_path)
                        hasFrame = True
                    else:
                        frame = None
                        hasFrame = False
                if args.dst_size and hasFrame:
                    frame, _, _ = preprocess_face_detect(frame, (args.dst_size, args.dst_size), trans_color=False)
                continue
        bbox_LTRB = decode_face_detections(face_detections, (FACE_DETECTIOM_HEIGHT, FACE_DETECTION_WIDTH))
        # img_show = cv2.rectangle(img_crop, (int(bbox_LTRB[0]), int(bbox_LTRB[1]) ), (int(bbox_LTRB[2]), int(bbox_LTRB[3]) ), (255, 0, 0), 2)
        
        bbox_LTRB = [x * scale_ratio for x in bbox_LTRB]
        center_tmp, scale_tmp = bbox2cs(bbox_LTRB)
        # if counter == 0:
        center = center_tmp
        scale = scale_tmp
        trans_face_inv = get_affine_transform(center, scale, rot=0,
                                              output_size=(WIDTH, HEIGHT), inv=1)

        img_crop = crop(img_crop_origin, center, scale, (WIDTH, HEIGHT), rot=0)
        # cv2.imwrite("./test.jpg", img_crop)

        img_np_nchw = get_img_np_nchw(img_crop).astype(dtype=np.float32)
        input_for_torch = torch.from_numpy(img_np_nchw).to(device)  # .cuda()
        if USE_TIME_SE:
            input[0, :time_se_channel] = input_for_torch
            input_for_torch = input
        if cfg.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(img_np_nchw, 3).copy()
            input_flipped = torch.from_numpy(input_flipped)
            if not args.cpu:
                input_flipped = input_flipped.to(device)
            inputs = torch.cat([input_for_torch, input_flipped])
            outputs = model(inputs)
            output = outputs[0][None, ...]
            output_flipped = outputs[1][None, ...]
            # output_flipped_pose = output_flipped[:12]
            # output_flipped_face = output_flipped[12:]

            output_flipped_pose = flip_back(output_flipped[:12].cpu().numpy(), flip_pairs)
            output_flipped_face = flip_back(output_flipped[12:].cpu().numpy(), MATCHED_PARTS['WFLW'])
            output_flipped = np.vstack([output_flipped_pose, output_flipped_face])

            # output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy())  # .cuda()
            if not args.cpu:
                output_flipped = output_flipped.to(device)
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5
        else:
            output = model(input_for_torch)
            # print(output['heatmap'].shape)
            # print('output: ', output['heatmap'])
            # print(output.size())
        # print('input_for_torch shape: ', img_crop.shape)
        # print('HEIGHT: ', HEIGHT)
        if isinstance(output, dict):
            if USE_HEATMAP:
                preds = output['heatmap']
            else:
                preds = output['regress'] * WIDTH
            if args.multi_eye_branch:
                preds_eye = output['s4_regress'] * WIDTH
                preds_eye = preds_eye.view(-1, 2)
                # preds = preds.reshape([num_joints, -1])
                preds_eye = preds_eye.squeeze().cpu().numpy()
                preds_eye_inv = preds_eye.copy()
                for i in range(preds_eye.shape[0]):
                    preds_eye_inv[i, 0:2] = affine_transform(preds_eye[i, 0:2], trans_face_inv)
        else:
            if USE_HEATMAP:
                preds = output['heatmap']
            else:
                preds = output['regress'] * WIDTH
        if USE_BOUNDARY_MAP:
            preds_bd = preds[:, -1, ...]
            preds_bd_vis = preds_bd.clone().squeeze().detach().cpu().numpy() * 255.0
            preds_bd_vis = preds_bd_vis.astype(np.uint8)
            preds = preds[:, :-1, ...]

        if USE_HEATMAP:
            if args.use_sbr_model:
                preds = preds.squeeze(0)
                preds, scores = find_tensor_peak_batch(preds, 4, 2)
                preds = preds.detach().cpu().numpy()
                print("preds: ", preds.shape)
                print("scores: ", scores.mean())
            else:
                if cfg.MODEL.HEATMAP_DM:
                    preds = gaussian_modulation_torch(preds, cfg.MODEL.FACE_SIGMA)
                preds, scores = demo_preds_function(cfg, preds.detach().cpu().numpy(), cfg.MODEL.FACE_SIGMA)
                preds *= cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.HEATMAP_SIZE[0]
                preds = preds.squeeze(0)
            # print("scores.mean(): ", scores.mean())
            if scores.mean() < args.thresh:
                print("frame idx: {} low face kpts estimation scores {}.".format(counter, scores.mean()))
                low_score_idxes.append([counter, scores.mean()])
                if args.testMode != 'filepath':
                    hasFrame, frame = cap.read()
                else:
                    if len(pic_paths) > 0:
                        frame_path = pic_paths.pop(0)
                        frame = cv2.imread(frame_path)
                        hasFrame = True
                    else:
                        frame = None
                        hasFrame = False
                if args.dst_size and hasFrame:
                    frame, _, _ = preprocess_face_detect(frame, (args.dst_size, args.dst_size), trans_color=False)
                counter += 1
                continue
            # print("preds shape: ", preds.shape)
        else:
            preds = preds.view(-1, 2)
            # preds = preds.reshape([num_joints, -1])
            preds = preds.squeeze().cpu().numpy()

        # preds_inv = preds.copy()

        if args.smoother:
            # print("preds: ", preds)
            pred_norm = preds[33].copy()
            preds_sm = preds - pred_norm
            preds_smooth = preds_sm / WIDTH
            for i in range(RECEPTIVE_FIELD):
                dq.append(preds_smooth)
            input_smooth = torch.from_numpy(np.array(dq)).unsqueeze(0).float().to(device)
            # print("input_smooth: ", input_smooth)
            output_smooth = model_smooth(input_smooth)
            output_smooth_arr = output_smooth.cpu().numpy()
            output_smooth_arr = output_smooth_arr * WIDTH + pred_norm
            # print("output_smooth_arr: ", output_smooth_arr)

            output_smooth_show = output_smooth_arr.squeeze()
            img_smooth_show = draw_circle_map(img_crop, output_smooth_show.astype(np.int32))
            # cv2.waitKey()
        preds_inv = preds.copy()
        for i in range(num_joints):
            preds_inv[i, 0:2] = affine_transform(preds[i, 0:2], trans_face_inv)
        lmk = preds_inv
        if args.use_optical_flow:
            good_new = lmk.copy().reshape(-1, 1, 2)
            if frame_counter == 0:
                lmk_bf_flow = lmk.copy().reshape(-1, 1, 2)
            else:
                # Create some random colors
                img_for_flow = img_crop_origin.copy()
                color = (0, 255, 255)

                frame_gray = cv2.cvtColor(img_for_flow, cv2.COLOR_BGR2GRAY)
                # calculate optical flow
                lmk_aft_flow, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                       frame_gray,
                                                       lmk_bf_flow, None,)
                                                       # **lk_params)
                diff = lmk_aft_flow - good_new
                diff = diff.squeeze()
                norm = np.linalg.norm(diff, axis=1)
                norm = norm[:, np.newaxis]
                good_new[norm < 4] = lmk_aft_flow[norm < 4]

                for i, new in enumerate(good_new):
                    a, b = new.ravel()
                    frame_flow = cv2.circle(img_for_flow, (round(a), round(b)), 2,
                                            color, -1)

                lmk_bf_flow = good_new.reshape(-1, 1, 2)
                lmk = good_new.squeeze()
            # frame_counter += 1

        """ eye detection """
        img_eyes_gray, img_eyes, trans_eyes = transform_eye_imgs(img_crop_origin, lmk, args.face_type)
        img_eyes_l = img_eyes[0]
        img_eyes_r = img_eyes[1]
        # img_eyes_l = cv2.cvtColor(img_eyes_l, cv2.COLOR_RGB2BGR)
        # img_eyes_r = cv2.cvtColor(img_eyes_r, cv2.COLOR_RGB2BGR)
        input_eyes = torch.from_numpy(img_eyes_gray).to(device)

        output_eyes = eye_model(input_eyes)
        if isinstance(output_eyes, dict):
            if args.eye_heatmap_decode:
                output_eyes = output_eyes['heatmap']
            else:
                output_eyes = output_eyes['regress']
        if args.eye_heatmap_decode:
            sigma = cfg.MODEL.FACE_SIGMA
            # print("output_eyes shape: ", output_eyes.shape)
            output_eyes_dm = output_eyes.clone()
            output_eyes_l_dm = output_eyes_dm[0, ...].unsqueeze(0)
            output_eyes_r_dm = output_eyes_dm[1, ...].unsqueeze(0)
            # output_eyes_l_dm = gaussian_modulation_torch(output_eyes_l_dm, sigma)
            # output_eyes_r_dm = gaussian_modulation_torch(output_eyes_r_dm, sigma)
            if cfg.MODEL.HEATMAP_DM:
                output_eyes_l_dm = gaussian_modulation_torch(output_eyes_l_dm, cfg_eye.MODEL.FACE_SIGMA)
                output_eyes_r_dm = gaussian_modulation_torch(output_eyes_r_dm, cfg_eye.MODEL.FACE_SIGMA)
            output_eyes_l, eyes_l_maxvals = demo_preds_function(cfg_eye, output_eyes_l_dm.detach().cpu().numpy(), cfg_eye.MODEL.FACE_SIGMA)
            output_eyes_r, eyes_r_maxvals = demo_preds_function(cfg_eye, output_eyes_r_dm.detach().cpu().numpy(), cfg_eye.MODEL.FACE_SIGMA)
            output_eyes_l = output_eyes_l.squeeze(0) * cfg_eye.MODEL.IMAGE_SIZE[0] // cfg_eye.MODEL.HEATMAP_SIZE[0]
            output_eyes_r = output_eyes_r.squeeze(0) * cfg_eye.MODEL.IMAGE_SIZE[0] // cfg_eye.MODEL.HEATMAP_SIZE[0]
            # print("output_eyes_l: ", output_eyes_l)
            # print("output_eyes_r: ", output_eyes_r)

            # print("eyes_l_maxvals shape: ", eyes_l_maxvals.shape)
            # print("eyes_l_maxvals mean: ", eyes_l_maxvals.mean())
            # print("eyes_r_maxvals mean: ", eyes_r_maxvals.mean())

            # print("eye ball maxvals: {}  {}".format(eyes_l_maxvals[:, -1], eyes_r_maxvals[:, -1]))
        else:
            output_eyes[..., 0] = output_eyes[..., 0] * EYE_WIDTH
            output_eyes[..., 1] = output_eyes[..., 1] * EYE_HEIGHT

            output_eyes = output_eyes.view(-1, num_eye_points, 2).cpu().numpy()
            output_eyes_l = output_eyes[0]
            output_eyes_r = output_eyes[1]

        if args.flip_eye:
            output_eyes_l = fliplr_eye_joints(output_eyes_l, EYE_WIDTH, dataset=args.face_type)

        output_eyes_l_ori = output_eyes_l.copy()
        output_eyes_r_ori = output_eyes_r.copy()
        for i in range(num_eye_points):
            output_eyes_l_ori[i, 0:2] = affine_transform(output_eyes_l[i, 0:2], trans_eyes[0])
            output_eyes_r_ori[i, 0:2] = affine_transform(output_eyes_r[i, 0:2], trans_eyes[1])

        if args.use_optical_flow:
            good_new_eye = np.vstack([output_eyes_l_ori, output_eyes_r_ori])
            good_new_eye = good_new_eye.reshape(-1, 1, 2)

            if frame_counter == 0:
                lmk_bf_flow_eye = good_new_eye.copy().reshape(-1, 1, 2)
            else:
                # Create some random colors
                # img_for_flow = img_crop_origin.copy()
                color = (0, 0, 255)

                # frame_gray = cv2.cvtColor(img_for_flow, cv2.COLOR_BGR2GRAY)
                # calculate optical flow
                lmk_aft_flow_eye, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                       frame_gray,
                                                       lmk_bf_flow_eye, None,)
                                                       # **lk_params)
                # Select good points
                diff = lmk_aft_flow_eye - good_new_eye
                diff = diff.squeeze()
                norm = np.linalg.norm(diff, axis=1)
                norm = norm[:, np.newaxis]
                good_new_eye[norm < 4] = lmk_aft_flow_eye[norm < 4]
                # print("good_new shape: ", good_new)
                # draw the tracks
                for i, new in enumerate(good_new_eye):
                    a, b = new.ravel()

                    frame_flow = cv2.circle(frame_flow, (round(a), round(b)), 2,
                                            color, -1)

                # Updating Previous frame and points
                old_gray = frame_gray.copy()
                lmk_bf_flow_eye = good_new_eye.reshape(-1, 1, 2)
                output_eyes_l_ori = good_new_eye[:num_eye_points].squeeze()
                output_eyes_r_ori = good_new_eye[num_eye_points:].squeeze()
            frame_counter += 1

        face_landmarks_68 = lmk.copy()
        if args.face_type == 'WLFW':
            face_landmarks_68 = face_landmarks_68[face_kpts_98_to_68, :]
        num_kpts = face_landmarks_68.shape[0] * face_landmarks_68.shape[1]
        # kpts_print = preds_inv.transpose((0, 1)).reshape(-1)
        kpts_print = face_landmarks_68.reshape(-1)
        # for i in range(num_kpts):
        #     print("{:.2f}f, ".format(kpts_print[i]), end=' ')
        # exit()
        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(face_landmarks_68)
        # print("orignal pose: ", pose)
        r_vec = pose[0]
        t_vec = pose[1]

        r_mat, _ = cv2.Rodrigues(r_vec)
        pose_mat = cv2.hconcat((r_mat, t_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        # print("euler_angles: ", euler_angles)
        # Stabilize the pose.
        steady_pose = []
        pose_np = np.array(pose).flatten()

        steady_pose_euro = []
        # for i in range(6):
        #     kp_aft_filter = kp_filters[i](pose_np[i], counter)
        #     steady_pose_euro.append(kp_aft_filter)

        for value, ps_stb in zip(pose_np, pose_stabilizers):
            ps_stb.update([value])
            steady_pose.append(ps_stb.state[0])
        steady_pose = np.reshape(steady_pose, (-1, 3))
        steady_pose = np.reshape(steady_pose, (-1, 3))

        # if args.outputVidPath:
        #     save_kp2d_to_json(preds_bf_filter, scores=maxvals,
        #                       dump_dir=os.path.join(JSON_OUTPUT_DIR, 'frame_{:05d}.json'.format(counter)))

        maxvals_face = np.ones(preds.shape[0])
        maxvals_eye = np.ones(output_eyes_l.shape[0])

        face_landmarks_show = lmk.copy()
        # face_landmarks_show = face_landmarks_show[face_kpts_98_to_68, :]
        # face_landmarks_show[96, :] = 0
        # face_landmarks_show[97, :] = 0
        face_landmarks_show = np.vstack([face_landmarks_show[:36, :], face_landmarks_show[48:, :]])

        img_crop_origin = cv2.cvtColor(img_crop_origin, cv2.COLOR_RGB2BGR)
        # face green
        # img_crop_origin = draw_circle(img_crop_origin, face_landmarks_show, maxvals_face,
        #                             thresh=0.3, color=(0, 255, 0))
        img_crop_origin = draw_circle(img_crop_origin, lmk, maxvals_face,
                                    thresh=0.0, color=(0, 255, 0))

        if args.flip_eye:
            img_eyes_l = np.fliplr(img_eyes_l)
        img_eye_l_show = draw_circle(img_eyes_l, output_eyes_l, maxvals_eye,
                                   thresh=0.0, color=(0, 255, 0))

        img_eye_r_show = draw_circle(img_eyes_r, output_eyes_r, maxvals_eye,
                                   thresh=0.0, color=(0, 255, 0))

        # eye blue
        # print("output_eyes_l_ori.shape: ", output_eyes_l_ori.shape)
        # output_eyes_l_300W = output_eyes_l_ori[[0, 1, 3, 4, 5, 7], ...]
        # output_eyes_r_300W = output_eyes_r_ori[[0, 1, 3, 4, 5, 7], ...]
        # img_crop_origin = draw_circle(img_crop_origin, output_eyes_l_300W, maxvals_eye,
        #                            thresh=0.3, color=(255, 0, 0))

        # img_crop_origin = draw_circle(img_crop_origin, output_eyes_r_300W, maxvals_eye,
        #                            thresh=0.3, color=(255, 0, 0))

        # img_crop_origin = draw_circle(img_crop_origin, output_eyes_l_ori, maxvals_eye,
        #                            thresh=0.3, color=(255, 0, 0))

        # img_crop_origin = draw_circle(img_crop_origin, output_eyes_r_ori, maxvals_eye,
        #                            thresh=0.3, color=(255, 0, 0))
        # # iris red
        # img_crop_origin = draw_circle(img_crop_origin, output_eyes_l_ori[-1, :][np.newaxis, :], maxvals_eye,
        #                            thresh=0.3, color=(0, 0, 255))

        # img_crop_origin = draw_circle(img_crop_origin, output_eyes_r_ori[-1, :][np.newaxis, :], maxvals_eye,
        #                            thresh=0.3, color=(0, 0, 255))

        if args.multi_eye_branch:
            scores = np.ones(preds_eye_inv.shape[0])
            img_crop_origin = draw_circle(img_crop_origin, preds_eye_inv, scores, color=(255, 0, 0))

        if args.draw_pose_annot:
            # Uncomment following line to draw pose annotation on frame.
            pose_estimator.draw_annotation_box(
                img_crop_origin, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     img_crop_origin, pose[0], pose[1], color=(255, 0, 0))
            # pose_estimator.draw_annotation_box(
            #     img_crop_origin, steady_pose[0], steady_pose[1], color=(0, 255, 0))
            # pose_estimator.draw_annotation_box(
            #     img_crop_origin, steady_pose_euro[0], steady_pose_euro[1], color=(0, 0, 255))

            # Uncomment following line to draw head axes on frame.
            pose_estimator.draw_axes(img_crop_origin, pose[0], pose[1])
        
        if args.outputCropPath:
            if not os.path.exists(args.outputCropPath):
                os.makedirs(args.outputCropPath)
            img_crop_save = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.outputCropPath, "{:06d}.jpg".format(counter)), img_crop_save)

        if args.outputVidPath:
            # print("img_crop_origin shape: ", img_crop_origin.shape)
            out_vid.write(img_crop_origin)

        if args.outputSavePath:
            save_dict = {
                'pose_mat': pose_mat,
                'pose_np': pose_np,
                'steady_pose': steady_pose,
                'face_landmarks_show': face_landmarks_show
            }
            
            out_lms_path = os.path.join(args.outputSavePath, 'lms')
            out_pkl_path = os.path.join(args.outputSavePath, 'pkl')
            if not os.path.exists(out_lms_path):
                os.makedirs(out_lms_path)
            if not os.path.exists(out_pkl_path):
                os.makedirs(out_pkl_path)
            lands = lmk.copy() # preds[0].reshape(-1, 2)
            np.savetxt(os.path.join(out_lms_path, '{:06d}.lms'.format(counter)), lands, '%f')
            with open(os.path.join(out_pkl_path, '{:06d}.pkl'.format(counter)), 'wb') as f:
                pkl.dump(save_dict, f)
        # if (counter > 0):
        #     infer_time = time.time() - t_start
        #     # print('Inference Time:{}'.format(infer_time))
        #     infer_times.append(infer_time)
        counter += 1
        t_start = time.time()
        img_show = img_crop_origin

        # if img_show is None:
        #     print("Cant Load Image")
        # if img_eye_l_show is None:
        #     print("Cant Load img_eye_l_show Image")
        # if img_eye_r_show is None:
        #     print("Cant Load  img_eye_r_show Image")

        key = cv2.waitKey(1)  # ms
        if key == 27:  # esc
            cv2.destroyAllWindows()
            break
        if args.testMode != 'filepath':
            hasFrame, frame = cap.read()
        else:
            if len(pic_paths) > 0:
                frame_path = pic_paths.pop(0)
                frame = cv2.imread(frame_path)
                hasFrame = True
            else:
                frame = None
                hasFrame = False
        if args.dst_size and hasFrame:
            frame, _, _ = preprocess_face_detect(frame, (args.dst_size, args.dst_size), trans_color=False)
            
    if args.testMode != 'filepath':
        cap.release()
    # print('Average Inference Time:{}'.format(np.mean(infer_time)))
    if args.outputVidPath:
        out_vid.release()
    print("low_score_idxes: ", low_score_idxes)
