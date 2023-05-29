import cv2
import numpy as np
# import face_alignment
from skimage import io
import torch
import torch.nn.functional as F
import json
import shutil
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from subprocess import Popen
from glob import glob
from scipy.ndimage import gaussian_filter1d
import argparse

BG_LABEL = 255
HEAD_LABEL = 29
NECK_LABEL = 150
BODY_LABEL = 76

dir_realpath = os.path.dirname(os.path.realpath(__file__))
def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def landmark_smooth_2d(pts2d, smooth_sigma=0, area='all'):
    ''' smooth the input 2d landmarks using gaussian filters on each dimension.
    Args:
        pts3d: [N, 68, 2]
    '''
    # per-landmark smooth
    if not smooth_sigma == 0:
        if area == 'all':
            pts2d = gaussian_filter1d(pts2d.reshape(-1, 68*2), smooth_sigma, axis=0).reshape(-1, 68, 2)
        elif area == 'wo_mouth':
            pts2d_wo_mouth = pts2d[:, :47, :].copy()
            pts2d_wo_mouth = gaussian_filter1d(pts2d_wo_mouth.reshape(-1, 47*2), smooth_sigma, axis=0).reshape(-1, 47, 2)
            pts2d[:, :47, :] = pts2d_wo_mouth
        elif area == 'only_jaw':
            pts2d_wo_mouth = pts2d[:, :16, :].copy()
            pts2d_wo_mouth = gaussian_filter1d(pts2d_wo_mouth.reshape(-1, 16*2), smooth_sigma, axis=0).reshape(-1, 16, 2)
            pts2d[:, :16, :] = pts2d_wo_mouth

    return pts2d

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
                    default='obama', help='identity of target person')
parser.add_argument('--step', type=int,
                    default=0, help='step for running')
parser.add_argument('--no_use_opFlow4FaceAlign', action='store_true',
                    help='Do not use optical flow in face alignment.')
parser.add_argument('--smooth_lms', action='store_true')
parser.add_argument('--bc_pic_path', type=str, default='')
parser.add_argument('--dst_size', type=int, default=-1)

args = parser.parse_args()
id = args.id
vid_file = os.path.join('dataset', 'vids', id+'.mp4')
# if 'fps25' not in vid_file:
vid_file_fps25 = os.path.join('dataset', 'vids', '{}_fps25.mp4'.format(id))
# else:
#     vid_file_fps25 = vid_file
# if os.path.exists(vid_file_fps25):
#     vid_file = vid_file_fps25
#     print("vid_file: ", vid_file)
    
if not os.path.isfile(vid_file):
    print('no video')
    exit()


id_dir = os.path.join('dataset', id)
Path(id_dir).mkdir(parents=True, exist_ok=True)
face_align_dir = os.path.join('dataset', id, 'face_alignment')
Path(face_align_dir).mkdir(parents=True, exist_ok=True)
ori_imgs_dir = os.path.join('dataset', id, 'ori_imgs')
Path(ori_imgs_dir).mkdir(parents=True, exist_ok=True)
parsing_dir = os.path.join(id_dir, 'parsing')
Path(parsing_dir).mkdir(parents=True, exist_ok=True)
head_imgs_dir = os.path.join('dataset', id, 'head_imgs')
Path(head_imgs_dir).mkdir(parents=True, exist_ok=True)
com_imgs_dir = os.path.join('dataset', id, 'com_imgs')
Path(com_imgs_dir).mkdir(parents=True, exist_ok=True)

mask_dir = os.path.join(id_dir, 'face_mask')
Path(mask_dir).mkdir(parents=True, exist_ok=True)
flow_dir = os.path.join(id_dir, 'flow_result')
Path(flow_dir).mkdir(parents=True, exist_ok=True)

running_step = args.step

# Step -1: convert the input video to 25 fps
if running_step == -1:
    print('--- Step-1: convert the input video to 25 fps ---')
    
    cvt_fps_cmd = "ffmpeg -i {} -c:v libx264 -r 25 -crf 0 -c:a aac -strict -2  {}".format(vid_file, vid_file_fps25)
    print(cvt_fps_cmd)
    p = Popen(cvt_fps_cmd, shell=True)
    p.wait()
    
if os.path.exists(vid_file_fps25):
    vid_file = vid_file_fps25
print("vid_file: ", vid_file)


# Step 0: extract wav & deepspeech feature, better run in terminal to parallel with
# below commands since this may take a few minutes
if running_step == 0:
    print('--- Step0: extract deepspeech feature ---')
    wav_file = os.path.join(id_dir, 'aud.wav')
    extract_wav_cmd = 'ffmpeg -i ' + vid_file + ' -f wav -ar 16000 ' + wav_file
    os.system(extract_wav_cmd)
    extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + id_dir
    os.system(extract_ds_cmd)
    exit()

# Step 1: extract images
if running_step == 1:
    print('--- Step1: extract images from vids ---')
    cap = cv2.VideoCapture(vid_file)
    frame_num = 0
    while(True):
        _, frame = cap.read()
        if frame is None:
            break
        if args.dst_size > 0:
            frame = cv2.resize(frame, (args.dst_size, args.dst_size))
        cv2.imwrite(os.path.join(ori_imgs_dir, '{:06d}.jpg'.format(frame_num)), frame)
        frame_num = frame_num + 1
    cap.release()
    exit()

# Step 2: detect lands
if running_step == 2:
    print('--- Step 2: detect landmarks ---')
    vid_realpath = os.path.join(dir_realpath, "../", vid_file)
    ori_img_realpath = os.path.join(dir_realpath, "../", ori_imgs_dir)
    face_align_dir = os.path.join(dir_realpath, "../", face_align_dir)

    if not args.no_use_opFlow4FaceAlign:
        face_align_dir += '_opFlow'

    face_align_cmd = "cd data_util/face-alignment\n \
    python demo_face_eye_detectPerframe_save.py --cfg experiments/300w_lp_menpo2D/hrnet_hm.yaml \
    --cfg_eye experiments/eye_300w_menpo/ghostnet_en_de.yaml \
    --testModelPath ./models/face_lms_68kpts_hrnet.pth \
    --testEyeModelPath ./models/eye_lms_6kpts.pth \
    --inputPath {0} \
    --outputVidPath {1} \
    --outputSavePath {1} \
    --testMode filepath \
    --eye_heatmap_decode \
    --face_type 300W ".format(ori_img_realpath, face_align_dir)
    if not args.no_use_opFlow4FaceAlign:
        face_align_cmd += "--use_optical_flow \n cd ../../"
    else:
        face_align_cmd += "\n cd../../"
    p = Popen(face_align_cmd, shell=True)
    p.wait()
    
    out_lms_dir = os.path.join(face_align_dir, "lms")
    lms_filelist = sorted(glob(os.path.join(out_lms_dir, "*.lms")))
    lms_list = []
    if args.smooth_lms:
        for path_lms in lms_filelist:
            lms = np.loadtxt(path_lms).astype(np.float32)
            lms_list.append(lms)

        lms_arr = np.array(lms_list)
        lms_arr = landmark_smooth_2d(lms_arr, 1.5, 'only_jaw')
        for i, path_lms in enumerate(lms_filelist):
            np.savetxt(path_lms, lms_arr[i], '%f')
    
    for filepath in lms_filelist:
        shutil.copy(filepath, ori_imgs_dir)
        
max_frame_num = 100000
valid_img_ids = []
for i in range(max_frame_num):
    if os.path.isfile(os.path.join(ori_imgs_dir, '{:06d}.lms'.format(i))):
        valid_img_ids.append(i)
print("valid_img_ids: ", valid_img_ids)
valid_img_num = len(valid_img_ids)
tmp_img = cv2.imread(os.path.join(ori_imgs_dir, '{:06d}.jpg'.format(valid_img_ids[0])))
h, w = tmp_img.shape[0], tmp_img.shape[1]

# Step 3: face parsing
if running_step == 3:
    print('--- Step 3: face parsing ---')
    face_parsing_cmd = 'python data_util/face_parsing/test.py --respath=dataset/' + \
        id + '/parsing --imgpath=dataset/' + id + '/ori_imgs'
    os.system(face_parsing_cmd)

# Step 4: extract bc image
if running_step == 4:
    if not args.bc_pic_path:
        print('--- Step 4: extract background image ---')
        sel_ids = np.array(valid_img_ids)[np.arange(0, valid_img_num, 20)]
        all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
        distss = []
        for i in sel_ids:
            parse_img = cv2.imread(os.path.join(id_dir, 'parsing', '{:06d}.png'.format(i)))
            bg = (parse_img[..., 0] == 255) & (
                parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
            fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
            dists, _ = nbrs.kneighbors(all_xys)
            distss.append(dists)
        distss = np.stack(distss)
        print(distss.shape)
        max_dist = np.max(distss, 0)
        max_id = np.argmax(distss, 0)
        bc_pixs = max_dist > 5
        bc_pixs_id = np.nonzero(bc_pixs)
        bc_ids = max_id[bc_pixs]
        imgs = []
        num_pixs = distss.shape[1]
        for i in sel_ids:
            img = cv2.imread(os.path.join(ori_imgs_dir, '{:06d}.jpg'.format(i)))
            imgs.append(img)
        imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
        bc_img = np.zeros((h*w, 3), dtype=np.uint8)
        bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
        bc_img = bc_img.reshape(h, w, 3)
        max_dist = max_dist.reshape(h, w)
        bc_pixs = max_dist > 5
        bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
        fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        distances, indices = nbrs.kneighbors(bg_xys)
        bg_fg_xys = fg_xys[indices[:, 0]]
        print(fg_xys.shape)
        print(np.max(bg_fg_xys), np.min(bg_fg_xys))
        bc_img[bg_xys[:, 0], bg_xys[:, 1],
            :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
        cv2.imwrite(os.path.join(id_dir, 'bc.jpg'), bc_img)
    else:
        print('--- Step 4: use target background image ---')
        bc_img = cv2.imread(args.bc_pic_path)
        h_bc, w_bc = bc_img.shape[:2]
        if h_bc != h or w_bc != w:
            bc_img = cv2.resize(bc_img, (h, w))
        cv2.imwrite(os.path.join(id_dir, 'bc.jpg'), bc_img)

# Step 5: save training images
if running_step == 5:
    print('--- Step 5: save training images ---')
    bc_img = cv2.imread(os.path.join(id_dir, 'bc.jpg'))
    for i in valid_img_ids:
        parsing_img = cv2.imread(os.path.join(parsing_dir, '{:06d}.png'.format(i)))
        mask_img = np.zeros_like(parsing_img)
        head_part = (parsing_img[:, :, 0] == 255) & (
            parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 0)
        bc_part = (parsing_img[:, :, 0] == 255) & (
            parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
        img = cv2.imread(os.path.join(ori_imgs_dir, '{:06d}.jpg'.format(i)))
        img[bc_part] = bc_img[bc_part]
        cv2.imwrite(os.path.join(com_imgs_dir, '{:06d}.jpg'.format(i)), img)
        img[~head_part] = bc_img[~head_part]
        cv2.imwrite(os.path.join(head_imgs_dir, '{:06d}.jpg'.format(i)), img)
        mask_img[head_part, :] = 255
        cv2.imwrite(os.path.join(mask_dir, '{:06d}.png'.format(i)), mask_img)

# Step 6: Estimate dense optical flow
if running_step == 6:
    torch.cuda.empty_cache()
    print('--- Step 6: Estimate dense optical flow ---')
    ref_id = 18

    with open(os.path.join(id_dir, 'flow_list.txt'), 'w') as file:
        for i in range(0, valid_img_num):
            file.write('dataset/' + id + '/ori_imgs/' + '{:06d}.jpg '.format(ref_id) +
                    'dataset/' + id + '/face_mask/' + '{:06d}.png '.format(ref_id) +
                    'dataset/' + id + '/ori_imgs/' + '{:06d}.jpg '.format(i) +
                    'dataset/' + id + '/face_mask/' + '{:06d}.png\n'.format(i))
        file.close()
    est_flow_cmd = 'python data_util/UNFaceFlow/test_flow.py --datapath=dataset/' + id + '/flow_list.txt ' + \
        '--savepath=dataset/' + id + '/flow_result' + \
        ' --width=' + str(w) + ' --height=' + str(h)
    os.system(est_flow_cmd)
    face_img = cv2.imread(os.path.join(ori_imgs_dir, '{:06d}.jpg'.format(ref_id)))
    face_img_mask = cv2.imread(os.path.join(mask_dir, '{:06d}.png'.format(ref_id)))
    rigid_mask = face_img_mask[..., 0] > 250
    rigid_num = np.sum(rigid_mask)
    flow_frame_num = 2500
    flow_frame_num = min(flow_frame_num, valid_img_num)
    rigid_flow = np.zeros((flow_frame_num, 2, rigid_num), np.float32)
    for i in range(flow_frame_num):
        # flow = np.load(os.path.join(flow_dir, str(ref_id) +
        #                             '_' + str(valid_img_ids[i]) + '.npy'))
        flow = np.load(os.path.join(flow_dir, 
                                    '{:06d}_{:06d}.npy'.format(ref_id, valid_img_ids[i])))
        rigid_flow[i] = flow[:, rigid_mask]
    rigid_flow = rigid_flow.transpose((2, 1, 0))
    rigid_flow = torch.as_tensor(rigid_flow).cuda()
    lap_kernel = torch.Tensor(
        (-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().cuda()
    flow_lap = F.conv1d(
        rigid_flow.reshape(-1, 1, rigid_flow.shape[-1]), lap_kernel)
    flow_lap = flow_lap.view(rigid_flow.shape[0], 2, -1)
    flow_lap = torch.norm(flow_lap, dim=1)
    valid_frame = torch.mean(flow_lap, dim=0) < (torch.mean(flow_lap)*3)
    flow_lap = flow_lap[:, valid_frame]
    rigid_flow_mean = torch.mean(flow_lap, dim=1)
    rigid_flow_show = (rigid_flow_mean-torch.min(rigid_flow_mean)) / \
        (torch.max(rigid_flow_mean)-torch.min(rigid_flow_mean)) * 255
    rigid_flow_show = rigid_flow_show.byte().cpu().numpy()
    rigid_flow_img = np.zeros((h, w, 1), dtype=np.uint8)
    rigid_flow_img[...] = 255
    rigid_flow_img[rigid_mask, 0] = rigid_flow_show
    cv2.imwrite(os.path.join(id_dir, 'rigid_flow.jpg'), rigid_flow_img)

    win_size, d_size = 5, 5
    sel_xys = np.zeros((h, w), dtype=np.int32)
    xys = []
    for y in range(0, h-win_size, win_size):
        for x in range(0, w-win_size, win_size):
            min_v = int(40)
            id_x = -1
            id_y = -1
            for dy in range(0, win_size):
                for dx in range(0, win_size):
                    if rigid_flow_img[y+dy, x+dx, 0] < min_v:
                        min_v = rigid_flow_img[y+dy, x+dx, 0]
                        id_x = x+dx
                        id_y = y+dy
            if id_x >= 0:
                if(np.sum(sel_xys[id_y-d_size:id_y+d_size+1, id_x-d_size:id_x+d_size+1]) == 0):
                    cv2.circle(face_img, (id_x, id_y), 1, (255, 0, 0))
                    xys.append(np.array((id_x, id_y), np.int32))
                    sel_xys[id_y, id_x] = 1
    xys = np.array(xys)
    cv2.imwrite(os.path.join(id_dir, 'keypts.jpg'), face_img)
    np.savetxt(os.path.join(id_dir, 'keypoints.txt'), xys, '%d')
    key_xys = np.loadtxt(os.path.join(id_dir, 'keypoints.txt'), np.int32)
    track_xys = np.zeros((valid_img_num, key_xys.shape[0], 2), dtype=np.float32)
    track_dir = os.path.join('dataset', id, 'flow_result')
    track_paths = sorted(glob(os.path.join(track_dir, '*.npy')))

    for i, path in enumerate(track_paths):

        flow = np.load(path)
        for j in range(key_xys.shape[0]):
            x = key_xys[j, 0]
            y = key_xys[j, 1]
            track_xys[i, j, 0] = x+flow[0, y, x]
            track_xys[i, j, 1] = y+flow[1, y, x]
    np.save(os.path.join(id_dir, 'track_xys.npy'), track_xys)

# Step 7: estimate head pose
if running_step == 7:
    print('--- Step 7: Estimate Head Pose ---')
    est_pose_cmd = 'python data_util/face_tracking_ba/face_tracker.py --idname=' + \
        id + ' --img_h=' + str(h) + ' --img_w=' + str(w) + \
        ' --frame_num=' + str(max_frame_num)
    os.system(est_pose_cmd)
    exit()

# Step 8: BA
if running_step == 8:
    print('--- Step 8: Bundle Adjustment ---')
    bundle_adjustmen_cmd = 'python data_util/bundle_adjustment/bundle_adjustment.py --id=' + \
        id + ' --img_h=' + \
        str(h) + ' --img_w=' + str(w)
    os.system(bundle_adjustmen_cmd)

# Step 9: save transform param & write config file
if running_step == 9:
    print('--- Step 9: Save Transform Param ---')
    params_dict = torch.load(os.path.join(id_dir, 'bundle_adjustment.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans']
    valid_num = euler_angle.shape[0]
    train_val_split = int(valid_num*10/11) # 10000
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)
    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())
    for i in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[i]
        save_id = save_ids[i]
        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = int(valid_img_ids[i])
            frame_dict['aud_id'] = int(valid_img_ids[i])
            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]
            frame_dict['transform_matrix'] = pose.numpy().tolist()
            lms = np.loadtxt(os.path.join(
                ori_imgs_dir, '{:06d}.lms'.format(valid_img_ids[i])))
            min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
            cx = int((min_x+max_x)/2.0)
            cy = int(lms[27, 1])
            h_w = int((max_x-cx)*1.5)
            h_h = int((lms[8, 1]-cy)*1.15)
            rect_x = cx - h_w
            rect_y = cy - h_h
            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            rect_w = min(w-1-rect_x, 2*h_w)
            rect_h = min(h-1-rect_y, 2*h_h)
            rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
            frame_dict['face_rect'] = rect.tolist()
            transform_dict['frames'].append(frame_dict)
        with open(os.path.join(id_dir, 'transforms_' + save_id + '_ba.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    testskip = int(val_ids.shape[0]/7)

    HeadNeRF_config_file = os.path.join(id_dir, 'HeadNeRF_config_ba.txt')
    with open(HeadNeRF_config_file, 'w') as file:
        file.write('expname = ' + id + '_head\n')
        file.write('datadir = ' + os.path.join(dir_path, 'dataset', id) + '\n')
        file.write('basedir = ' + os.path.join(dir_path,
                                            'dataset', id, 'logs') + '\n')
        file.write('near = ' + str(mean_z-0.2) + '\n')
        file.write('far = ' + str(mean_z+0.4) + '\n')
        file.write('testskip = ' + str(1) + '\n')
    Path(os.path.join(dir_path, 'dataset', id, 'logs', id + '_head')
        ).mkdir(parents=True, exist_ok=True)

    ComNeRF_config_file = os.path.join(id_dir, 'TorsoNeRF_config_ba.txt')
    with open(ComNeRF_config_file, 'w') as file:
        file.write('expname = ' + id + '_com\n')
        file.write('datadir = ' + os.path.join(dir_path, 'dataset', id) + '\n')
        file.write('basedir = ' + os.path.join(dir_path,
                                            'dataset', id, 'logs') + '\n')
        file.write('near = ' + str(mean_z-0.2) + '\n')
        file.write('far = ' + str(mean_z+0.4) + '\n')
        file.write('testskip = ' + str(1) + '\n')
    Path(os.path.join(dir_path, 'dataset', id, 'logs', id + '_com')
        ).mkdir(parents=True, exist_ok=True)

    ComNeRFTest_config_file = os.path.join(id_dir, 'TorsoNeRFTest_config_ba.txt')
    with open(ComNeRFTest_config_file, 'w') as file:
        file.write('expname = ' + id + '_com\n')
        file.write('datadir = ' + os.path.join(dir_path, 'dataset', id) + '\n')
        file.write('basedir = ' + os.path.join(dir_path,
                                            'dataset', id, 'logs') + '\n')
        file.write('near = ' + str(mean_z-0.2) + '\n')
        file.write('far = ' + str(mean_z+0.4) + '\n')
        file.write('with_test = ' + str(1) + '\n')

    print(id + ' data processed done!')

# Step 10: extract face disentangle parameters
if running_step == 10:
    print('--- Step 10: Extract Face Disentangle Parameters ---')
    id_realpath = os.path.join(dir_realpath, "../", id_dir)
    face_dis_cmd1 = "python data_util/extract_exp_from_trackPt.py --input_path {0}/track_params_ba.pt \
                    --save_path {0}/raw_exps/".format(id_dir)

    p = Popen(face_dis_cmd1, shell=True)
    p.wait()
    face_dis_cmd2 = "cd data_util/face_disentangle_3dmm\n \
    CUDA_VISIBLE_DEVICES=0, python test_model.py --ckpt_path pretrained_models/64_32 \
        --test_path {0}/raw_exps \
        --save_path {0} \
        --dim_o 64 \
        --dim_m 32\n \
    cd ../../".format(id_realpath)
    p = Popen(face_dis_cmd2, shell=True)
    p.wait()

# Step 11: wav2exp
if running_step == 11:
    print('--- Step 11: Wav to Expression ---')
    id_realpath = os.path.join(dir_realpath, "../", id_dir)
    wav2exp_cmd = "cd data_util/wav2exp\n \
    python test_w2l_audio.py --input_path {0}/aud.wav \
        --save_path {0}/{1}_aud.pt\n \
    cd ../../".format(id_realpath, id)
    p = Popen(wav2exp_cmd, shell=True)
    p.wait()