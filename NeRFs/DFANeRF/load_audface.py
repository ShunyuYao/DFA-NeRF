import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from glob import glob


def load_audface_data_split(basedir, testskip=1, test_file=None, aud_file=None, exp_file='face.pt',
                            no_com=False, all_speaker=False, use_ori=False, use_ba=False, test_offset=0):

    if test_file:
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        exps = []
        exp_features = torch.load(os.path.join(basedir, exp_file))['exp_o'].numpy()[test_offset:]
        aud_features = torch.load(os.path.join(basedir, aud_file)).cpu().numpy()
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            auds.append(
                aud_features[min(frame['img_id'], aud_features.shape[0]-1)]
                )
            exps.append(
                exp_features[min(frame['img_id'], exp_features.shape[0]-1)]
                )

        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        exps = np.array(exps).astype(np.float32)
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(
            meta['cx']), float(meta['cy'])

        data = {
            'poses':poses,
            'auds':auds,
            'bc_img':bc_img,
            'hwfcxy':[H, W, focal, cx, cy],
            'exp': exps,
        }   

        return data

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        if use_ba:
            with open(os.path.join(basedir, 'transforms_{}_ba.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
        else:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
    all_imgs = []
    all_imgs_com = []
    all_imgs_ori = []
    all_poses = []
    all_auds = []
    all_exps = []
    all_sample_rects = []
    exp_features = torch.load(os.path.join(basedir, exp_file))['exp_o'].numpy() # [3:]
    aud_features = torch.load(os.path.join(basedir, aud_file)).cpu().numpy()

    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        imgs_torso = []
        imgs_com= []
        imgs_ori = []
        poses = []
        auds = []
        sample_rects = []
        exps = []

        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, 'head_imgs',
                                '{:06d}.jpg'.format(frame['img_id']))
            fname_ori = os.path.join(basedir, 'ori_imgs',
                                '{:06d}.jpg'.format(frame['img_id']))
            fname_com = os.path.join(basedir, 'com_imgs',
                                '{:06d}.jpg'.format(frame['img_id']))
            imgs.append(fname)

            if not no_com:
                imgs_com.append(fname_com)
            if use_ori:
                imgs_ori.append(fname_ori)
            
            pose = np.array(frame['transform_matrix'])
            poses.append(pose)
            
            auds.append(
                aud_features[min(frame['aud_id'], aud_features.shape[0]-1)]
                )
            exps.append(
                exp_features[min(frame['img_id'], exp_features.shape[0]-1)]
            )
            sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))

        imgs = np.array(imgs)
        imgs_torso = np.array(imgs_torso)
        imgs_ori = np.array(imgs_ori)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        exps = np.array(exps).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_exps.append(exps)

        if not no_com:
            all_imgs_com.append(imgs_com)
        if use_ori:
            all_imgs_ori.append(imgs_ori)
        all_poses.append(poses)
        all_auds.append(auds)
        all_sample_rects.append(sample_rects)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)                  # [B,]

    if not no_com:
        imgs_com = np.concatenate(all_imgs_com, 0)
    if use_ori:
        imgs_ori = np.concatenate(all_imgs_ori, 0)
    poses = np.concatenate(all_poses, 0)                # [B, 4, 4]
    auds = np.concatenate(all_auds, 0)                  # [B, 16, 29]
    exps = np.concatenate(all_exps, 0)
    sample_rects = np.concatenate(all_sample_rects, 0)  # [B, 4]  x, y, w, h

    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    speak_frames = np.zeros(auds.shape[0], dtype=np.int32)

    if all_speaker:
        speak_frames = speak_frames + 1
    else:
        speak_time = np.load(os.path.join(basedir, 'speak_time.npy'))
        fps = 30
        for i in range(speak_time.shape[0]):
            last_time = np.arange(int(speak_time[i,0] * fps) + 1, int(speak_time[i,1] * fps) - 1)
            speak_frames[last_time] = 1

    if no_com:
        imgs_com = None
    if not use_ori:
        imgs_ori = None
        
    data = {
        'imgs':imgs,
        "imgs_com":imgs_com,
        'poses':poses,
        'auds':auds,
        'bc_img':bc_img,
        'hwfcxy':[H, W, focal, cx, cy],
        'sample_rects':sample_rects,
        'i_split':i_split,
        'speak_frames':speak_frames,
        'exp': exps,
        'imgs_ori': imgs_ori
    }

    return data