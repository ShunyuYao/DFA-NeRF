# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import math


MATCHED_PARTS = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
             [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
             [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]),
    "AFLW": ([1, 6],  [2, 5], [3, 4],
             [7, 12], [8, 11], [9, 10],
             [13, 15],
             [16, 18]),
    "COFW": ([1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]),
    "WFLW": ([0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
             [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
             [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
             [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
             [55, 59], [56, 58],
             [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
             [88, 92], [89, 91], [95, 93], [96, 97])}

# EYE_WFLW_MATCHED_PARTS = [[60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73], [96, 97]]
EYE_WFLW_MATCHED_PARTS = {
    "WFLW": [[0, 4], [1, 3], [5, 7]],
    "300W": [[0, 3], [1, 2], [4, 5]],
    "WFLW_EYE": [[0, 4], [1, 3], [5, 7],
                 [9, 13], [10, 12], [14, 17], [15, 16]]
}

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_eye_joints(x, width, dataset='WFLW'):
    """
    flip coords
    """
    # 9, 10, 11, 12, 13
    # 14, 15, 16, 17

    # 13, 12, 11, 10, 9,
    # 17, 16, 15, 14
    matched_parts = EYE_WFLW_MATCHED_PARTS[dataset]
    # Flip horizontal
    x[:, 0] = width - x[:, 0]
    for pair in matched_parts:
        tmp = x[pair[0], :].copy()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x

def fliplr_face_joints(x, width, dataset='aflw'):
    """
    flip coords
    """
    matched_parts = MATCHED_PARTS[dataset]
    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    if dataset == 'WFLW':
        for pair in matched_parts:
            tmp = x[pair[0], :].copy()
            x[pair[0], :] = x[pair[1], :]
            x[pair[1], :] = tmp
    else:
        for pair in matched_parts:
            tmp = x[pair[0] - 1, :].copy()
            x[pair[0] - 1, :] = x[pair[1] - 1, :]
            x[pair[1] - 1, :] = tmp
    return x


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform_bbox(bbox, output_size, inv=0):
    dst_w = output_size[0]
    dst_h = output_size[1]
    x, y, w, h = bbox

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = np.array([x, y], dtype=np.float32)
    src[1, :] = np.array([x+w, y+h], dtype=np.float32)
    dst[0, :] = np.array([0, 0], dtype=np.float32)
    dst[1, :] = np.array([dst_w, dst_h], dtype=np.float32)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


def crop_bbox(img, bbox, output_inv_trans=False):
    x, y, w, h = bbox
    w_scale, h_scale = 480 / w, 640 / h
    min_scale = min(w_scale, h_scale)
    if  min_scale > 1:
        output_size = [w * min_scale, h * min_scale]
    else:
        output_size = [w, h]
    trans = get_affine_transform_bbox(bbox, output_size, inv=0)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    if output_inv_trans:
        trans_inv = get_affine_transform_bbox(bbox, output_size, inv=1)
        return dst_img, trans, trans_inv
    return dst_img, trans


def xywh_to_LTRB(box):
    l = box[0]
    t = box[1]
    r = box[0] + box[2]
    b = box[1] + box[3]
    out = [l, t, r, b]
    return out


def LTRB_to_xywh(box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    out = [x, y, w, h]
    return out


def box2cs(box):
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)


def xywh2cs(x, y, w, h, aspect_ratio=1/1.3, pixel_std=200):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def pts2cs(pts, pixel_std=200.0):
    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])

    center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
    center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

    scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / pixel_std
    center = np.array([center_w, center_h], dtype=np.float32)

    # scale *= 1.25

    return center, scale


def pts2ltrb(pts):
    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])
    return xmin, xmax, ymin, ymax


def pts2wh(pts):
    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])

    w = xmax - xmin
    h = ymax - ymin
    return w, h


def pts2wh_center(pts):
    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])

    w = xmax - xmin
    h = ymax - ymin
    return w, h, (xmin + xmax) / 2, (ymin + ymax) / 2


def DARK_decode(hm, px, py, sigma):
    derivative_1 = np.array(
        [
            (2*np.log(hm[py][px+1])+np.log(hm[py+1][px+1])
             + np.log(hm[py-1][px+1]) -
             2*np.log(hm[py][px-1])-np.log(hm[py+1][px-1])
             - np.log(hm[py-1][px-1]))/4,
            (2*np.log(hm[py+1][px])+np.log(hm[py+1][px+1])
             + np.log(hm[py+1][px-1]) -
             2*np.log(hm[py-1][px])-np.log(hm[py-1][px+1])
             - np.log(hm[py-1][px-1]))/4
        ]
    )
    # derivative_2 laplace
    derivative_2 = np.array(
        [
            -1/(sigma**2),
            -1/(sigma**2)
        ]
    )
    # coords[n][p] -=

    return derivative_1/derivative_2
