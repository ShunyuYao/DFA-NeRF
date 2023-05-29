import numpy as np


def square_the_bbox(bbox):
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    return [top, left, bottom, right]


def scale_bbox(bbox, scales):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper
    scale_x = scales[0]
    scale_y = scales[1]

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale_x * width), int(scale_y * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def scale_bbox_with_limits(bbox, scales, limits):
    left, upper, right, lower = bbox
    left_limit, upper_limit, right_limit, lower_limit = limits
    width, height = right - left, lower - upper
    scale_x = scales[0]
    scale_y = scales[1]

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale_x * width), int(scale_y * height)

    new_left = x_center - new_width // 2
    if new_left < left_limit:
        new_left = left_limit

    new_right = new_left + new_width
    if new_right > right_limit:
        new_right = right_limit - 1

    new_upper = y_center - new_height // 2
    if new_upper < upper_limit:
        new_upper = upper_limit

    new_lower = new_upper + new_height
    if new_lower > lower_limit:
        new_lower = lower_limit - 1

    return new_left, new_upper, new_right, new_lower


def get_bbox_from_pose(keypoints_2d, square_bbox=True):  # expand_rate=0.3 image_shape
    x_max, y_max = np.max(keypoints_2d, 0)
    x_min, y_min = np.min(keypoints_2d, 0)

    if square_bbox is True:
        bbox = np.array(square_the_bbox([x_min, y_min, x_max, y_max]))
    else:
        bbox = np.array([x_min, y_min, x_max, y_max])

    return bbox

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
