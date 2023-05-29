import numpy as np
import cv2

keypoints: {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"}

skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
skeleton_woFace = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
                   [7,9],[8,10],[9,11],[5,7]]
skeleton_array = np.array(skeleton)-1
skeleton_woFace_array = np.array(skeleton)-6

face_kpts_98_to_68 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,  # face
                      33, 34, 35, 36, 37,  42, 43, 44, 45, 46,  # brow
                      51, 52, 53, 54,  55, 56, 57, 58, 59,  # nose
                      60, 61, 63, 64, 65, 67,  # right eye
                      68, 69, 71, 72, 73, 75,  # left eye
                      76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,  # lip outside
                      88, 89, 90, 91, 92, 93, 94, 95  # lip inside
                      ]


def draw_coco(input_img, preds, scores, thresh=0.3, color=(0, 0, 255), draw_lines=True, draw_face=True):
    """Short summary.

    Parameters
    ----------
    input_img : type numpy array
        cv2 image
    preds : type numpy array
        keypoints of shape: (num_keypoints, 2)
    scores : type numpy array
        scores of shape: (num_keypoints)
    thresh : type float
        the threshold of keypoints scores to visualize
    color : type str or tuple
        color of the keypoints and lines
    draw_lines : type bool
        whether to draw lines or not
    draw_face: type bool
        whether to draw 5 face keypoints or not

    Returns
    -------
    the image with keypoints

    """
    vis = scores > thresh
    preds = preds.astype(np.int32)
    input_img = input_img.copy()
    num_kps = preds.shape[0]
    if draw_face:
        skeleton_to_draw = skeleton_array
    else:
        skeleton_to_draw = skeleton_woFace_array
    for i in range(num_kps):
        # positive = preds[i][0] > 0 and preds[i][1] > 0
        if vis[i]:
            cv2.circle(input_img, (preds[i][0], preds[i][1]), 2, color, 3)

    if draw_lines:
        for kp in skeleton_to_draw:
            if vis[kp[0]] and vis[kp[1]]:
                cv2.line(input_img, (preds[kp[0]][0], preds[kp[0]][1]), (preds[kp[1]][0], preds[kp[1]][1]), color, 2)

    return input_img

def draw_circle(input_img, preds, scores, thresh=0.5, r=2, color=(0, 0, 255)):
    """draw circles in image

    Args:
        input_img ([numpy.array]): input image
        preds ([numpy.array]): pred coords of shape (Num keypoints, x coordinate, y coordinate)
        scores ([numpy.array]): pred scores of shape (Num keypoints, )
        thresh (float, optional): visualize score Defaults to 0.5.
        r (int, optional): draw radius. Defaults to 2.
        color (tuple, optional): [description]. Defaults to (0, 0, 255).

    Returns:
        [type]: [description]
    """
    vis = scores > thresh
    preds = preds.astype(np.int32)
    num_kps = preds.shape[0]
    input_img = input_img.copy()
    for i in range(num_kps):
        if vis[i]:
            cv2.circle(input_img, (preds[i][0], preds[i][1]), r, color, -1)

    return input_img