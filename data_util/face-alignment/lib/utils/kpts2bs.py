import numpy as np
# import pandas as pd

shapeGroup = ["JawOpen", "EyeBlinkRight", "EyeBlinkLeft",
              "BrowOuterUpRight", "BrowOuterUpLeft", "MouthSmileRight", "MouthSmileLeft"]

KEYPOINT_PAIRS = [[51, 57], [37, 41], [38, 40], [43, 47], [44, 46],
                  [20, 39], [21, 39], [23, 42], [22, 42], [48, 8], [54, 8]]
DIR_CHANGE_SHAPE = [[0, 1], [1, 0], [1, 0], [1, 0], [1, 0],
                    [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
NUM_PAIRS = [1, 2, 2, 2, 2, 1, 1]


def calDistance(x, y):
    dist_abs = x - y
    return np.sqrt(dist_abs[0] * dist_abs[0] + dist_abs[1] * dist_abs[1])


# Init blendshape calibration params
# cali_df = pd.read_csv('./clibration_blendShape.txt', sep='\t', header=None)
# cali_df = cali_df.iloc[:, :3]
# cali_arr = cali_df.values


def landmark2blendShape(faceKeypoints, cali_arr):
    """Convert face landmarks to blendshapes

    Args:
        faceKeypoints (np.array, dtype=float64, shape=[68, 2]): input face landmarks
    Returns:
        faceBlendShapes (np.array, dtype=float64, shape=[7, 1]): output blendshapes, refer to
        ["JawOpen", "EyeBlinkRight", "EyeBlinkLeft",
            "BrowOuterUpRight", "BrowOuterUpLeft", "MouthSmileRight", "MouthSmileLeft"]
    """
    faceSize = (calDistance(faceKeypoints[27], faceKeypoints[8]) +
                calDistance(faceKeypoints[0], faceKeypoints[16])) / 2
    BlendShapes_distGroup = np.zeros((11, 1), dtype=np.float32)
    mixBlendShapes = np.zeros((7, 1), dtype=np.float32)
    for i in range(len(KEYPOINT_PAIRS)):
        currDistance = calDistance(
            faceKeypoints[KEYPOINT_PAIRS[i][0]], faceKeypoints[KEYPOINT_PAIRS[i][1]]) / faceSize
        BlendShapes_distGroup[i] = ((currDistance - cali_arr[i, 2]) * DIR_CHANGE_SHAPE[i][1] + (cali_arr[i, 2] - currDistance) * DIR_CHANGE_SHAPE[i][0]) \
            / ((cali_arr[i, 2] - cali_arr[i, 0]) * DIR_CHANGE_SHAPE[i][0] +
               (cali_arr[i, 1] - cali_arr[i, 2]) * DIR_CHANGE_SHAPE[i][1])

    idx = 0
    # A blendShape may correspond to several keypoint pairs
    for i, num_pair in enumerate(NUM_PAIRS):
        for j in range(num_pair):
            mixBlendShapes[i] = mixBlendShapes[i] + BlendShapes_distGroup[idx]
            idx = idx + 1
        mixBlendShapes[i] = mixBlendShapes[i] / num_pair

    # Normalize blendshapes to [0, 1]
    mixBlendShapes = np.where(mixBlendShapes > 0, mixBlendShapes, 0)
    mixBlendShapes = np.where(mixBlendShapes < 1, mixBlendShapes, 1)

    return mixBlendShapes


def initialize_kpts(faceKeypoints, cali_arr):
    faceHeight = calDistance(faceKeypoints[27], faceKeypoints[8])
    faceWidth = calDistance(faceKeypoints[0], faceKeypoints[16])
    faceSize = (faceHeight + faceWidth) / 2
    for i in range(len(KEYPOINT_PAIRS)):
        distance = calDistance(
            faceKeypoints[KEYPOINT_PAIRS[i][0]], faceKeypoints[KEYPOINT_PAIRS[i][1]]) / faceSize
        cali_arr[i, 0] = distance
        cali_arr[i, 1] = distance
        cali_arr[i, 2] = distance
    return cali_arr


def calibration(faceKeypoints, cali_arr):
    faceHeight = calDistance(faceKeypoints[27], faceKeypoints[8])
    faceWidth = calDistance(faceKeypoints[0], faceKeypoints[16])
    faceSize = (faceHeight + faceWidth) / 2

    for i in range(len(KEYPOINT_PAIRS)):
        distance = calDistance(
            faceKeypoints[KEYPOINT_PAIRS[i][0]], faceKeypoints[KEYPOINT_PAIRS[i][1]]) / faceSize
        cali_arr[i, 0] = distance if cali_arr[i,
                                              0] > distance else cali_arr[i, 0]
        cali_arr[i, 1] = distance if cali_arr[i,
                                              1] < distance else cali_arr[i, 1]
    return cali_arr


