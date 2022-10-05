import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def generateTarget(joints, numKeypoints, hSize, iSize, isCoord=False, sigmas=None):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    if hSize == 64:
        sigma = 2
    elif hSize == 128:
        sigma = 3
    heatmapSize = np.array([hSize, hSize])
    imgSize = np.array([iSize, iSize])

    target = np.zeros((numKeypoints,
                       heatmapSize[1],
                       heatmapSize[0]),
                      dtype=np.float32)

    targetKpts = np.zeros((numKeypoints, 2))

    tmp_size = sigma * 3

    for joint_id in range(numKeypoints):
        feat_stride = imgSize / heatmapSize
        if sigmas is not None:
            sigma = sigmas[joint_id] * 10
            tmp_size = sigma * 3
        if isCoord:
            mu_x = int(joints[joint_id][0] * hSize)
            mu_y = int(joints[joint_id][1] * hSize)
        else:
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmapSize[0] or ul[1] >= heatmapSize[1] or br[0] < 0 or br[1] < 0:
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmapSize[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmapSize[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmapSize[0])
        img_y = max(0, ul[1]), min(br[1], heatmapSize[1])

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        targetKpts[joint_id][0] = mu_x
        targetKpts[joint_id][1] = mu_y

    return target, targetKpts