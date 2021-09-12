import numpy as np
from random import sample
from scipy import signal


def generateGaussianKernel(kernlen, std, dim):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    if dim == 2:
        gkern = np.outer(gkern1d, gkern1d)
    else:
        gkern = np.outer(gkern1d, gkern1d, gkern1d)
    return gkern

def generateKeypoints(dims, num_frames, num_keypoints=16, kernlen=9):
    frame_keypoints = []
    temp = []
    for num in dims:
        l = [i for i in range(kernlen//2, num - kernlen//2)]
        temp.append(sample(l, num_keypoints))
    for k in range(num_frames):
        keypoints = []
        for i in range(num_keypoints):
            keypoints.append([temp[j][i] for j in range(len(dims))])
        frame_keypoints.append(keypoints)
    return frame_keypoints

#len_kernel=31, std=11
def generateHeatmapsFromKeypoints(dims, keypoints, num_keypoints, len_kernel=11, std=5):
    assert len(dims) == 2 or len(dims) == 3#dim: (w, h) or (w, h, z)
    padding = len_kernel//2
    
    if len(dims) == 2:
        heatmaps = np.zeros((num_keypoints, dims[1]+padding*2, dims[0]+padding*2))# (keypoints, h, w)        
    else:
        heatmaps =  np.zeros((num_keypoints, dims[2]+padding*2, dims[1]+padding*2, dims[0]+padding*2))
    
    gkern = generateGaussianKernel(len_kernel, std, len(dims))
    idx = 0

    for keypoint in keypoints:
        x_center = int(keypoint[0]+padding)
        y_center = int(keypoint[1]+padding)
        heatmap = heatmaps[idx] 
        if len(dims) == 2:
            #heatmap[x_center-padding:x_center+padding+1,y_center-padding:y_center+padding+1] = gkern
            heatmap[y_center-padding:y_center+padding+1,x_center-padding:x_center+padding+1] = gkern
        else:
            z_center = keypoint[2]
            #heatmap[x_center-padding:x_center+padding+1,y_center-padding:y_center+padding+1,z_center-padding:z_center+padding+1] = gkern
            heatmap[z_center-padding:z_center+padding+1,y_center-padding:y_center+padding+1,x_center-padding:x_center+padding+1] = gkern
        heatmaps[idx] = heatmap
        idx += 1

    #heatmap = np.max(heatmaps, axis = 0)

    if len(dims) == 2:
        return heatmaps[:, padding:dims[1]+padding, padding:dims[0]+padding]
    else:
        return heatmaps[:, padding:dims[2]+padding, padding:dims[1]+padding, padding:dims[0]+padding]

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

def generateTarget(joints, numKeypoints, hSize, iSize, isCoord=False):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    #target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
    #target_weight[:, 0] = joints_vis[:, 0]
    sigma = 2#cfg.MODEL.SIGMA
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
        if isCoord:
            mu_x = int(joints[joint_id][0] * hSize)
            mu_y = int(joints[joint_id][1] * hSize)
        else:
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        #if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
        #        or br[0] < 0 or br[1] < 0:
        #    # If not, just return the image as is
        #    target_weight[joint_id] = 0
        #    continue
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

        #v = target_weight[joint_id]
        #if v > 0.5:
        #    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
        #        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        targetKpts[joint_id][0] = mu_x
        targetKpts[joint_id][1] = mu_y
    #if self.use_different_joints_weight:
    #    target_weight = np.multiply(target_weight, self.joints_weight)

    #return target, target_weight
    #target = np.max(target, axis = 0)
    return target, targetKpts