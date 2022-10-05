import numpy as np
import matplotlib.pyplot as plt




def PlotMaps(name, x_indices, y_indices, idx, matrix1, matrix2, matrix3=None, matrix4=None):

    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(matrix1, extent=[-60, 60, 0, 100])
    plt.xticks(x_indices)
    plt.yticks(y_indices)

    fig.add_subplot(rows, columns, 2)
    plt.imshow(matrix2, extent=[-60, 60, 0, 100])
    plt.xticks(x_indices)
    plt.yticks(y_indices)

    if matrix3 is not None:
        fig.add_subplot(rows, columns, 3)
        plt.imshow(matrix3)
    
    if matrix4 is not None:
        fig.add_subplot(rows, columns, 4)
        plt.imshow(matrix4)

    fig.savefig(name)
    if (idx % 500) == 0:
        print("clean")
        plt.close('all')


def PlotHeatmaps(joints, numKeypoints):
    #heatmaps = GenerateHeatmapsFromKeypoints([256, 256], joints, numKeypoints)
    heatmaps = generate_target(joints, numKeypoints)
    return heatmaps


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

def generate_target(joints, numKeypoints):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        #target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        #target_weight[:, 0] = joints_vis[:, 0]
        sigma = 2 #cfg.MODEL.SIGMA
        heatmapSize = np.array([64, 64])
        imgSize = np.array([256, 256]) #np.array([512, 512])#


        
        target = np.zeros((numKeypoints,
                           heatmapSize[1],
                           heatmapSize[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(numKeypoints):
            feat_stride = imgSize / heatmapSize
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

        #if self.use_different_joints_weight:
        #    target_weight = np.multiply(target_weight, self.joints_weight)

        #return target, target_weight
        target = np.max(target, axis = 0)
        return target
