# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
    
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):#thr=0.5
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(preds, GTs, modelType='heatmap', metricType='PCK', ratio=1.0, bbox=None):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    output, kypts = preds
    target, gtKypts = GTs
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    idx = list(range(output.shape[1])) # = num of keypoints
    norm = 1.0
    thr = 0.5
    if modelType == 'heatmap_regress':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        #pred = (pred + kypts.detach().cpu().ceil().numpy()) / 2.0
        pred = kypts.detach().cpu().ceil().numpy()
        #target = (target + gtKypts.detach().cpu().ceil().numpy()) / 2.0
    elif modelType == "heatmap_heatmap":
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        pred2, _ = get_max_preds(kypts.detach().cpu().numpy())
        #pred = (pred + pred2) / 2.0
        pred = pred2
    elif modelType == "heatmap_refine":
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        pred = pred + kypts.detach().cpu().numpy()
    elif 'heatmap' in modelType:
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
    elif 'regress' in modelType:
        pred = kypts.detach().cpu().ceil().numpy()
        target = gtKypts.detach().cpu().ceil().numpy()
    

    if metricType == 'OKS': # the results may not be correct
        cnt = np.array([len(pred)])
        acc = np.zeros((1, len(idx)))
        avg_acc = np.zeros((1))
        oks = compute_oks(pred, target, bbox)
        oks = np.sum(oks, axis=0)
        acc[0] = oks
        avg_acc[0] = np.sum(oks, axis=0)  / len(idx)
        return avg_acc, acc, cnt, pred, target
    
    elif metricType == 'l2norm_PCK':
        cnt = len(pred)
        # pixel_error = np.zeros((pred.shape[0], len(idx) + 1))
        # pixel_error[:,1:] += np.linalg.norm((pred - target) * ratio, axis=2)
        # avg_pixel_error = np.sum(np.mean(pixel_error[:,1:], axis=1), axis=0)

        pixel_error = np.zeros((pred.shape[0], len(idx)))
        pixel_error += np.linalg.norm((pred - target) * ratio, axis=2)
        avg_pixel_error = np.sum(np.mean(pixel_error, axis=1), axis=0)

        pixel_error = np.sum(pixel_error, axis=0)

        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = calc_dists(pred, target, norm)

        #acc = np.zeros((len(idx) + 1))
        acc = np.zeros((len(idx)))
        avg_acc = 0

        for b in range(pred.shape[0]):
            for i in range(len(idx)):
                accTmp = dist_acc(dists[idx[i]][b])
                if accTmp >= 0:
                    #acc[i + 1] += accTmp
                    acc[i] += accTmp

        #avg_acc = np.mean(acc[1:], axis=0)
        avg_acc = np.mean(acc, axis=0)
        return [avg_pixel_error, avg_acc], [pixel_error, acc], [cnt, cnt], pred, target
    elif metricType == 'APCK':
        PCK_threshold = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        cnt = np.array([len(pred) for i in range(len(PCK_threshold))])
        #acc = np.zeros((len(PCK_threshold), len(idx) + 1))
        acc = np.zeros((len(PCK_threshold), len(idx)))
        avg_acc = np.zeros((len(PCK_threshold)))
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = calc_dists(pred, target, norm)
        for j in range(len(PCK_threshold)):
            for b in range(pred.shape[0]):
                for i in range(len(idx)):
                    accTmp = dist_acc(dists[idx[i]][b], PCK_threshold[j])
                    if accTmp >= 0:
                        #acc[j][i + 1] += accTmp
                        acc[j][i] += accTmp
            # avg_acc[j] = np.mean(acc[j, 1:], axis=0)
            avg_acc[j] = np.mean(acc[j, :], axis=0)
        return avg_acc, acc, cnt, pred, target




def compute_oks(src_keypoints, dst_keypoints, bbox=None):
    sigmas = np.array([.107, .087, .089, .107, .087, .089, .029, .029, .079, .072, .062, .079, .072, .062])
    vars = (sigmas * 2) ** 2

    if bbox is not None:
        #src_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / 4 # other papers use division of 2
        src_area = (bbox[:, 2] * bbox[:, 3]) / 4
        src_area = src_area.view(src_area.size(0), 1).numpy()
    else:
        src_roi = [1, 1, 8, 8]
        src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, :, 0] - src_keypoints[:, :, 0]
    dy = dst_keypoints[:, :, 1] - src_keypoints[:, :, 1]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2

    #e = np.sum(np.exp(-e)) / e.shape[0]
    #e = np.exp(-e) / e.shape[1]
    #print(e.shape)
    e = np.exp(-e)
    return e

