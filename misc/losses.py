import torch
import torch.nn as nn
from misc.utils import generateHeatmapsFromKeypoints
from misc.utils import generateTarget
from misc.metrics import get_max_preds

import matplotlib.pyplot as plt
import numpy as np

class LossComputer():
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.numFrames = self.cfg.DATASET.numFrames
        self.numGroupFrames = self.cfg.DATASET.numGroupFrames
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.modelType = self.cfg.MODEL.type
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
    
    # only for heatmap
    def computeMultiLoss(self, preds, gt):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.numGroupFrames, self.numKeypoints, self.height, self.width))
        gtKpts = torch.zeros((b, self.numGroupFrames, self.numKeypoints, 2))
        for i in range(len(gt)):
            for j in range(self.numGroupFrames):
                heatmap, gtKpt = generateTarget(gt[i][j], self.numKeypoints, self.heatmapSize, self.imgSize)
                heatmaps[i, j, :] = torch.tensor(heatmap)
                gtKpts[i, j] = torch.tensor(gtKpt)
            #print('gt', gtKpt, heatmap.max())
        loss = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps.view(-1, self.numKeypoints, self.height, self.width))
        preds = preds.reshape(-1, self.numKeypoints, self.height, self.width)
        heatmaps = heatmaps.reshape(-1, self.numKeypoints, self.height, self.width)
        return loss, None, (preds, None), (heatmaps, None)

    def computeLoss(self, preds, gt):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.numKeypoints, self.height, self.width))
        gtKpts = torch.zeros((b, self.numKeypoints, 2))
        for i in range(len(gt)):
            heatmap, gtKpt = generateTarget(gt[i], self.numKeypoints, self.heatmapSize, self.imgSize)
            heatmaps[i, :] = torch.tensor(heatmap)
            gtKpts[i] = torch.tensor(gtKpt)
        if self.modelType == "heatmap":
            loss = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
            preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            return loss, None, (preds, None), (heatmaps, None)
        elif self.modelType == "heatmap_heatmap":
            preds, preds2 = preds      
            loss = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
            preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            loss2 = self.computeBCESingleFrame(preds2.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
            preds2 = preds2.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            loss += loss2
            return loss, loss2, (preds, preds2), (heatmaps, gtKpts)
        elif self.modelType == "heatmap_regress": # for fronModel of temporal5
            preds, predJoints = preds      
            loss = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
            preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            #loss2 = self.computeMSESingleFrame(predJoints, gtKpts) * 0.01
            #loss2 = self.computeL1Normalized(predJoints, gtKpts)
            loss2 = self.computeMSENormalized(predJoints, gtKpts)
            loss += loss2
            return loss, loss2, (preds, predJoints * self.heatmapSize), (heatmaps, gtKpts)
        elif self.modelType == "heatmap_refine":
            preds, predJoints = preds      
            loss = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
            preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            predKpts = torch.zeros((b, self.numGroupFrames, self.numKeypoints, 2))
            predKpts, _ = get_max_preds(preds.view(-1, self.numKeypoints, self.height, self.width).detach().cpu().numpy())
            predKpts = torch.FloatTensor(predKpts).to(self.device)
            loss2 = self.l1(predJoints, (gtKpts.float().to(self.device)-predKpts)/self.heatmapSize)
            loss += loss2
            return loss, loss2, (preds, predJoints * self.heatmapSize), (heatmaps, gtKpts)
        else:
            raise RuntimeError("Wrong model type, not {self.modelType}.")
    
    def computeL1(self, preds, gt):
        #gt range: 0 ~ heatmapsize
        loss = self.l1(preds, gt.float().to(self.device))
        return loss

    def computeL1Normalized(self, preds, gt):
        #gt range: 0 ~ heatmapsize
        gt = gt / self.heatmapSize
        loss = self.l1(preds, gt.float().to(self.device))
        return loss

    def computeMSENormalized(self, preds, gt):
        #print(gt, gt.size())
        gt = gt / self.heatmapSize
        loss = self.mse(preds, gt.float().to(self.device))
        return loss

    def computeMSESingleFrame(self, preds, gt):
        #gt: (batch, num_joints, 2), LongTensor
        #gt = gt / self.heatmapSize
        preds = preds * self.heatmapSize
        loss = self.mse(preds, gt.float().to(self.device))
        return loss
    
    def computeBCESingleFrame(self, preds, gt):
        loss = self.bce(preds, gt.to(self.device))
        return loss

    def computeCE(self, logits):
        b = logits.size(0)
        gt = torch.arange(self.numKeypoints)
        gt = gt.unsqueeze(0).repeat(b, 1).view(-1).to(self.device)
        loss = self.ce(logits.view(b * self.numKeypoints, -1), gt)
        return loss


# elif self.modelType == "justregress":
#     kypts = preds
#     #loss = self.computeL1(kypts, gtKpts)
#     loss = self.computeL1Normalized(kypts, gtKpts)
#     preds = torch.zeros((b, self.numKeypoints, self.height, self.width))
#     for i in range(len(kypts)):
#         #print('before')
#         heatmap, out = generateTarget(kypts[i], self.numKeypoints, self.heatmapSize, self.imgSize, isCoord=True)
#         preds[i, :] = torch.tensor(heatmap)
#     return loss, None, (preds, kypts * self.heatmapSize), (heatmaps, gtKpts) 
#     #return loss, None, (preds, kypts), (heatmaps, gtKpts) 