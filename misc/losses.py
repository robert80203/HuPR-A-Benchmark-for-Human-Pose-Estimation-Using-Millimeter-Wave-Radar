import torch
import torch.nn as nn
from misc.utils import generateHeatmapsFromKeypoints
from misc.utils import generateTarget

import matplotlib.pyplot as plt
import numpy as np

class LossComputer():
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.numFrames = self.cfg.DATASET.numFrames
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.modelType = self.cfg.MODEL.type
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def computeLoss(self, preds, gt):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.numKeypoints, self.height, self.width))
        gtKpts = torch.zeros((b, self.numKeypoints, 2))
        for i in range(len(gt)):
            #print('before')
            heatmap, gtKpt = generateTarget(gt[i], self.numKeypoints, self.heatmapSize, self.imgSize)
            heatmaps[i, :] = torch.tensor(heatmap)
            gtKpts[i] = torch.tensor(gtKpt)
            #print('gt', gtKpt, heatmap.max())
        if self.modelType == "heatmap":
            loss = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
            preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            return loss, None, (preds, None), (heatmaps, None)
        elif self.modelType == "regression":
            logits, kypts = preds
            loss = self.computeL2SingleFrame(kypts, gtKpts)
            lossCls = self.computeCE(logits)
            loss += lossCls
            #preds_ = kypts.long()
            preds = torch.zeros((b, self.numKeypoints, self.height, self.width))
            for i in range(len(kypts)):
                #print('before')
                heatmap, out = generateTarget(kypts[i], self.numKeypoints, self.heatmapSize, self.imgSize, isCoord=True)
                preds[i, :] = torch.tensor(heatmap)
                #print('out', out, heatmap.max())
            #print(preds.size())
            #preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
            return loss, lossCls, (preds, kypts * self.heatmapSize), (heatmaps, gtKpts)
        else:
            raise RuntimeError(F"model type should be regression/heatmap, not {self.modelType}.")
        
    def computeL2SingleFrame(self, preds, gt):
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