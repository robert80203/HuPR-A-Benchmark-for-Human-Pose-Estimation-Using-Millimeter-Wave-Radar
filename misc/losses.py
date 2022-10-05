import torch
import numpy as np
import torch.nn as nn
from math import sqrt, pi, log
import torch.nn.functional as F
from misc import get_max_preds, generateTarget


class LossComputer():
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.numFrames = self.cfg.DATASET.numFrames
        self.numGroupFrames = self.cfg.DATASET.numGroupFrames
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.lossDecay = self.cfg.TRAINING.lossDecay
        self.alpha = 0.0
        self.beta = 1.0
        self.bce = nn.BCELoss()
    
    def computeLoss(self, preds, gt):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.numKeypoints, self.height, self.width))
        gtKpts = torch.zeros((b, self.numKeypoints, 2))
        for i in range(len(gt)):
            heatmap, gtKpt = generateTarget(gt[i], self.numKeypoints, self.heatmapSize, self.imgSize)
            heatmaps[i, :] = torch.tensor(heatmap)
            gtKpts[i] = torch.tensor(gtKpt)
        preds, preds2 = preds      
        loss1 = self.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
        loss2 = self.computeBCESingleFrame(preds2.view(-1, self.numKeypoints, self.height, self.width), heatmaps)
        preds2 = preds2.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
        if self.alpha < 1.0:
            self.alpha += self.lossDecay
            self.beta -= self.lossDecay
        if self.lossDecay != -1:
            loss = self.alpha * loss1 + self.beta * loss2
        else:
            loss = loss1 + loss2
        pred2d, _ = get_max_preds(preds2.detach().cpu().numpy())
        gt2d, _ = get_max_preds(heatmaps.detach().cpu().numpy())
        return loss, loss2, pred2d, gt2d

    def computeBCESingleFrame(self, preds, gt):
        loss = self.bce(preds, gt.to(self.device))
        return loss