import torch
import torch.nn as nn
from misc.utils import generateHeatmapsFromKeypoints
from misc.utils import generate_target

import matplotlib.pyplot as plt
import numpy as np

class LossComputer():
    def __init__(self, numFrames, numKeypoints, dimsWidthHeight, device):
        self.numFrames = numFrames
        self.numKeypoints = numKeypoints
        self.device = device
        self.dimsWidthHeight = dimsWidthHeight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
    
    def computeL2MutiFrames(self, preds, gt):
        heatmaps = torch.zeros_like(preds)
        #gt: (batch, num_frames, num_joints, 2)
        for i in range(len(gt)):
            for j in range(self.numFrames):
                # self-designed generation of heatmap
                #heatmap = generateHeatmapsFromKeypoints(self.dimsWidthHeight, gt[i][j], self.numKeypoints)
                # adopted from other implementation
                heatmap = generate_target(gt[i][j], self.numKeypoints)
                
                # temp = np.max(heatmap, axis = 0)
                # fig = plt.figure(figsize=(10, 7))
                # rows = 2
                # columns = 2
                # fig.add_subplot(rows, columns, 1)
                # plt.imshow(temp)
                # fig.savefig('./test.png')
                
                # heatmap: (numKeypoints, h, w)
                heatmaps[i, :, j] = torch.tensor(heatmap)
        #print(preds.size(), heatmaps.size())
        loss = self.mse(preds, heatmaps.to(self.device)) * 10
        return loss, heatmaps
    
    def computeBCEMutiFrames(self, preds, gt):
        heatmaps = torch.zeros_like(preds)
        #gt: (batch, num_frames, num_joints, 2)
        for i in range(len(gt)):
            for j in range(self.numFrames):               
                heatmap = generate_target(gt[i][j], self.numKeypoints)
                heatmaps[i, :, j] = torch.tensor(heatmap)
        #loss = self.bce(preds, torch.sigmoid(heatmaps.to(self.device))) * 10
        loss = self.bce(preds, heatmaps.to(self.device)) * 10
        return loss, heatmaps
    
    def computeL2SingleFrame(self, preds, gt):
        heatmaps = torch.zeros_like(preds)
        #gt: (batch, num_frames, num_joints, 2)
        for i in range(len(gt)):
            heatmap = generate_target(gt[i], self.numKeypoints)
            heatmaps[i, :] = torch.tensor(heatmap)
        #print(preds.size(), heatmaps.size())
        loss = self.mse(preds, heatmaps.to(self.device)) * 10
        return loss, heatmaps
    
    def computeBCESingleFrame(self, preds, gt):
        heatmaps = torch.zeros_like(preds)
        #gt: (batch, num_frames, num_joints, 2)
        for i in range(len(gt)):
            heatmap = generate_target(gt[i], self.numKeypoints)
            heatmaps[i, :] = torch.tensor(heatmap)
        #print(preds.size(), heatmaps.size())
        #loss = self.bce(preds, torch.sigmoid(heatmaps.to(self.device))) * 10
        loss = self.bce(preds, heatmaps.to(self.device)) * 10
        return loss, heatmaps