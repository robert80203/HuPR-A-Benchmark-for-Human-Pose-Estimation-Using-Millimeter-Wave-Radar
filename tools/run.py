import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os

from datasets import getDataset
from misc.plot import plotHumanPoseRGBWithGT, plotHumanPose
from misc.utils import generateHeatmapsFromKeypoints, generateTarget
from misc.metrics import accuracy
from tools.base import BaseTrainer
from models import BaseModel


class Trainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(Trainer, self).__init__(args, cfg)    
        if not args.eval:
            self.trainSet = getDataset('train', cfg, args.sampling_ratio)
            self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batchSize,
                                  shuffle=True,
                                  num_workers=cfg.SETUP.numWorkers)
        else:
            self.trainLoader = [0] # an empty loader

        self.testSet = getDataset('test' if args.eval else 'val', cfg, args.sampling_ratio)
        #self.testLoader = data.DataLoader(self.testSet, 1 if self.args.vis else self.cfg.TRAINING.batchSize,
        self.testLoader = data.DataLoader(self.testSet, 
                              self.cfg.TEST.batchSize,
                              shuffle=False,
                              num_workers=cfg.SETUP.numWorkers)
        self.model = BaseModel(self.cfg).to(self.device)
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)  
        if self.cfg.TRAINING.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4) 
        self.initialize()
    
    def eval(self, visualization=True, epoch=0):#should set batch size = 1
        self.model.eval()
        self.logger.clear(len(self.testLoader.dataset))
        savePreds = []
        for idx, batch in enumerate(self.testLoader):
            horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
            vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
            keypoints = batch['jointsGroup']
            bbox = batch['bbox']
            imageId = batch['imageId']
            with torch.no_grad():
                preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)
                loss, loss2, predGroup, GTGroup = self.lossComputer.computeLoss(preds, keypoints)
                acc, accTable, cnt, preds, gts = accuracy(predGroup, GTGroup, self.modelType, self.metricType, self.imgHeatmapRatio, bbox)
                self.logger.update(acc, accTable, cnt)
                self.logger.display(loss, loss2, horiMapRangeAzimuth.size(0), epoch)
                if visualization:
                    # plotHumanPose(preds*self.imgHeatmapRatio, self.cfg, 
                    #               self.visDir, imageId, bbox)
                    plotHumanPose(gts*self.imgHeatmapRatio, self.cfg, 
                                 self.visDir, imageId, bbox) 
                    # numVisFrames = horiMapRangeAzimuth.size(0)
                    # for idxFrames in range(numVisFrames):
                    #     procPreds, predKpts = generateTarget(preds[idxFrames] * self.cfg.DATASET.upsamplingFactor, 
                    #                                         self.numKeypoints, self.heatmapSize, self.imgSize)
                    #     procGTs, GTKpts = generateTarget(gts[idxFrames] * self.cfg.DATASET.upsamplingFactor, 
                    #                                 self.numKeypoints, self.heatmapSize, self.imgSize)
                    #     # plotHumanPoseRGBWithGT(procPreds, procGTs, self.args.eval, self.cfg,
                    #     #                        self.visDir, idx * numVisFrames + idxFrames)
                    #     plotHumanPose(predKpts*self.imgHeatmapRatio, self.cfg, self.visDir, idx * numVisFrames + idxFrames, bbox)
                    #     # GTKpts = GTKpts*self.imgHeatmapRatio
                    #     # GTKpts[:, 0] = 256 - GTKpts[:, 0]
                    #     # plotHumanPose(GTKpts, self.args.eval, self.cfg, self.visDir, 'gt', idx * numVisFrames + idxFrames)    
            if self.DEBUG:
                break
            self.saveKeypoints(savePreds, preds*self.imgHeatmapRatio, bbox, imageId, predGroup[1])
        self.writeKeypoints(savePreds)
        self.logger.showAccTable(self.numKeypoints, self.cfg.DATASET.idxToJoints, self.cfg.TRAINING.metric)
        self.testSet.evaluate(self.dir)

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            self.logger.clear(len(self.trainLoader.dataset))
            for idxBatch, batch in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
                vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
                keypoints = batch['jointsGroup']
                bbox = batch['bbox']
                preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)
                loss, loss2, predGroup, GTGroup = self.lossComputer.computeLoss(preds, keypoints)
                loss.backward()
                self.optimizer.step()                    
                acc, accTable, cnt, _, _ = accuracy(predGroup, GTGroup, self.modelType, self.metricType, self.imgHeatmapRatio, bbox)
                self.logger.update(acc, accTable, cnt)
                self.logger.display(loss, loss2, horiMapRangeAzimuth.size(0), epoch)
                if self.DEBUG:
                    break
                if idxBatch % 500 == 0:
                    self.adjustLR(epoch)
            self.saveModelWeight(epoch)
            self.eval(visualization=False, epoch=epoch)