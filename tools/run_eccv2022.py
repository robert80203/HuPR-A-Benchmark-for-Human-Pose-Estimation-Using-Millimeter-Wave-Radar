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
from models.networks_eccv2022 import BaseModel, RefineModel, RadarModel


class Runner(BaseTrainer):
    def __init__(self, args, cfg):
        super(Runner, self).__init__(args, cfg)    
        if not args.eval:
            self.trainSet = getDataset('train', cfg, args.sampling_ratio)
            self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batchSize,
                                  shuffle=True,
                                  num_workers=cfg.SETUP.numWorkers)
        else:
            self.trainLoader = [0] # an empty loader
        self.testSet = getDataset('test' if args.eval else 'val', cfg, args.sampling_ratio)
        self.testLoader = data.DataLoader(self.testSet, 
                              self.cfg.TEST.batchSize,
                              shuffle=False,
                              num_workers=cfg.SETUP.numWorkers)
        if "stage2" in cfg.MODEL.frontModel:
            self.model = RefineModel(self.cfg).to(self.device)
        elif cfg.MODEL.frontModel == 'radar':
            self.model = RadarModel(self.cfg).to(self.device)
        else:
            self.model = BaseModel(self.cfg).to(self.device)
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)  
        if self.cfg.TRAINING.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4) 
        self.initialize()
    
    def eval(self, visualization=True, epoch=0):
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
                    plotHumanPose(preds*self.imgHeatmapRatio, self.cfg, 
                                  self.visDir, imageId, bbox)   
            if self.DEBUG:
                break
            self.saveKeypoints(savePreds, preds*self.imgHeatmapRatio, bbox, imageId)
        self.writeKeypoints(savePreds)
        self.logger.showAccTable(self.numKeypoints, self.cfg.DATASET.idxToJoints, self.cfg.TRAINING.metric)
        self.testSet.evaluate(self.dir)

    def evalRefine(self, visualization=True, epoch=0):
        self.model.eval()
        self.logger.clear(len(self.testLoader.dataset))
        savePreds = []
        for idx, batch in enumerate(self.testLoader):
            jointsdata = batch['jointsdata'][:, :2, :, :].float().to(self.device)
            keypoints = batch['jointsGroup']
            bbox = batch['bbox']
            imageId = batch['imageId']
            with torch.no_grad():
                if self.cfg.MODEL.gcnType == 'coord':
                    preds = self.model(jointsdata)
                else:
                    sigma = batch['sigma']
                    preds = self.model(jointsdata, sigma.cuda())
                loss, loss2, predGroup, GTGroup = self.lossComputer.computeLoss(preds, keypoints)
                acc, accTable, cnt, preds, gts = accuracy(predGroup, GTGroup, self.modelType, self.metricType, self.imgHeatmapRatio, bbox)
                self.logger.update(acc, accTable, cnt)
                self.logger.display(loss, loss2, jointsdata.size(0), epoch)
                if visualization:
                    plotHumanPose(preds*self.imgHeatmapRatio, self.cfg, 
                                  self.visDir, imageId, bbox)
                    #plotHumanPose(gts*self.imgHeatmapRatio, self.cfg, 
                    #              self.visDir, imageId, bbox)      
            if self.DEBUG:
                break
            self.saveKeypoints(savePreds, preds*self.imgHeatmapRatio, bbox, imageId)
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
            self.eval(visualization=False, epoch=epoch)
            self.saveModelWeight(epoch)

    def trainRefine(self):
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            self.logger.clear(len(self.trainLoader.dataset))
            for idxBatch, batch in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                #(B, 3, T, V)
                jointsdata = batch['jointsdata'][:, :2, :, :].float().to(self.device)
                keypoints = batch['jointsGroup']
                bbox = batch['bbox']
                #print(batch['imageId'], batch['imageId2']) # data and annot are matched
                if self.cfg.MODEL.gcnType == 'coord':
                    preds = self.model(jointsdata)
                else:
                    sigma = batch['sigma']
                    preds = self.model(jointsdata, sigma.cuda())
                loss, loss2, predGroup, GTGroup = self.lossComputer.computeLoss(preds, keypoints)
                loss.backward()
                self.optimizer.step()
                acc, accTable, cnt, _, _ = accuracy(predGroup, GTGroup, self.modelType, self.metricType, self.imgHeatmapRatio, bbox)
                self.logger.update(acc, accTable, cnt)
                self.logger.display(loss, loss2, jointsdata.size(0), epoch)
                if self.DEBUG:
                    break
                if idxBatch % 500 == 0:
                    self.adjustLR(epoch)
            self.saveModelWeight(epoch)
            self.evalRefine(visualization=False, epoch=epoch)