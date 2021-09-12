import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os

from datasets import HoriVertiDataset
from misc.plot import plotHumanPoseRGBWithGT
from misc.utils import generateHeatmapsFromKeypoints, generateTarget
from misc.metrics import accuracy
from tools.base import BaseTrainer
from models import BaseModel



class Trainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(Trainer, self).__init__(args, cfg)
        
        if not args.eval:
            trainSet = HoriVertiDataset('train', cfg)
            self.trainLoader = data.DataLoader(trainSet,
                                  self.cfg.TRAINING.batchSize,
                                  shuffle=True,
                                  num_workers=cfg.SETUP.numWorkers)
        else:
            self.trainLoader = [0] # an empty loader

            
        testSet = HoriVertiDataset('test' if args.eval else 'val', cfg)
        self.testLoader = data.DataLoader(testSet, 1,
                              shuffle=False,
                              num_workers=cfg.SETUP.numWorkers)
        
        self.model = BaseModel(self.cfg).to(self.device)
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch

        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)
        if self.cfg.TRAINING.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
        
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
        self.initialize()
    
    def eval(self, visualization=True):#should set batch size = 1
        self.model.eval()
        self.logger.clear(len(self.testLoader.dataset))
        #with tqdm(total=len(self.testLoader.dataset)) as progress_bar:
        for idx, batch in enumerate(self.testLoader):
            horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
            vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
            keypoints = batch['jointsGroup']
            with torch.no_grad():
                preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)
                #print(preds[0])
                #loss, loss2, preds, heatmapsGT = self.lossComputer.computeLoss(preds, keypoints)
                loss, loss2, predGroup, GTGroup = self.lossComputer.computeLoss(preds, keypoints)

                # acc, accTable, cnt, preds, gts = accuracy(preds.cpu().numpy(), 
                #                                 heatmapsGT.cpu().numpy(), self.modelType)
                
                acc, accTable, cnt, preds, gts = accuracy(predGroup, GTGroup, self.modelType)
                #print(keypoints[0][0], gts[0], gts.shape)
                self.logger.update(acc, accTable, cnt)
                self.logger.display(loss, loss2, horiMapRangeAzimuth.size(0))

                if visualization:
                    numVisFrames = horiMapRangeAzimuth.size(0)
                    for idxFrames in range(numVisFrames):
                        procPreds, predKpts = generateTarget(preds[idxFrames] * self.cfg.DATASET.upsamplingFactor, 
                                                            self.numKeypoints, self.heatmapSize, self.imgSize)
                        procGTs, GTKpts = generateTarget(gts[idxFrames] * self.cfg.DATASET.upsamplingFactor, 
                                                    self.numKeypoints, self.heatmapSize, self.imgSize)
                        plotHumanPoseRGBWithGT(procPreds, procGTs, self.args.eval, self.cfg,
                                               self.visDir, idx * numVisFrames + idxFrames)
            if self.DEBUG:
                break
        self.logger.showAccTable(self.numKeypoints, self.cfg.LOGGER.idxToJoints)

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            self.logger.clear(len(self.trainLoader.dataset))
            #with tqdm(total=len(self.trainLoader.dataset)) as progress_bar:
            for idxBatch, batch in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
                vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
                keypoints = batch['jointsGroup']
                preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)
                loss, loss2, predGroup, GTGroup = self.lossComputer.computeLoss(preds, keypoints)
                loss.backward()
                #print(loss.item())
                self.optimizer.step()                    
                # permute the output to (batch, numFrames, numKeypoints, h, w)
                #acc, accTable, cnt, _, _ = accuracy(preds.detach().cpu().numpy(), 
                #                                heatmapsGT.detach().cpu().numpy())
                acc, accTable, cnt, _, _ = accuracy(predGroup, GTGroup, self.modelType)
                self.logger.update(acc, accTable, cnt)
                self.logger.display(loss, loss2, horiMapRangeAzimuth.size(0))

                if self.DEBUG:
                    break
                
                if idxBatch % 10 == 0:
                    self.adjustLR(epoch)

            self.eval(visualization=False)
            self.saveModelWeight(epoch)