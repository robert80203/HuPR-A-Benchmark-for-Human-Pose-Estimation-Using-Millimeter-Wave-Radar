from tqdm import tqdm
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os

from datasets import HoriVertiDataset
from misc.plot import plotHumanPoseRGBWithGT
from misc.utils import generateHeatmapsFromKeypoints, generate_target
from misc.metrics import accuracy
from tools.base import BaseTrainer
from models import BaseModel



class Trainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(Trainer, self).__init__(args, cfg)
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.dimsWidthHeight = (self.width, self.height)
        self.start_epoch = 0
        self.global_step = 0
        self.DEBUG = args.debug
        self.mode = self.cfg.DATASET.mode
        self.numFrames = self.cfg.DATASET.numFrames

        trainSet = HoriVertiDataset(cfg.DATASET.dataDir, 'train', cfg)
        testSet = HoriVertiDataset(cfg.DATASET.dataDir, 'test' if args.eval else 'val', cfg)
        #testSet = HoriVertiDataset(cfg.DATASET.dataDir, 'val', cfg)

        self.trainLoader = data.DataLoader(trainSet,
                              self.cfg.TRAINING.batchSize,
                              shuffle=True,
                              num_workers=cfg.SETUP.numWorkers)
        self.testLoader = data.DataLoader(testSet, 1,
                              shuffle=False,
                              num_workers=cfg.SETUP.numWorkers)
        
        self.model = BaseModel(self.cfg).to(self.device)
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch

        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (1.005 ** self.stepSize)

        if self.cfg.TRAINING.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
        
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
        self.initialize()
        
        print('Train set size:', len(self.trainLoader))
        print('Test set size:', len(self.testLoader))
    
    def adjustLR(self, epoch):
        if epoch < self.cfg.TRAINING.warmupEpoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.cfg.TRAINING.warmupGrowth
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.cfg.TRAINING.lrDecay

    def eval(self, visualization=True):#should set batch size = 1
        self.model.eval()
        self.logger.clear()
        with tqdm(total=len(self.testLoader.dataset)) as progress_bar:
            for idx, batch in enumerate(self.testLoader):
                horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
                vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
                keypoints = batch['jointsGroup']
                with torch.no_grad():
                    preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)
                    #singleFrameKP = keypoints[:, 0, :, :]
                    loss, heatmapsGT = self.lossComputer.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), keypoints)
                    
                    preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                    acc, accTable, cnt, preds, gts = accuracy(preds.cpu().numpy(), 
                                                    heatmapsGT.cpu().numpy())
                    #print(keypoints[0][0], gts[0], gts.shape)
                    self.logger.update(acc, accTable, cnt)
                    progress_bar.set_postfix(Loss=loss.item(), ACC=self.logger.showAcc())
                    progress_bar.update(horiMapRangeAzimuth.size(0))
                    if visualization:
                        numVisFrames = horiMapRangeAzimuth.size(0)
                        for idxFrames in range(numVisFrames):
                            procPreds = generate_target(preds[idxFrames] * self.cfg.DATASET.upsamplingFactor, 
                                                        self.numKeypoints, self.heatmapSize, self.imgSize)
                            procGTs = generate_target(gts[idxFrames] * self.cfg.DATASET.upsamplingFactor, 
                                                        self.numKeypoints, self.heatmapSize, self.imgSize)
                            plotHumanPoseRGBWithGT(procPreds, procGTs, self.args.eval, self.cfg,
                                                   self.visDir, idx * numVisFrames + idxFrames)
                if self.DEBUG:
                    break
            self.logger.showAccTable(self.numKeypoints, self.cfg.LOGGER.idxToJoints)

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            self.logger.clear()
            with tqdm(total=len(self.trainLoader.dataset)) as progress_bar:
                for idxBatch, batch in enumerate(self.trainLoader):
                    self.optimizer.zero_grad()
                    horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
                    vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
                    #print('hori input', horiMapRangeAzimuth.max(), horiMapRangeAzimuth.min())
                    #print('vert input', vertMapRangeAzimuth.max(), vertMapRangeAzimuth.min())
                    #print(batch['vertPath'])
                    keypoints = batch['jointsGroup']
                    preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)
                    #singleFrameKP = keypoints[:, 0, :, :]
                    loss, heatmapsGT = self.lossComputer.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), keypoints)
                    preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                    loss.backward()
                    #print(loss.item())
                    self.optimizer.step()                    
                    # permute the output to (batch, numFrames, numKeypoints, h, w)
                    acc, accTable, cnt, _, _ = accuracy(preds.detach().cpu().numpy(), 
                                                    heatmapsGT.detach().cpu().numpy())
                    self.logger.update(acc, accTable, cnt)
                    progress_bar.set_postfix(Loss=loss.item(), ACC=self.logger.showAcc())
                    progress_bar.update(horiMapRangeAzimuth.size(0))
                    self.global_step += horiMapRangeAzimuth.size(0)
                    if self.DEBUG:
                        break
                    
                    # if self.cfg.TRAINING.optimizer == 'sgd':
                    #     self.adjustLR(epoch)
                    if idxBatch % 10 == 0:
                        self.adjustLR(epoch)
            #self.scheduler.step()
            self.eval(visualization=False)
            self.saveModelWeight(epoch)
