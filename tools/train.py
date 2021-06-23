from tqdm import tqdm
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os

from datasets import HoriVertiDataset
from models import BaseModel
from misc.plot import plotHumanPose, plotHumanPoseRGBWithGT
from misc.losses import LossComputer
from misc.utils import generateHeatmapsFromKeypoints, generate_target
from misc.metrics import accuracy
from misc.logger import Logger

DEBUG = False

class Trainer():
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() and args.gpuIDs else 'cpu'
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #self.imgSize = 256
        self.heatmapSize = 64
        self.width = self.height = self.heatmapSize#self.imgSize
        
        self.numKeypoints = 17
        self.dimsWidthHeight = (self.width, self.height)
        self.start_epoch = 0
        # mode: 'multiFrames', 'multiChirps'
        # 'multiFrames': multiple input (frames), multiple output (frames)
        # 'multiChirps': multiple input (chirps), single output (frame)
        self.mode = args.dataset 
        if self.mode == 'multiFrames':
            self.numFrames = 30
        elif self.mode == 'multiChirps':
            self.numFrames = 16
        trainSet = HoriVertiDataset(args.dataDir, 'train', self.dimsWidthHeight, 
                                    self.dimsWidthHeight, self.numFrames,
                                    self.mode)
        testSet = HoriVertiDataset(args.dataDir, 'test', self.dimsWidthHeight, 
                                   self.dimsWidthHeight, self.numFrames,
                                   self.mode)
        self.trainLoader = data.DataLoader(trainSet,
                              args.batchSize,
                              shuffle=True,
                              num_workers=args.numWorkers)
        self.testLoader = data.DataLoader(testSet,
                              args.batchSize,
                              shuffle=False,
                              num_workers=args.numWorkers)
        print('Train set size:', len(self.trainLoader))
        print('Test set size:', len(self.testLoader))
        self.args = args
        self.model = BaseModel(self.numFrames, self.numKeypoints, self.dimsWidthHeight, self.mode).to(self.device)
        
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.lossComputer = LossComputer(self.numFrames, self.numKeypoints, self.dimsWidthHeight, self.device)
        self.logger = Logger()
        self.initialize()
        
    def initialize(self):
        if not os.path.isdir(self.args.saveDir):
            os.mkdir(self.args.saveDir)
        if not os.path.isdir(self.args.visDir):
            os.mkdir(self.args.visDir)
    
    def saveModelWeight(self, epoch):
        saveGroup = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.logger.showAcc(mode='best'),
        }
        if self.logger.isBestAcc():
            print('Save the best model...')
            torch.save(saveGroup, os.path.join(self.args.saveDir, 'model_best.pth'))
        if (epoch + 1) % 5 == 0:
            print('Save the model...')
            torch.save(saveGroup, os.path.join(self.args.saveDir, 'model_{}.pth'.format(epoch + 1)))
    
    def loadModelWeight(self):
        checkpoint = torch.load(self.args.loadDir)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.logger.updateBestAcc(checkpoint['accuracy'])
    
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
                    if self.mode == 'multiFrames':
                        #loss, heatmapsGT = self.lossComputer.computeL2MutiFrames(preds, keypoints)
                        loss, heatmapsGT = self.lossComputer.computeBCEMutiFrames(preds, keypoints)
                        # permute the output to (batch, numFrames, numKeypoints, h, w)
                        heatmapsGT = heatmapsGT.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                    elif self.mode == 'multiChirps':
                        singleFrameKP = keypoints[:, 0, :, :]
                        #loss, heatmapsGT = self.lossComputer.computeL2SingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), singleFrameKP)
                        loss, heatmapsGT = self.lossComputer.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), singleFrameKP)
                        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                    acc, cnt, preds, gts = accuracy(preds.cpu().numpy(), 
                                                    heatmapsGT.cpu().numpy())
                    #print(keypoints[0][0], gts[0], gts.shape)
                    self.logger.update(acc, cnt)
                    progress_bar.set_postfix(Loss=loss.item(), ACC=self.logger.showAcc())
                    progress_bar.update(horiMapRangeAzimuth.size(0))
                    if visualization: # only for the first frame of the first batch
                        # visualize preds (heatmaps)
                        #plotHumanPose(preds[0][0].cpu().numpy(), self.args.visDir, self.mode, idx)
                        
                        # visualize processed preds (preds -> joints -> heatmaps)
                        #procPreds = generateHeatmapsFromKeypoints(self.dimsWidthHeight, gts[0], self.numKeypoints)
                        if self.mode == 'multiFrames':
                            numVisFrames = self.numFrames * horiMapRangeAzimuth.size(0)
                            for idxFrames in range(numVisFrames):
                                procPreds = generate_target(preds[idxFrames] * 4, self.numKeypoints)
                                procGTs = generate_target(gts[idxFrames] * 4, self.numKeypoints)
                                #plotHumanPose(procPreds, self.args.visDir, idx * numVisFrames + idxFrames)
                                plotHumanPoseRGBWithGT(procPreds, procGTs, self.args.visDir, idx * numVisFrames + idxFrames)
                        elif self.mode == 'multiChirps':
                            numVisFrames = horiMapRangeAzimuth.size(0)
                            for idxFrames in range(numVisFrames):
                                procPreds = generate_target(preds[idxFrames] * 4, self.numKeypoints)
                                procGTs = generate_target(gts[idxFrames] * 4, self.numKeypoints)
                                #plotHumanPose(procPreds, self.args.visDir, idx * numVisFrames + idxFrames)
                                plotHumanPoseRGBWithGT(procPreds, procGTs, self.args.visDir, idx * numVisFrames + idxFrames)
                if DEBUG:
                    break
    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            self.logger.clear()
            with tqdm(total=len(self.trainLoader.dataset)) as progress_bar:
                for batch in self.trainLoader:
                    self.optimizer.zero_grad()

                    horiMapRangeAzimuth = batch['horiImg'].float().to(self.device)
                    vertMapRangeAzimuth = batch['vertImg'].float().to(self.device)
                    keypoints = batch['jointsGroup']
                    preds = self.model(horiMapRangeAzimuth, vertMapRangeAzimuth)

                    #keypoints = generateKeypoints([self.img_size, self.img_size], self.num_frames)
                    #loss = L2Loss(output, [keypoints], [self.img_size, self.img_size], self.num_frames, self.device)
                    if self.mode == 'multiFrames':
                        #loss, heatmapsGT = self.lossComputer.computeL2MutiFrames(preds, keypoints)
                        loss, heatmapsGT = self.lossComputer.computeBCEMutiFrames(preds, keypoints)
                        # permute the output to (batch, numFrames, numKeypoints, h, w)
                        heatmapsGT = heatmapsGT.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                    elif self.mode == 'multiChirps':
                        singleFrameKP = keypoints[:, 0, :, :]
                        #loss, heatmapsGT = self.lossComputer.computeL2SingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), singleFrameKP)
                        loss, heatmapsGT = self.lossComputer.computeBCESingleFrame(preds.view(-1, self.numKeypoints, self.height, self.width), singleFrameKP)
                        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.numKeypoints, self.height, self.width)
                    loss.backward()
                    self.optimizer.step()                    
                    # permute the output to (batch, numFrames, numKeypoints, h, w)
                    acc, cnt, _, _ = accuracy(preds.detach().cpu().numpy(), 
                                                    heatmapsGT.detach().cpu().numpy())
                    self.logger.update(acc, cnt)
                    progress_bar.set_postfix(Loss=loss.item(), ACC=self.logger.showAcc())
                    progress_bar.update(horiMapRangeAzimuth.size(0))
                    if DEBUG:
                        break
            self.eval(visualization=False)
            self.saveModelWeight(epoch)