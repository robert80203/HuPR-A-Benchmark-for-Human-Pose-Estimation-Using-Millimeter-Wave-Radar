from tqdm import tqdm
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import json


from misc.losses import LossComputer
from misc.logger import Logger


class BaseTrainer():
    def __init__(self, args, cfg):
        self.device = 'cuda' if torch.cuda.is_available() and args.gpuIDs else 'cpu'
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        self.dir = './logs/' + args.dir
        self.visDir = './visualization/' + args.visDir
        #self.loadDir = './logs/' + args.loadDir
        self.args = args
        self.cfg = cfg
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.dimsWidthHeight = (self.width, self.height)
        self.start_epoch = 0
        #self.global_step = 0
        self.DEBUG = args.debug
        #self.mode = self.cfg.DATASET.mode
        self.numFrames = self.cfg.DATASET.numFrames
        self.modelType = self.cfg.MODEL.type
        self.fronModel = self.cfg.MODEL.frontModel
        self.metricType = self.cfg.TRAINING.metric
        self.imgHeatmapRatio = self.cfg.DATASET.imgSize / self.cfg.DATASET.heatmapSize
        self.aspectRatio = self.imgWidth * 1.0 / self.imgHeight
        self.pixel_std = 200

    def initialize(self):
        self.lossComputer = LossComputer(self.cfg, self.device)
        self.logger = Logger(False if 'l2norm' in self.cfg.TRAINING.metric else True)
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        if not os.path.isdir(self.visDir):
            os.mkdir(self.visDir)
        if not self.args.eval:
            print('==========>Train set size:', len(self.trainLoader))
        print('==========>Test set size:', len(self.testLoader))
    
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspectRatio * h:
            h = w * 1.0 / self.aspectRatio
        elif w < self.aspectRatio * h:
            w = h * self.aspectRatio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def adjustLR(self, epoch):
        if epoch < self.cfg.TRAINING.warmupEpoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.cfg.TRAINING.warmupGrowth
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.cfg.TRAINING.lrDecay


    def saveModelWeight(self, epoch):
        saveGroup = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.logger.showAcc(mode='best'),
        }
        if self.logger.isBestAcc():
            print('Save the best model...')
            torch.save(saveGroup, os.path.join(self.dir, 'model_best.pth'))

        print('Save the latest model...')
        torch.save(saveGroup, os.path.join(self.dir, 'checkpoint.pth'))
    
    def loadModelWeight(self, mode):
        checkpoint = os.path.join(self.dir, '%s.pth'%mode)
        if os.path.isdir(self.dir) and os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.logger.updateBestAcc(checkpoint['accuracy'])
            print('Load the model weight from %s, saved at epoch %d' %(self.dir, checkpoint['epoch']))
        else:
            print('No loading is performed')
    
    def saveKeypoints(self, savePreds, preds, bbox, image_id, predHeatmap=None):
        
        visidx = np.ones((len(preds), self.numKeypoints, 1))
        preds = np.concatenate((preds, visidx), axis=2)
        predsigma = np.zeros((len(preds), self.numKeypoints))
        
        for j in range(len(preds)):
            block = {}
            #center, scale = self._xywh2cs(bbox[j][0], bbox[j][1], bbox[j][2] - bbox[j][0], bbox[j][3] - bbox[j][1])
            center, scale = self._xywh2cs(bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3])
            block["category_id"] = 1
            block["center"] = center.tolist()
            block["image_id"] = image_id[j].item()
            block["scale"] = scale.tolist()
            block["score"] = 1.0
            block["keypoints"] = preds[j].reshape(self.numKeypoints*3).tolist()
            if predHeatmap is not None:
                for kpts in range(self.numKeypoints):
                    predsigma[j, kpts] = predHeatmap[j, kpts].var().item() * self.heatmapSize
                block["sigma"] = predsigma[j].tolist()
            block_copy = block.copy()
            savePreds.append(block_copy)

        return savePreds

    def writeKeypoints(self, preds):
        predFile = os.path.join(self.dir, "test_results.json" if self.args.eval else "val_results.json")
        with open(predFile, 'w') as fp:
            json.dump(preds, fp)

    def eval(self, visualization=True, isTest=False):#should set batch size = 1
        pass
    
    def train(self):
        pass
