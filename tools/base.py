import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from misc.logger import Logger
import torch.utils.data as data
import torch.nn.functional as F
from misc.losses import LossComputer

class BaseRunner():
    def __init__(self, args, cfg):
        self.device = 'cuda' if torch.cuda.is_available() and args.gpuIDs else 'cpu'
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.dir = './logs/' + args.dir
        self.visDir = './visualization/' + args.visDir
        self.args = args
        self.cfg = cfg
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.dimsWidthHeight = (self.width, self.height)
        self.start_epoch = 0
        self.numFrames = self.cfg.DATASET.numFrames
        self.F = self.cfg.DATASET.numGroupFrames
        self.imgHeatmapRatio = self.cfg.DATASET.imgSize / self.cfg.DATASET.heatmapSize
        self.aspectRatio = self.imgWidth * 1.0 / self.imgHeight
        self.pixel_std = 200

    def initialize(self, LR):
        self.lossComputer = LossComputer(self.cfg, self.device)
        self.logger = Logger()
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        if not os.path.isdir(self.visDir):
            os.mkdir(self.visDir)
        if not self.args.eval:
            print('==========>Train set size:', len(self.trainLoader))
        print('==========>Test set size:', len(self.testLoader))
  
        if self.cfg.TRAINING.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)

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


    def saveModelWeight(self, epoch, acc):
        saveGroup = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.logger.showBestAP(),
        }
        if self.logger.isBestAccAP(acc):
            saveGroup['accuracy'] = self.logger.showBestAP()
            print('==========>Save the best model...')
            torch.save(saveGroup, os.path.join(self.dir, 'model_best.pth'))

        print('==========>Save the latest model...')
        torch.save(saveGroup, os.path.join(self.dir, 'checkpoint.pth'))
        if epoch % 5 == 0:
            torch.save(saveGroup, os.path.join(self.dir, 'checkpoint_%d.pth'%epoch))

    def saveCheckpoint(self, epoch):
        saveGroup = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.logger.showAcc(mode='best'),
        }
        print('==========>Save the latest model...')
        torch.save(saveGroup, os.path.join(self.dir, 'checkpoint_%d.pth' % epoch))
        
    def saveLosslist(self, epoch, loss_list, mode):
        with open(os.path.join(self.dir,'%s_loss_list_%d.json'%(mode, epoch)), 'w') as fp:
            json.dump(loss_list, fp)
    
    def loadModelWeight(self, mode):
        checkpoint = os.path.join(self.dir, '%s.pth'%mode)
        if os.path.isdir(self.dir) and os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not self.args.eval:
                if not self.args.pretrained: #self.args.pretrained_encoder:
                    print('==========>Load the previous optimizer')
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch']
                    self.logger.updateBestAcc(checkpoint['accuracy'])
                else:
                    print('==========>Load a new optimizer')
                
            print('==========>Load the model weight from %s, saved at epoch %d' %(self.dir, checkpoint['epoch']))
        else:
            print('==========>Train the model from scratch')
    
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
