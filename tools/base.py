from tqdm import tqdm
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os


from misc.losses import LossComputer
from misc.logger import Logger


class BaseTrainer():
    def __init__(self, args, cfg):
        self.device = 'cuda' if torch.cuda.is_available() and args.gpuIDs else 'cpu'
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        self.saveDir = './logs/' + args.saveDir
        self.visDir = './visualization/' + args.visDir
        self.loadDir = './logs/' + args.loadDir
        self.args = args
        self.cfg = cfg
        
    def initialize(self):
        self.lossComputer = LossComputer(self.numFrames, self.numKeypoints, self.heatmapSize, self.imgSize, self.device)
        self.logger = Logger()
        if not os.path.isdir(self.saveDir):
            os.mkdir(self.saveDir)
        if not os.path.isdir(self.visDir):
            os.mkdir(self.visDir)
    
    def saveModelWeight(self, epoch):
        saveGroup = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.logger.showAcc(mode='best'),
        }
        if self.logger.isBestAcc():
            print('Save the best model...')
            torch.save(saveGroup, os.path.join(self.saveDir, 'model_best.pth'))
        if (epoch + 1) % 5 == 0:
            print('Save the model...')
            torch.save(saveGroup, os.path.join(self.saveDir, 'model_{}.pth'.format(epoch + 1)))
    
    def loadModelWeight(self):
        checkpoint = torch.load(self.loadDir)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.logger.updateBestAcc(checkpoint['accuracy'])
        print('Load the model weight from %s, saved at epoch %d' %(self.loadDir, checkpoint['epoch']))
    
    def eval(self, visualization=True, isTest=False):#should set batch size = 1
        pass
    
    def train(self):
        pass
