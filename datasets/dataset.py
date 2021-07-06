import os
import random
from random import sample
from datasets.base import BaseDataset
from PIL import Image
import numpy as np
import torch

class HoriVertiDataset(BaseDataset):
    def __init__(self, dataDir, phase, cfg):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HoriVertiDataset, self).__init__(dataDir, phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        #for 20210609
        #if phase == 'test' or phase == 'val':
        #    dirGroup = ['single_6']
        #else:
        #    dirGroup = ['single_1','single_2','single_3','single_4']

        #for 20210628
        if phase == 'test':
             dirGroup = ['single_1']
        elif phase == 'val':
             dirGroup = ['single_11', 'single_21']
        else:
             dirGroup = ['single_%d' % i for i in range(2, 11)] + \
                         ['single_%d' % i for i in range(12, 21)] + \
                         ['single_%d' % i for i in range(22, 31)]
        
        numFramesEachClip = self.duration * self.numFrames
        idxFrameGroup = [('%09d' % i) for i in range(numFramesEachClip)]
        self.horiPaths = self.getPaths(dataDir, dirGroup, 'hori', idxFrameGroup)
        self.vertPaths = self.getPaths(dataDir, dirGroup, 'verti', idxFrameGroup)
        self.annots = self.getAnnots(dataDir, dirGroup, 'annot', 'hrnet_annot.json')
        self.transformFunc = self.getTransformFunc()

    def __getitem__(self, index):
        if self.mode == 'multiChirps':
            for i in range(index * self.numFrames, (index + 1) * self.numFrames):
                horiPath = self.horiPaths[i]
                vertPath = self.vertPaths[i]
                
                horiImg = np.zeros((self.h, self.w, 2))
                vertImg = np.zeros((self.h, self.w, 2))

                horiRealImag = np.loadtxt(horiPath)##################not sure
                vertRealImag = np.loadtxt(vertPath)

                horiImg[:,:,0] = horiRealImag[:,:self.w]
                horiImg[:,:,1] = horiRealImag[:,self.w:]
                vertImg[:,:,0] = vertRealImag[:,:self.w]
                vertImg[:,:,1] = vertRealImag[:,self.w:]
                
                horiImg = self.transformFunc(horiImg)
                vertImg = self.transformFunc(vertImg)

                if i == index * self.numFrames:
                    horiImgs = horiImg.unsqueeze(0)
                    vertImgs = vertImg.unsqueeze(0)
                    #jointsGroup = joints.unsqueeze(0)
                else:
                    horiImgs = torch.cat((horiImgs, horiImg.unsqueeze(0)), 0)
                    vertImgs = torch.cat((vertImgs, vertImg.unsqueeze(0)), 0)
                    #jointsGroup = torch.cat((jointsGroup, joints.unsqueeze(0)), 0)

            joints = torch.LongTensor(self.annots[index]['joints'])

            return {'horiImg': horiImgs,
                    'horiPath': horiPath,
                    'vertImg': vertImgs,
                    'vertPath': vertPath,
                    'jointsGroup': joints}
        elif self.mode == 'multiFramesChirps':
            # collect past frames and furture frames for the center target frame
            padSize = index % self.duration
            idx = index - self.numGroupFrames//2 - 1
            for j in range(self.numGroupFrames):
                if (j + padSize) < self.numGroupFrames//2:
                    idx = index - padSize
                elif j >= (self.duration - 1 - padSize) + self.numGroupFrames//2:
                    idx = index + (self.duration - 1 - padSize)
                else:
                    idx += 1
                
                for i in range(idx * self.numFrames, (idx + 1) * self.numFrames):
                    horiPath = self.horiPaths[i]
                    vertPath = self.vertPaths[i]
                    
                    horiImg = np.zeros((self.h, self.w, 2))
                    vertImg = np.zeros((self.h, self.w, 2))

                    horiRealImag = np.loadtxt(horiPath)##################not sure
                    vertRealImag = np.loadtxt(vertPath)

                    horiImg[:,:,0] = horiRealImag[:,:self.w]
                    horiImg[:,:,1] = horiRealImag[:,self.w:]
                    vertImg[:,:,0] = vertRealImag[:,:self.w]
                    vertImg[:,:,1] = vertRealImag[:,self.w:]
                    
                    horiImg = self.transformFunc(horiImg)
                    vertImg = self.transformFunc(vertImg)

                    joints = torch.LongTensor(self.annots[i//self.numFrames]['joints'])

                    if i == idx * self.numFrames:
                        horiImgs = horiImg.unsqueeze(0)
                        vertImgs = vertImg.unsqueeze(0)
                        #jointsGroup = joints.unsqueeze(0)
                    else:
                        horiImgs = torch.cat((horiImgs, horiImg.unsqueeze(0)), 0)
                        vertImgs = torch.cat((vertImgs, vertImg.unsqueeze(0)), 0)
                        #jointsGroup = torch.cat((jointsGroup, joints.unsqueeze(0)), 0)
                 
                if j == 0:
                    horiImgss = horiImgs.unsqueeze(0)
                    vertImgss = vertImgs.unsqueeze(0)
                    #jointsGroups = jointsGroup.unsqueeze(0)
                else:
                    horiImgss = torch.cat((horiImgss, horiImgs.unsqueeze(0)), 0)
                    vertImgss = torch.cat((vertImgss, vertImgs.unsqueeze(0)), 0)
                    #jointsGroups = torch.cat((jointsGroups, jointsGroup.unsqueeze(0)), 0)
            
            joints = torch.LongTensor(self.annots[index]['joints'])    
            return {'horiImg': horiImgss,
                    'horiPath': horiPath,
                    'vertImg': vertImgss,
                    'vertPath': vertPath,
                    'jointsGroup': joints}
    def __len__(self):
        return len(self.horiPaths)//self.numFrames
        #if self.mode == 'multiChirps':
        #    return len(self.horiPaths)//self.numFrames
        #elif self.mode == 'multiFramesChirps':
        #    return len(self.horiPaths)//(self.numFrames * self.numGroupFrames)
