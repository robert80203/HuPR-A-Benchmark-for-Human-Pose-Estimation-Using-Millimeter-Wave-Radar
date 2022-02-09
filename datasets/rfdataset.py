import os
import random
from random import sample
from datasets.base import BaseDataset
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def getRFPoseDataset(phase, cfg, sampling_ratio):
    return RFPoseRDA(phase, cfg, sampling_ratio) if cfg.DATASET.useVelocity else RFPoseRA(phase, cfg, sampling_ratio)

class RFPoseRA(BaseDataset):
    def __init__(self, phase, cfg, sampling_ratio=1):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(RFPoseRA, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        #self.numSamples = cfg.DATASET.numSamples
        self.numChirps = cfg.DATASET.numChirps
        self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.chirpModel = cfg.MODEL.chirpModel
        self.sampling_ratio = sampling_ratio
        
        if phase == 'test':
             dirGroup = cfg.DATASET.testName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        elif phase == 'val':
             dirGroup = cfg.DATASET.valName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        else:
             dirGroup = cfg.DATASET.trainName
             dataDirGroup = cfg.DATASET.dataDir
        
        #numFramesEachClip = self.duration * self.numFrames
        numFramesEachClip = self.duration
        idxFrameGroup = [('%09d' % i) for i in range(numFramesEachClip)]
        self.horiPaths = self.getPaths(dataDirGroup, dirGroup, 'hori', idxFrameGroup)
        self.vertPaths = self.getPaths(dataDirGroup, dirGroup, 'verti', idxFrameGroup)
        self.annots = self.getAnnots(dataDirGroup, dirGroup, 'annot', 'hrnet_annot.json')
        self.transformFunc = self.getTransformFunc(cfg)

    def __getitem__(self, index):
        #index = index * self.sampling_ratio
        index = index * random.randint(1, self.sampling_ratio)
        if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            # collect past frames and furture frames for the center target frame
            padSize = index % self.duration
            idx = index - self.numGroupFrames//2 - 1
            
            horiImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
            vertImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
            joints = torch.zeros((self.numGroupFrames, self.numKeypoints, 2))

            for j in range(self.numGroupFrames):
                if (j + padSize) <= self.numGroupFrames//2:
                    idx = index - padSize
                elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                    idx = index + (self.duration - 1 - padSize)
                else:
                    idx += 1

                horiPath = self.horiPaths[idx]
                vertPath = self.vertPaths[idx]
                
                horiRealImag = np.load(horiPath)
                vertRealImag = np.load(vertPath)

                idxSampleChirps = 0
                #for idxChirps in range(self.numFrames):
                for idxChirps in range(0, self.numChirps, self.numChirps//self.numFrames):
                    horiImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(horiRealImag[idxChirps].real)
                    horiImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(horiRealImag[idxChirps].imag)
                    vertImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(vertRealImag[idxChirps].real)
                    vertImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(vertRealImag[idxChirps].imag)
                    idxSampleChirps += 1

                joints[j] = torch.LongTensor(self.annots[idx]['joints'])

            if self.chirpModel == 'maxpool':
                #print(horiImgss.size())
                horiImgss = F.max_pool3d(horiImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)
                vertImgss = F.max_pool3d(vertImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)

            

            return {'horiImg': horiImgss, # shape: (# of frames, # of chirps, # of channel=2, h, w)
                    'horiPath': horiPath,
                    'vertImg': vertImgss,  # shape: (# of frames, # of chirps, # of channel=2, h, w)
                    'vertPath': vertPath,
                    'jointsGroup': joints}
    def __len__(self):
        return len(self.horiPaths)//self.sampling_ratio

class RFPoseRDA(BaseDataset):
    def __init__(self, phase, cfg, sampling_ratio=1):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(RFPoseRDA, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        #self.numSamples = cfg.DATASET.numSamples
        self.numChirps = cfg.DATASET.numChirps
        self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.chirpModel = cfg.MODEL.chirpModel
        self.sampling_ratio = sampling_ratio
        
        if phase == 'test':
             dirGroup = cfg.DATASET.testName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        elif phase == 'val':
             dirGroup = cfg.DATASET.valName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        else:
             dirGroup = cfg.DATASET.trainName
             dataDirGroup = cfg.DATASET.dataDir
        
        #numFramesEachClip = self.duration * self.numFrames
        numFramesEachClip = self.duration
        idxFrameGroup = [('%09d' % i) for i in range(numFramesEachClip)]
        self.horiPaths = self.getPaths(dataDirGroup, dirGroup, 'hori', idxFrameGroup)
        self.vertPaths = self.getPaths(dataDirGroup, dirGroup, 'verti', idxFrameGroup)
        self.horiVPaths = self.getPaths(dataDirGroup, dirGroup, 'horiv', idxFrameGroup)
        self.vertVPaths = self.getPaths(dataDirGroup, dirGroup, 'vertv', idxFrameGroup)
        self.annots = self.getAnnots(dataDirGroup, dirGroup, 'annot', 'hrnet_annot.json')
        self.transformFunc = self.getTransformFunc(cfg)

    def __getitem__(self, index):
        #index = index * self.sampling_ratio
        index = index * random.randint(1, self.sampling_ratio)
        if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            # collect past frames and furture frames for the center target frame
            padSize = index % self.duration
            idx = index - self.numGroupFrames//2 - 1
            
            horiImgss = torch.zeros((self.numGroupFrames, self.numFrames, 3, self.h, self.w))
            vertImgss = torch.zeros((self.numGroupFrames, self.numFrames, 3, self.h, self.w))
            joints = torch.zeros((self.numGroupFrames, self.numKeypoints, 2))

            for j in range(self.numGroupFrames):
                if (j + padSize) <= self.numGroupFrames//2:
                    idx = index - padSize
                elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                    idx = index + (self.duration - 1 - padSize)
                else:
                    idx += 1

                horiPath = self.horiPaths[idx]
                vertPath = self.vertPaths[idx]
                horiVPath = self.horiVPaths[idx]
                vertVpath = self.vertVPaths[idx]
                
                horiRealImag = np.load(horiPath)
                vertRealImag = np.load(vertPath)
                horiV = np.load(horiVPath)
                vertV = np.load(vertVpath)

                idxSampleChirps = 0
                for idxChirps in range(0, self.numChirps, self.numChirps//self.numFrames):
                    horiImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(horiRealImag[idxChirps].real)
                    horiImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(horiRealImag[idxChirps].imag)
                    horiImgss[j, idxSampleChirps, 2, :, :] = self.transformFunc(horiV[idxChirps])
                    vertImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(vertRealImag[idxChirps].real)
                    vertImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(vertRealImag[idxChirps].imag)
                    vertImgss[j, idxSampleChirps, 2, :, :] = self.transformFunc(vertV[idxChirps])

                    idxSampleChirps += 1
                
                joints[j] = torch.LongTensor(self.annots[idx]['joints'])
            
            if self.chirpModel == 'maxpool':
                #print(horiImgss.size())
                horiImgss = F.max_pool3d(horiImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)
                vertImgss = F.max_pool3d(vertImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)

            return {'horiImg': horiImgss, # shape: (# of frames, # of chirps, # of channel=3, h, w)
                    'vertImg': vertImgss,  # shape: (# of frames, # of chirps, # of channel=3, h, w)
                    'jointsGroup': joints}
    def __len__(self):
        return len(self.horiPaths)//self.sampling_ratio