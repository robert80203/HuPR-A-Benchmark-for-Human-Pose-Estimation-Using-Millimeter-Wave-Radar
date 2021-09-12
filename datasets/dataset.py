import os
import random
from random import sample
from datasets.base import BaseDataset
from PIL import Image
import numpy as np
import torch

class HoriVertiDataset(BaseDataset):
    def __init__(self, phase, cfg):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HoriVertiDataset, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        #self.numSamples = cfg.DATASET.numSamples
        self.numChirps = cfg.DATASET.numChirps
        self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize

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
        self.transformFunc = self.getTransformFunc()

    def __getitem__(self, index):
        if self.mode == 'multiChirps':
            #i = index
            #for i in range(index * self.numFrames, (index + 1) * self.numFrames):
            #horiPath = self.horiPaths[i]
            #vertPath = self.vertPaths[i]
            
            #horiImg = np.zeros((self.h, self.w, 2))
            #vertImg = np.zeros((self.h, self.w, 2))

            #horiRealImag = np.loadtxt(horiPath)
            #vertRealImag = np.loadtxt(vertPath)
            
            #horiImg[:,:,0] = horiRealImag[:,:self.w]
            #horiImg[:,:,1] = horiRealImag[:,self.w:]
            #vertImg[:,:,0] = vertRealImag[:,:self.w]
            #vertImg[:,:,1] = vertRealImag[:,self.w:]
            
            #horiImg = self.transformFunc(horiImg)
            #vertImg = self.transformFunc(vertImg)
            
            #if i == index * self.numFrames:
            #    horiImgs = horiImg.unsqueeze(0)
            #    vertImgs = vertImg.unsqueeze(0)
            #else:
            #    horiImgs = torch.cat((horiImgs, horiImg.unsqueeze(0)), 0)
            #    vertImgs = torch.cat((vertImgs, vertImg.unsqueeze(0)), 0)
            
            horiPath = self.horiPaths[index]
            vertPath = self.vertPaths[index]
            
            horiRealImag = np.load(horiPath)
            vertRealImag = np.load(vertPath)
            
            horiImgs = torch.zeros((self.numFrames, 2, self.h, self.w))
            vertImgs = torch.zeros((self.numFrames, 2, self.h, self.w))
            #horiImgs = torch.zeros((self.numSamples, 2, self.h, self.w))
            #vertImgs = torch.zeros((self.numSamples, 2, self.h, self.w))
            
            idxSampleChirps = 0
            for idxChirps in range(0, self.numChirps, self.numChirps//self.numFrames):
                horiImgs[idxSampleChirps, 0, :, :] = self.transformFunc(horiRealImag[idxChirps].real)
                horiImgs[idxSampleChirps, 1, :, :] = self.transformFunc(horiRealImag[idxChirps].imag)
                vertImgs[idxSampleChirps, 0, :, :] = self.transformFunc(vertRealImag[idxChirps].real)
                vertImgs[idxSampleChirps, 1, :, :] = self.transformFunc(vertRealImag[idxChirps].imag)
                idxSampleChirps += 1

            joints = torch.LongTensor(self.annots[index]['joints'])

            return {'horiImg': horiImgs,
                    'horiPath': horiPath,
                    'vertImg': vertImgs,
                    'vertPath': vertPath,
                    'jointsGroup': joints}
        
        elif self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            # collect past frames and furture frames for the center target frame
            padSize = index % self.duration
            idx = index - self.numGroupFrames//2 - 1
            
            horiImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
            vertImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
            
            #horiImgss = torch.zeros((self.numGroupFrames, self.numSamples, 2, self.h, self.w))
            #vertImgss = torch.zeros((self.numGroupFrames, self.numSamples, 2, self.h, self.w))

            for j in range(self.numGroupFrames):
                if (j + padSize) < self.numGroupFrames//2:
                    idx = index - padSize
                elif j >= (self.duration - 1 - padSize) + self.numGroupFrames//2:
                    idx = index + (self.duration - 1 - padSize)
                else:
                    idx += 1

                #for i in range(idx * self.numFrames, (idx + 1) * self.numFrames):
                #    horiPath = self.horiPaths[i]
                #    vertPath = self.vertPaths[i]
                    
                #    horiImg = np.zeros((self.h, self.w, 2))
                #    vertImg = np.zeros((self.h, self.w, 2))

                #    horiRealImag = np.loadtxt(horiPath)##################not sure
                #    vertRealImag = np.loadtxt(vertPath)

                #    horiImg[:,:,0] = horiRealImag[:,:self.w]
                #    horiImg[:,:,1] = horiRealImag[:,self.w:]
                #    vertImg[:,:,0] = vertRealImag[:,:self.w]
                #    vertImg[:,:,1] = vertRealImag[:,self.w:]
                    
                #    horiImg = self.transformFunc(horiImg)
                #    vertImg = self.transformFunc(vertImg)

                #    joints = torch.LongTensor(self.annots[i//self.numFrames]['joints'])

                #    if i == idx * self.numFrames:
                #        horiImgs = horiImg.unsqueeze(0)
                #        vertImgs = vertImg.unsqueeze(0)
                #        #jointsGroup = joints.unsqueeze(0)
                #    else:
                #        horiImgs = torch.cat((horiImgs, horiImg.unsqueeze(0)), 0)
                #        vertImgs = torch.cat((vertImgs, vertImg.unsqueeze(0)), 0)
                #        #jointsGroup = torch.cat((jointsGroup, joints.unsqueeze(0)), 0)
                 
                #if j == 0:
                #    horiImgss = horiImgs.unsqueeze(0)
                #    vertImgss = vertImgs.unsqueeze(0)
                #    #jointsGroups = jointsGroup.unsqueeze(0)
                #else:
                #    horiImgss = torch.cat((horiImgss, horiImgs.unsqueeze(0)), 0)
                #    vertImgss = torch.cat((vertImgss, vertImgs.unsqueeze(0)), 0)
                #    #jointsGroups = torch.cat((jointsGroups, jointsGroup.unsqueeze(0)), 0)
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

            joints = torch.LongTensor(self.annots[index]['joints'])    
            return {'horiImg': horiImgss,
                    'horiPath': horiPath,
                    'vertImg': vertImgss,
                    'vertPath': vertPath,
                    'jointsGroup': joints}
    def __len__(self):
        return len(self.horiPaths)
        #return len(self.horiPaths)//self.numFrames
        #if self.mode == 'multiChirps':
        #    return len(self.horiPaths)//self.numFrames
        #elif self.mode == 'multiFramesChirps':
        #    return len(self.horiPaths)//(self.numFrames * self.numGroupFrames)