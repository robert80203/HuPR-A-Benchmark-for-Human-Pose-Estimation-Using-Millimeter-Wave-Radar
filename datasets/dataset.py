import os
import random
from random import sample
from datasets.base import BaseDataset
from PIL import Image
import numpy as np
import torch

class HoriVertiDataset(BaseDataset):
    def __init__(self, dataDir, phase, resizeShape, cropShape, numFrames, mode):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HoriVertiDataset, self).__init__(dataDir, phase, resizeShape, cropShape)
        duration = 1800 # 30 FPS * 60 seconds
        if phase == 'test':
            dirGroup = ['single_6']
        else:
            dirGroup = ['single_1', 'single_2', 'single_3', 'single_4']
        if mode == 'multiFrames':
            numFramesEachClip = duration
        elif mode == 'multiChirps':
            numFramesEachClip = duration * numFrames
        idxFrameGroup = [('%09d' % i) for i in range(numFramesEachClip)]
        #self.hor_dir = os.path.join(data_dir, phase, 'hori')
        #self.ver_dir = os.path.join(data_dir, phase, 'verti')
        #self.hor_paths = self.get_paths(self.hor_dir, dir_list, frame_list)
        #self.ver_paths = self.get_paths(self.ver_dir, dir_list, frame_list)
        self.horiPaths = self.getPaths(dataDir, dirGroup, 'hori', idxFrameGroup)
        self.vertPaths = self.getPaths(dataDir, dirGroup, 'verti', idxFrameGroup)
        self.annots = self.getAnnots(dataDir, dirGroup, 'annot', 'hrnet_annot.json')
        self.transformFunc = self.getTransformFunc()
        self.numFrames = numFrames
        self.mode = mode
        self.h = 64
        self.w = 8
    def __getitem__(self, index):
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
            if self.mode == 'multiChirps':
                joints = torch.LongTensor(self.annots[i//self.numFrames]['joints'])
            elif self.mode == 'multiFrames':
                joints = torch.LongTensor(self.annots[i]['joints'])
            
            #print(hor_path, self.annots[i]['image'])
            #print(joints)
            if i == index * self.numFrames:
                horiImgs = horiImg.unsqueeze(0)
                vertImgs = vertImg.unsqueeze(0)
                jointsGroup = joints.unsqueeze(0)
            else:
                horiImgs = torch.cat((horiImgs, horiImg.unsqueeze(0)), 0)
                vertImgs = torch.cat((vertImgs, vertImg.unsqueeze(0)), 0)
                jointsGroup = torch.cat((jointsGroup, joints.unsqueeze(0)), 0)

        return {'horiImg': horiImgs,
                'horiPath': horiPath,
                'vertImg': vertImgs,
                'vertPath': vertPath,
                'jointsGroup': jointsGroup}

    def __len__(self):
        return len(self.horiPaths)//self.numFrames