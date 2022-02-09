import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import Transformer,  PositionEmbeddingLearned, Joiner

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Temporalmaxpool(nn.Module):
    def __init__(self, numFrames):
        super(Temporalmaxpool, self).__init__()
        #self.temporalMaxpool = nn.MaxPool3d((numFrames, 1, 1), (numFrames, 1, 1))
    def forward(self, chirpMaps):
        return chirpMaps #self.temporalMaxpool(chirpMaps)

class MNet(nn.Module):
    def __init__(self, numFrames, numFilters, useVelocity=False):
        super(MNet, self).__init__()
        sizeTemp = sizeTempStride = numFrames//2
        self.temporalConvWx1x1 = nn.Conv3d(3 if useVelocity else 2, numFilters, (2, 1, 1), (2, 1, 1), (0, 0, 0))
        self.temporalMaxpool = nn.MaxPool3d((sizeTemp, 1, 1), (sizeTempStride, 1, 1))
    def forward(self, chirpMaps):
        # chirpMaps becomes : (batch, 2, # of chirps, h, w)
        chirpMaps = self.temporalConvWx1x1(chirpMaps)
        maps = self.temporalMaxpool(chirpMaps)
        return maps

class ChirpNet(nn.Module):
    def __init__(self, cfg):
        super(ChirpNet, self).__init__()
        featureSize = 64
        patchSize = 4
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames

        self.backbone = nn.Sequential(Identity())
        projInputDim = self.height * self.width * 2
        self.posEmbed = PositionEmbeddingLearned(self.numFrames, 1, num_pos_feats=featureSize//2)
        self.inputProj = nn.Linear(projInputDim, featureSize, 1)
        self.queryEmbed = nn.Embedding(self.numKeypoints, featureSize)
        self.horiChirpTrans = Transformer(d_model=featureSize, encoderOnly=True)
        self.vertChirpTrans = Transformer(d_model=featureSize, encoderOnly=True)
        self.extractor = Joiner(self.backbone, self.posEmbed, useResNet=False)

    def forward(self, horiMap, vertMap):
        batchSize = horiMap.size(0)
        for frameIdx in range(self.numGroupFrames):
            hMap = horiMap[:, frameIdx, :, :, :, :].permute(0, 2, 1, 3, 4)
            vMap = vertMap[:, frameIdx, :, :, :, :].permute(0, 2, 1, 3, 4)
            hMap, hPos = self.extractor(hMap.view(batchSize, self.numFrames, 1, -1))
            vMap, vPos = self.extractor(hMap.view(batchSize, self.numFrames, 1, -1))
            hMap = self.horiChirpTrans(self.inputProj(hMap), None, self.queryEmbed.weight, hPos)
            vMap = self.vertChirpTrans(self.inputProj(vMap), None, self.queryEmbed.weight, vPos)
            if frameIdx == 0:
                horiMaps = hMap.unsqueeze(2)
                vertMaps = vMap.unsqueeze(2)
            else:
                horiMaps = torch.cat((horiMaps, hMap.unsqueeze(2)), 2)
                vertMaps = torch.cat((vertMaps, vMap.unsqueeze(2)), 2)

