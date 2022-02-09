import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.layers import Identity, BasicBlock2D, BasicBlock3D, Block2D, DeconvBlock2D

class SpatialModel2(nn.Module):
    def __init__(self, cfg):
        super(SpatialModel2, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.rangeSize = cfg.DATASET.rangeSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.mode = cfg.DATASET.mode
        self.numFilters = cfg.MODEL.numFilters #32 is the old version value
        self.frontModel = cfg.MODEL.frontModel
        
        self.main = nn.Sequential(
            BasicBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
            nn.Upsample(size=(self.height//2, self.width//2), mode='bilinear', align_corners=True),
            #nn.Upsample(size=(self.height//2, self.width//2), mode='bicubic', align_corners=True),
            BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters*2, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
            nn.Upsample(size=(self.height//4, self.width//4), mode='bilinear', align_corners=True),
            #nn.Upsample(size=(self.height//4, self.width//4), mode='bicubic', align_corners=True),
            BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters*4, self.numFilters*8, (3, 3), (1, 1), (1, 1)),
        )
    def forward(self, x):
        return self.main(x)


class SpatialModel4(nn.Module):
    def __init__(self, cfg):
        super(SpatialModel4, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.rangeSize = cfg.DATASET.rangeSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.mode = cfg.DATASET.mode
        self.numFilters = cfg.MODEL.numFilters #32 is the old version value
        self.frontModel = cfg.MODEL.frontModel

        self.main = nn.Sequential(
            Block2D(self.numFilters, self.numFilters*2, (5, 5), (2, 2), (2, 2)),
            BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
            Block2D(self.numFilters*2, self.numFilters*4, (5, 5), (2, 2), (2, 2)),
            BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
            Block2D(self.numFilters*4, self.numFilters*8, (5, 5), (2, 2), (2, 2)),
            BasicBlock2D(self.numFilters*8, self.numFilters*8, (3, 3), (1, 1), (1, 1)),
            Block2D(self.numFilters*8, self.numFilters*16, (5, 5), (2, 2), (2, 2)),
        )
    def forward(self, x):
        return self.main(x)