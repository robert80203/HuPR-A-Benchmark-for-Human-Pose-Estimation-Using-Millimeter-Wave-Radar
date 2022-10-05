import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class MNet(nn.Module):
    def __init__(self, in_channels, out_channels, numFrames):
        super(MNet, self).__init__()
        sizeTemp = sizeTempStride = numFrames//2
        self.temporalConvWx1x1 = nn.Conv3d(in_channels, out_channels, (2, 1, 1), (2, 1, 1), (0, 0, 0))
        self.temporalMaxpool = nn.MaxPool3d((sizeTemp, 1, 1), (sizeTempStride, 1, 1))
    def forward(self, chirpMaps):
        # chirpMaps becomes : (batch, 2, # of chirps, h, w)
        chirpMaps = self.temporalConvWx1x1(chirpMaps)
        maps = self.temporalMaxpool(chirpMaps)
        return maps