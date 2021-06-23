import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

class BaseModel(nn.Module):
    def __init__(self, numFrames, numKeypoints, dimsWidthHeight, mode):
        super(BaseModel, self).__init__()
        self.numFrames = numFrames
        self.numKeypoints = numKeypoints
        self.width = dimsWidthHeight[0]
        self.height = dimsWidthHeight[1]
        self.numFilters = 32
        self.mode = mode
        #self.extractor = Identity()
        #for (75, 64, 8) input, padding = (4, 34, 6)
        if mode == 'multiFrames':
            self.encoderHoriMap = nn.Sequential(
                nn.Upsample(size=(self.numFrames, self.height, self.width//4), mode='trilinear'),
                ResidualBlock3D(2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames, self.height, self.width//2), mode='trilinear'),
                ResidualBlock3D(self.numFilters, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames, self.height, self.width), mode='trilinear'),
                ResidualBlock3D(self.numFilters*2, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            )
            self.encoderVertMap = nn.Sequential(
                nn.Upsample(size=(self.numFrames, self.height, self.width//4), mode='trilinear'),
                ResidualBlock3D(2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames, self.height, self.width//2), mode='trilinear'),
                ResidualBlock3D(self.numFilters, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames, self.height, self.width), mode='trilinear'),
                ResidualBlock3D(self.numFilters*2, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            )
            self.decoder = nn.Sequential(
                ResidualBlock3D(self.numFilters*4, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                ResidualBlock3D(self.numFilters*4, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                ResidualBlock3D(self.numFilters*2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Conv3d(self.numFilters, self.numKeypoints, 1, 1, 0),
                #nn.ReLU() #use relu in 06/22 version
                nn.Sigmoid()
            )
        elif mode == 'multiChirps':
            self.encoderHoriMap = nn.Sequential(
                nn.Upsample(size=(self.numFrames//2, self.height, self.width//4), mode='trilinear'),
                ResidualBlock3D(2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//2, self.height, self.width//2), mode='trilinear'),
                ResidualBlock3D(self.numFilters, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//4, self.height, self.width), mode='trilinear'),
                ResidualBlock3D(self.numFilters*2, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            )
            self.encoderVertMap = nn.Sequential(
                nn.Upsample(size=(self.numFrames//2, self.height, self.width//4), mode='trilinear'),
                ResidualBlock3D(2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//2, self.height, self.width//2), mode='trilinear'),
                ResidualBlock3D(self.numFilters, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//4, self.height, self.width), mode='trilinear'),
                ResidualBlock3D(self.numFilters*2, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            )
            self.decoder = nn.Sequential(
                ResidualBlock3D(self.numFilters*4, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//4, self.height, self.width), mode='trilinear'),
                ResidualBlock3D(self.numFilters*4, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//8, self.height, self.width), mode='trilinear'),
                ResidualBlock3D(self.numFilters*2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
                nn.Upsample(size=(self.numFrames//16, self.height, self.width), mode='trilinear'),
                nn.Conv3d(self.numFilters, self.numKeypoints, 1, 1, 0),
                #nn.ReLU() #use relu in 06/22 version
                nn.Sigmoid()
            )
    def forward(self, horiMap, vertMap):
        horiMap = horiMap.permute(0, 2, 1, 3, 4)
        #horiMap = F.interpolate(horiMap, size=(self.numFrames, self.height, self.width//4), mode='trilinear', align_corners=True)
        horiMap = self.encoderHoriMap(horiMap)

        vertMap = vertMap.permute(0, 2, 1, 3, 4)
        vertMap = self.encoderVertMap(vertMap)

        fusedMap = self.decoder(horiMap + vertMap)
        #fusedMap = F.interpolate(fusedMap, size=(self.numFrames, self.width*4, self.height*4), mode='trilinear', align_corners=True)
        
        return fusedMap