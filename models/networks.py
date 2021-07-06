import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *


class MNet(nn.Module):
    def __init__(self, numFrames, numFilters):
        super(MNet, self).__init__()
        sizeTemp = sizeTempStride = numFrames//2
        self.temporalConvWx1x1 = nn.Conv3d(2, numFilters, (2, 1, 1), (2, 1, 1), (0, 0, 0))
        self.temporalMaxpool = nn.MaxPool3d((sizeTemp, 1, 1), (sizeTempStride, 1, 1))
    def forward(self, chirpMaps):
        # chirpMaps becomes : (batch, 2, # of chirps, h, w)
        chirpMaps = self.temporalConvWx1x1(chirpMaps)
        maps = self.temporalMaxpool(chirpMaps)
        return maps

class RODNetSkipConnections(nn.Module):
    def __init__(self, numKeypoints, numFilters):
        super(RODNetSkipConnections, self).__init__()
        self.conv1_1 = ResidualBlock2D(numFilters*2, numFilters, (3, 3), (1, 1), (1, 1))
        self.conv1_2 = ResidualBlock2D(numFilters, numFilters*2, (3, 3), (1, 1), (1, 1))
        
        self.conv2_1 = ResidualBlock2D(numFilters*2, numFilters*2, (3, 3), (1, 1), (1, 1))
        self.conv2_2 = ResidualBlock2D(numFilters*2, numFilters*4, (3, 3), (1, 1), (1, 1))

        self.conv3_1 = ResidualBlock2D(numFilters*4, numFilters*4, (3, 3), (1, 1), (1, 1))
        self.conv3_2 = ResidualBlock2D(numFilters*4, numFilters*8, (3, 3), (1, 1), (1, 1))
        
        self.deconv3 = nn.Sequential(
            nn.Conv2d(numFilters*8, numFilters*4, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(numFilters*4),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(numFilters*4, numFilters*2, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(numFilters*2),
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(numFilters*2, numFilters, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(numFilters),
        )
        self.toHeatmap = nn.Sequential(
            nn.Conv2d(numFilters, numKeypoints, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )
    def forward(self, fusedMap):
        residual1 = self.conv1_1(fusedMap)
        output1 =  self.conv1_2(residual1)
        residual2 = F.interpolate(output1, scale_factor=0.5, mode='bilinear', align_corners=True)
        residual2 = self.conv2_1(residual2)
        output2 = self.conv2_2(residual2)
        residual3 = F.interpolate(output2, scale_factor=0.5, mode='bilinear', align_corners=True)
        residual3 = self.conv3_1(residual3)
        output3 = self.conv3_2(residual3)
        
        output2 = self.deconv3(output3) + residual3
        output2 = F.interpolate(output2, scale_factor=2.0, mode='bilinear', align_corners=True)
        output1 = self.deconv2(output2) + residual2
        output1 = F.interpolate(output1, scale_factor=2.0, mode='bilinear', align_corners=True)
        output = self.deconv1(output1) + residual1
        heatmaps = self.toHeatmap(output).unsqueeze(2)
        return heatmaps

class TemporalModel(nn.Module):
    def __init__(self, cfg):
        super(TemporalModel, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        
        
        self.conv0 = nn.Sequential(
            nn.Upsample(size=(self.numFrames, self.height, self.width//4), mode='trilinear', align_corners=True),
            ResidualBlock3D(self.numFilters, self.numFilters, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )
        self.conv1 = nn.Sequential(
            nn.Upsample(size=(self.numFrames, self.height, self.width//2), mode='trilinear', align_corners=True),
            ResidualBlock3D(self.numFilters, self.numFilters, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )
        self.conv1_dilated_2 = nn.Sequential(
            nn.Upsample(size=(self.numFrames, self.height, self.width//2), mode='trilinear', align_corners=True),
            ResidualBlock3D(self.numFilters, self.numFilters, (3, 3, 3), (1, 1, 1), (2, 1, 1), (2, 1, 1)),
        )
        self.conv1_dilated_4 = nn.Sequential(
            nn.Upsample(size=(self.numFrames, self.height, self.width//2), mode='trilinear', align_corners=True),
            ResidualBlock3D(self.numFilters, self.numFilters, (3, 3, 3), (1, 1, 1), (4, 1, 1), (4, 1, 1)),
        )
        self.conv2 = nn.Sequential(
            nn.Upsample(size=(self.numFrames, self.height, self.width), mode='trilinear', align_corners=True),
            ResidualBlock3D(self.numFilters*3, self.numFilters, (1, 1, 1), (1, 1, 1), (0, 0, 0)),
            nn.Conv3d(self.numFilters, self.numFilters, (self.numFrames, 1, 1), (1, 1, 1), (0, 0, 0)),
        )
    def forward(self, maps):
        #print(maps.size())
        maps = self.conv0(maps)
        #print(maps.size())
        maps1 = self.conv1(maps)
        maps1_2 = self.conv1_dilated_2(maps)
        maps1_4 = self.conv1_dilated_4(maps)
        #print(maps1.size(), maps1_2.size())
        output = self.conv2(torch.cat((maps1, maps1_2, maps1_4), 1)).squeeze(2)
        #print(output.size())
        return output
    
class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.rangeSize = cfg.DATASET.rangeSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numFilters = cfg.MODEL.numFilters #32 is the old version value
        self.mode = cfg.DATASET.mode
        '''
        if self.mode == 'multiFrames':
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
        '''
        if self.mode == 'multiFramesChirps':
            self.encoderHoriMap = TemporalModel(cfg)
            self.encoderVertMap = TemporalModel(cfg)
            self.decoder = RODNetSkipConnections(self.numKeypoints, self.numFilters)
            self.mnet = MNet(self.numFrames, self.numFilters)
            self.temporalMaxpool = nn.MaxPool3d((self.numGroupFrames, 1, 1), (self.numGroupFrames, 1, 1))
        elif self.mode == 'multiChirps':
            self.encoderHoriMap = nn.Sequential(
                nn.Upsample(size=(self.height, self.width//4), mode='bilinear', align_corners=True),
                ResidualBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
                nn.Upsample(size=(self.height, self.width//2), mode='bilinear', align_corners=True),
                ResidualBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
                nn.Upsample(size=(self.height, self.width), mode='bilinear', align_corners=True),
                ResidualBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
            )
            self.encoderVertMap = nn.Sequential(
                nn.Upsample(size=(self.height, self.width//4), mode='bilinear', align_corners=True),
                ResidualBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
                nn.Upsample(size=(self.height, self.width//2), mode='bilinear', align_corners=True),
                ResidualBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
                nn.Upsample(size=(self.height, self.width), mode='bilinear', align_corners=True),
                ResidualBlock2D(self.numFilters, self.numFilters, (3, 3), (1, 1), (1, 1)),
            )
            self.decoder = RODNetSkipConnections(self.numKeypoints, self.numFilters)
            self.mnet = MNet(self.numFrames, self.numFilters)            

    def forward(self, horiMap, vertMap):
        
        if self.mode == 'multiChirps':
            horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, channel=2, # of chirps=16, 64, 8)
            vertMap = vertMap.permute(0, 2, 1, 3, 4)
            horiMap = self.mnet(horiMap).squeeze(2)
            vertMap = self.mnet(vertMap).squeeze(2)
            horiMap = self.encoderHoriMap(horiMap)
            vertMap = self.encoderVertMap(vertMap)
        elif self.mode == 'multiFramesChirps':
            batchSize = horiMap.size(0)
            horiMap = horiMap.permute(0, 1, 3, 2, 4, 5) # (b, # of frames=30, channel=2, # of chirps=16, 64, 8)
            vertMap = vertMap.permute(0, 1, 3, 2, 4, 5)
            #print(horiMap.size())
            horiMap = horiMap.view(batchSize * self.numGroupFrames, 2, self.numFrames, self.rangeSize, self.azimuthSize)
            vertMap = vertMap.view(batchSize * self.numGroupFrames, 2, self.numFrames, self.rangeSize, self.azimuthSize)
            horiMap = self.mnet(horiMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
            vertMap = self.mnet(vertMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
            horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, channel=2, # of frames=30, 64, 8)
            vertMap = vertMap.permute(0, 2, 1, 3, 4)
            ####
            #horiMap = self.temporalMaxpool(horiMap).squeeze(2)
            #vertMap = self.temporalMaxpool(vertMap).squeeze(2)
            ####
            horiMap = self.encoderHoriMap(horiMap)
            vertMap = self.encoderVertMap(vertMap)
            ####
        
        #fusedMap = self.decoder(horiMap + vertMap) # old version, multiFrames and multiChirps
        fusedMap = self.decoder(torch.cat((horiMap, vertMap), 1))
        return fusedMap