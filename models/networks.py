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

class BaseModel(nn.Module):
    def __init__(self, numFrames, numKeypoints, dimsWidthHeight, mode):
        super(BaseModel, self).__init__()
        self.numFrames = numFrames
        self.numKeypoints = numKeypoints
        self.width = dimsWidthHeight[0]
        self.height = dimsWidthHeight[1]
        self.numFilters = 16 #32 is the old version value
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
            # self.encoderHoriMap = nn.Sequential(
            #     nn.Upsample(size=(self.numFrames//2, self.height, self.width//4), mode='trilinear'),
            #     ResidualBlock3D(2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//2, self.height, self.width//2), mode='trilinear'),
            #     ResidualBlock3D(self.numFilters, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//4, self.height, self.width), mode='trilinear'),
            #     ResidualBlock3D(self.numFilters*2, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            # )
            # self.encoderVertMap = nn.Sequential(
            #     nn.Upsample(size=(self.numFrames//2, self.height, self.width//4), mode='trilinear'),
            #     ResidualBlock3D(2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//2, self.height, self.width//2), mode='trilinear'),
            #     ResidualBlock3D(self.numFilters, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//4, self.height, self.width), mode='trilinear'),
            #     ResidualBlock3D(self.numFilters*2, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            # )
            # self.decoder = nn.Sequential(
            #     ResidualBlock3D(self.numFilters*4, self.numFilters*4, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//4, self.height, self.width), mode='trilinear'),
            #     ResidualBlock3D(self.numFilters*4, self.numFilters*2, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//8, self.height, self.width), mode='trilinear'),
            #     ResidualBlock3D(self.numFilters*2, self.numFilters, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            #     nn.Upsample(size=(self.numFrames//16, self.height, self.width), mode='trilinear'),
            #     nn.Conv3d(self.numFilters, self.numKeypoints, 1, 1, 0),
            #     nn.Sigmoid()
            # )
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
        horiMap = horiMap.permute(0, 2, 1, 3, 4)
        vertMap = vertMap.permute(0, 2, 1, 3, 4)
        if self.mode == 'multiChirps':
            horiMap = self.mnet(horiMap).squeeze(2)
            vertMap = self.mnet(vertMap).squeeze(2)
        horiMap = self.encoderHoriMap(horiMap)
        vertMap = self.encoderVertMap(vertMap)
        #fusedMap = self.decoder(horiMap + vertMap) # old version, multiFrames and multiChirps
        fusedMap = self.decoder(torch.cat((horiMap, vertMap), 1))
        return fusedMap