import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.layers import Identity, BasicBlock2D, BasicBlock3D, Block2D, DeconvBlock2D
from models.transformer import Transformer, ResNetBackbone, PositionEmbeddingLearned, Joiner, MLP


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

class BasicDecoder(nn.Module):
    def __init__(self, numKeypoints, numFilters, batchnorm=True, activation=nn.ReLU):
        super(BasicDecoder, self).__init__()
        self.decoder = nn.Sequential(
            BasicBlock2D(numFilters*2*8, numFilters*2*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(numFilters*2*8, numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
            #nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=True),
            BasicBlock2D(numFilters*4, numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(numFilters*4, numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
            #nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=True),
            BasicBlock2D(numFilters*2, numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(numFilters*2, numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, fusedMap):
        return self.sigmoid(self.decoder(fusedMap)).unsqueeze(2)

class BasicDecoder2(nn.Module):
    def __init__(self, numKeypoints, numFilters, activation=nn.PReLU):
        super(BasicDecoder2, self).__init__()
        self.decoder = nn.Sequential(
            DeconvBlock2D(numFilters*2*16, numFilters*16, (6, 6), (2, 2), (2, 2), activation),
            BasicBlock2D(numFilters*16, numFilters*16, (3, 3), (1, 1), (1, 1)),
            DeconvBlock2D(numFilters*16, numFilters*8, (6, 6), (2, 2), (2, 2), activation),
            BasicBlock2D(numFilters*8, numFilters*8, (3, 3), (1, 1), (1, 1)),
            DeconvBlock2D(numFilters*8, numFilters*4, (6, 6), (2, 2), (2, 2), activation),
            BasicBlock2D(numFilters*4, numFilters*4, (3, 3), (1, 1), (1, 1)),
            DeconvBlock2D(numFilters*4, numFilters*2, (6, 6), (2, 2), (2, 2), activation),
            nn.Conv2d(numFilters*2, numKeypoints, 1, 1, 0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, fusedMap):
        return self.sigmoid(self.decoder(fusedMap)).unsqueeze(2)


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
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        # self.conv1 = nn.Sequential(
        #     #nn.Upsample(size=(self.numGroupFrames, self.height, self.width), mode='trilinear', align_corners=True),
        #     #nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
        #     #nn.Conv3d(self.numFilters, self.numFilters*2, (3, 3, 3), (3, 1, 1), (0, 1, 1), (2, 1, 1)),
        #     nn.Conv3d(self.numFilters, self.numFilters*2, 3, 1, 1, 1),
        #     nn.BatchNorm3d(self.numFilters*2),
        #     nn.ReLU(),
        # )
        # self.conv2 = nn.Sequential(
        #     #nn.Upsample(size=(19, self.height//2, self.width//2), mode='trilinear', align_corners=True),
        #     nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
        #     nn.Conv3d(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
        #     nn.BatchNorm3d(self.numFilters*4),
        #     nn.ReLU(),
        # )
        # self.conv3 = nn.Sequential(
        #     #nn.Upsample(size=(5, self.height//4, self.width//4), mode='trilinear', align_corners=True),
        #     nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
        #     nn.Conv3d(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
        #     nn.BatchNorm3d(self.numFilters*8),
        #     nn.ReLU(),
        # )
        self.main = nn.Sequential(
            BasicBlock3D(self.numFilters, self.numFilters*2, 3, 1, 1, 1),
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
        )
        self.temporalMaxpool = nn.MaxPool3d((self.numGroupFrames//4, 1, 1), 1)
    def forward(self, maps):
        #maps = self.conv1(maps)
        #maps = self.conv2(maps)
        #maps = self.conv3(maps)
        maps = self.main(maps)
        maps = self.temporalMaxpool(maps)
        return maps.squeeze(2)

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
        self.mode = cfg.DATASET.mode
        self.numFilters = cfg.MODEL.numFilters #32 is the old version value
        self.frontModel = cfg.MODEL.frontModel

        if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            if self.frontModel == 'temporal':
                self.encoderHoriMap = TemporalModel(cfg)
                self.encoderVertMap = TemporalModel(cfg)
                #self.decoder = BasicDecoder2(self.numKeypoints, self.numFilters, activation=nn.LeakyReLU)
                self.decoder = BasicDecoder(self.numKeypoints, self.numFilters, batchnorm=False, activation=nn.PReLU)
            else:
                raise RuntimeError(F"activation should be temporal, not {self.mode}.")
            self.mnet = MNet(self.numFrames, self.numFilters)
        elif self.mode == 'multiChirps':
            if self.frontModel == 'spatial2':
                self.encoderHoriMap = nn.Sequential(
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
                self.encoderVertMap = nn.Sequential(
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
                self.decoder = BasicDecoder(self.numKeypoints, self.numFilters, batchnorm=False, activation=nn.PReLU)
            elif self.frontModel == 'spatial4':
                self.encoderHoriMap = nn.Sequential(
                    Block2D(self.numFilters, self.numFilters*2, (5, 5), (2, 2), (2, 2)),
                    BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
                    Block2D(self.numFilters*2, self.numFilters*4, (5, 5), (2, 2), (2, 2)),
                    BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
                    Block2D(self.numFilters*4, self.numFilters*8, (5, 5), (2, 2), (2, 2)),
                    BasicBlock2D(self.numFilters*8, self.numFilters*8, (3, 3), (1, 1), (1, 1)),
                    Block2D(self.numFilters*8, self.numFilters*16, (5, 5), (2, 2), (2, 2)),
                )
                self.encoderVertMap = nn.Sequential(
                    Block2D(self.numFilters, self.numFilters*2, (5, 5), (2, 2), (2, 2)),
                    BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
                    Block2D(self.numFilters*2, self.numFilters*4, (5, 5), (2, 2), (2, 2)),
                    BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
                    Block2D(self.numFilters*4, self.numFilters*8, (5, 5), (2, 2), (2, 2)),
                    BasicBlock2D(self.numFilters*8, self.numFilters*8, (3, 3), (1, 1), (1, 1)),
                    Block2D(self.numFilters*8, self.numFilters*16, (5, 5), (2, 2), (2, 2)),
                )
                self.decoder = BasicDecoder2(self.numKeypoints, self.numFilters, activation=nn.LeakyReLU)
            
            if self.frontModel == 'transformer':
                featureSize = 512
                self.backbone = ResNetBackbone('resnet50', train_backbone=True)
                self.backbone.body.conv1 = nn.Conv2d(self.numFilters, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.posEmbed = PositionEmbeddingLearned(self.height, self.width, num_pos_feats=featureSize//2)
                self.inputProj = nn.Conv2d(self.backbone.num_channels[-1], featureSize, 1)
                self.queryEmbed = nn.Embedding(self.numKeypoints, featureSize)
                self.encoderHoriMap = Transformer(d_model=featureSize)
                self.encoderVertMap = Transformer(d_model=featureSize)
                self.extractor = Joiner(self.backbone, self.posEmbed)
                self.kptEmbed = MLP(featureSize*2, featureSize//4, 2, 3)
                self.classEmbed = nn.Linear(featureSize*2, self.numKeypoints)
            self.mnet = MNet(self.numFrames, self.numFilters)
            #self.mnet = MNet(self.numFrames, 3)               
    
    def forward(self, horiMap, vertMap):
        
        if self.mode == 'multiChirps':
            horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, channel=2, # of chirps=16, 64, 8)
            vertMap = vertMap.permute(0, 2, 1, 3, 4)
            horiMap = self.mnet(horiMap).squeeze(2)
            vertMap = self.mnet(vertMap).squeeze(2)
            #print(vertMap.size())
            if 'spatial' in self.frontModel: 
                horiMap = self.encoderHoriMap(horiMap)
                vertMap = self.encoderVertMap(vertMap)
            elif 'transformer' in self.frontModel: 
                horiMap, pos = self.extractor(horiMap)
                vertMap, _ = self.extractor(vertMap)
                horiMap = self.encoderHoriMap(self.inputProj(horiMap[-1]), None, self.queryEmbed.weight, pos[-1])[0]
                vertMap = self.encoderVertMap(self.inputProj(vertMap[-1]), None, self.queryEmbed.weight, pos[-1])[0]
                logits = self.classEmbed(torch.cat((horiMap[-1], vertMap[-1]), 2))
                kpts = self.kptEmbed(torch.cat((horiMap[-1], vertMap[-1]), 2)).sigmoid()
                #print(self.backbone.body.conv1.weight[0,0,0])
                #print('query', self.queryEmbed.weight[0])
                return (logits, kpts)
            else:
                raise RuntimeError("frontModel should be spatial/transformer, not {self.frontModel}.")
            
            
        elif self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            batchSize = horiMap.size(0)
            horiMap = horiMap.permute(0, 1, 3, 2, 4, 5) # (b, # of frames=30, channel=2, # of chirps=16, 64, 8)
            vertMap = vertMap.permute(0, 1, 3, 2, 4, 5)
            if self.mode == 'multiFrames':
                horiMap = horiMap.squeeze(3)
                vertMap = vertMap.squeeze(3)
                horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, # of channels, # of frames, h, w)
                vertMap = vertMap.permute(0, 2, 1, 3, 4)
            else:
                horiMap = horiMap.view(batchSize * self.numGroupFrames, 2, self.numFrames, self.rangeSize, self.azimuthSize)
                vertMap = vertMap.view(batchSize * self.numGroupFrames, 2, self.numFrames, self.rangeSize, self.azimuthSize)
                horiMap = self.mnet(horiMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
                vertMap = self.mnet(vertMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
                horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, channel=2, # of frames=30, 64, 8)
                vertMap = vertMap.permute(0, 2, 1, 3, 4)
            
            horiMap = self.encoderHoriMap(horiMap)
            vertMap = self.encoderVertMap(vertMap)
            #print(vertMap.size())
        fusedMap = self.decoder(torch.cat((horiMap, vertMap), 1))
        #print(fusedMap.size())
        return fusedMap
