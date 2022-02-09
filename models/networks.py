import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_networks import *
from models.chirp_networks import *
from models.layers import BasicDecoder, BasicDecoderGCN

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
        self.useVelocity = cfg.DATASET.useVelocity
        self.numFilters = cfg.MODEL.numFilters
        self.frontModel = cfg.MODEL.frontModel
        self.backbone = cfg.MODEL.backbone
        self.chirpModel = cfg.MODEL.chirpModel
        self.gcnType = cfg.MODEL.gcnType

        if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            ###################################
            #model for preprocessing
            ###################################
            if self.chirpModel == 'maxpool':
                self.mnet = None
            elif self.chirpModel == 'mnet':
                self.mnet = MNet(self.numFrames, self.numFilters, self.useVelocity)
            elif self.chirpModel == 'identity':
                self.mnet = Identity()
            else:
                raise RuntimeError(F"chirpModel should be maxpool/mnet, not {self.chirpModel}.")
            
            ###################################
            #main model architecture
            ###################################
            if self.frontModel == 'temporal':#Baseline
                self.encoderHoriMap = TemporalModel(cfg)
                self.encoderVertMap = TemporalModel(cfg)
                if self.gcnType == 'None':
                    self.decoder = BasicDecoder(cfg, batchnorm=False, activation=nn.PReLU, backbone=self.backbone)
                else:
                    self.decoder = BasicDecoderGCN(cfg, batchnorm=False, activation=nn.PReLU, backbone=self.backbone)
            elif self.frontModel == 'temporal2':#U-Net
                self.temporalHoriNet = TemporalModel2(cfg, batchnorm=False, activation=nn.PReLU)
                self.temporalVertNet = TemporalModel2(cfg, batchnorm=False, activation=nn.PReLU)
                if self.gcnType == 'None':
                    self.fusedNet = FusedNet(cfg, batchnorm=False, activation=nn.PReLU)
                else:
                    self.fusedNet = FusedNetGCN(cfg, batchnorm=False, activation=nn.PReLU)
            elif self.frontModel == 'temporal3':#U-Net + DF
                self.temporalHoriNet = TemporalModel3(cfg, batchnorm=False, activation=nn.PReLU)
                self.temporalVertNet = TemporalModel3(cfg, batchnorm=False, activation=nn.PReLU)
                self.fusedNet = DynamicFilterNet(cfg, batchnorm=False, activation=nn.PReLU)
            elif self.frontModel == 'temporal4':#U-Net with InceptionA block + DF
                self.temporalHoriNet = TemporalModel4(cfg, batchnorm=False, activation=nn.PReLU)
                self.temporalVertNet = TemporalModel4(cfg, batchnorm=False, activation=nn.PReLU)
                self.fusedNet = DynamicFilterNet(cfg, batchnorm=False, activation=nn.PReLU, dfCh1=192, dfCh2=256)
            elif self.frontModel == 'temporal5':#U-Net + DF + GCN
                self.temporalHoriNet = TemporalModel3(cfg, batchnorm=False, activation=nn.PReLU)
                self.temporalVertNet = TemporalModel3(cfg, batchnorm=False, activation=nn.PReLU)
                self.fusedNet = DynamicFilterNetGCN(cfg, batchnorm=False, activation=nn.PReLU)
            else:
                raise RuntimeError(F"activation should be temporal~5, not {self.frontModel}.")
        else:
            raise RuntimeError(F"activation should be multiFramesChirps/multiFrames, not {self.mode}.")

    def forward(self, horiMap, vertMap):
        batchSize = horiMap.size(0)
        
        if self.chirpModel == 'maxpool':
            horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, # of channel, # of frames, h, w)
            vertMap = vertMap.permute(0, 2, 1, 3, 4)
        elif self.chirpModel == 'identity':
            ######################################################
            #input shape: (b, # of frames, 1, # of channels, h, w)
            ######################################################
            horiMap = horiMap.squeeze(2)
            vertMap = vertMap.squeeze(2)              
            horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, # of channels, # of frames, h, w)
            vertMap = vertMap.permute(0, 2, 1, 3, 4)
        else:
            horiMap = horiMap.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize)
            vertMap = vertMap.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize)
            horiMap = self.mnet(horiMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
            vertMap = self.mnet(vertMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
            horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, channel=2, # of frames=30, 64, 8)
            vertMap = vertMap.permute(0, 2, 1, 3, 4)
        
        if self.frontModel == 'temporal':# Baseline
            horiMap = self.encoderHoriMap(horiMap)
            vertMap = self.encoderVertMap(vertMap)
            fusedMap = self.decoder(torch.cat((horiMap, vertMap), 1))
        elif self.frontModel == 'temporal2':#U-Net 
            hl1maps, hl2maps, horiMap = self.temporalHoriNet.encode(horiMap)
            vl1maps, vl2maps, vertMap = self.temporalVertNet.encode(vertMap)
            l1maps = torch.cat((hl1maps, vl1maps), 1)
            l2maps = torch.cat((hl2maps, vl2maps), 1)
            fusedMap = torch.cat((horiMap, vertMap), 1)
            fusedMap = self.fusedNet(l1maps, l2maps, fusedMap)
        elif self.frontModel == 'temporal3' or self.frontModel == 'temporal4' or self.frontModel == 'temporal5':
            hl1maps, hl2maps, horiMap = self.temporalHoriNet.encode(horiMap)
            vl1maps, vl2maps, vertMap = self.temporalVertNet.encode(vertMap)
            fusedMap = torch.cat((horiMap, vertMap), 1)
            fusedMap = self.fusedNet(hl1maps, hl2maps, vl1maps, vl2maps, fusedMap)
        return fusedMap


class RFPose(nn.Module):
    def __init__(self, cfg):
        super(RFPose, self).__init__()
        numEncodeLayers = 4 #10
        numDecodeLayers = 3
        self.nf = cfg.MODEL.numFilters
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.nfEncoderList = [[3 if cfg.DATASET.useVelocity else 2, self.nf],
                              [self.nf, self.nf * 2],
                              [self.nf * 2, self.nf * 4],
                              [self.nf * 4, self.nf * 8]]
        self.nfDecoderList = [[self.nf * 8 * 2, self.nf * 4],
                              [self.nf * 4, self.nf * 2],
                              [self.nf * 2, self.nf]]

        self.RFHoriEncodeNet = nn.ModuleList([nn.Sequential(
            nn.Conv3d(self.nfEncoderList[i][0], self.nfEncoderList[i][1], (9, 5, 5), (1, 2, 2), (3, 2, 2), bias=False),
            nn.BatchNorm3d(self.nfEncoderList[i][1]),
            nn.ReLU(),
            #################
            #  different part
            nn.Conv3d(self.nfEncoderList[i][1], self.nfEncoderList[i][1], (9, 3, 3), (1, 1, 1), (4, 1, 1), bias=False),
            nn.BatchNorm3d(self.nfEncoderList[i][1]),
            nn.ReLU()
            #################
        ) for i in range(numEncodeLayers)])
        self.RFVertEncodeNet = nn.ModuleList([nn.Sequential(
            nn.Conv3d(self.nfEncoderList[i][0], self.nfEncoderList[i][1], (9, 5, 5), (1, 2, 2), (3, 2, 2), bias=False),
            nn.BatchNorm3d(self.nfEncoderList[i][1]),
            nn.ReLU(),
            #################
            #  different part
            nn.Conv3d(self.nfEncoderList[i][1], self.nfEncoderList[i][1], (9, 3, 3), (1, 1, 1), (4, 1, 1), bias=False),
            nn.BatchNorm3d(self.nfEncoderList[i][1]),
            nn.ReLU()
            #################
        ) for i in range(numEncodeLayers)])
        
        
        self.PoseDecodeNet = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose3d(self.nfDecoderList[i][0], self.nfDecoderList[i][1], (3, 6, 6), (1, 2, 2), (0, 2, 2), bias=False),
            nn.PReLU()
        ) for i in range(numDecodeLayers)])
        #################
        #  different part
        self.PoseDecodeNet.append(nn.Sequential(
            #nn.ConvTranspose3d(self.numFilters, self.numFilters, (3, 6, 6), (1, 4, 4), (0, 1, 1), bias=False),
            nn.ConvTranspose3d(self.nf, self.numKeypoints, (3, 6, 6), (1, 2, 2), (0, 2, 2), bias=False),
            nn.Sigmoid())
        )
        #################
    def forward(self, horiMap, vertMap):
        horiMap = horiMap.squeeze(2).permute(0, 2, 1, 3, 4) # alyways eliminate chirp dimension
        vertMap = vertMap.squeeze(2).permute(0, 2, 1, 3, 4)
        for i, l in enumerate(self.RFHoriEncodeNet):
            horiMap = l(horiMap)
        for i, l in enumerate(self.RFVertEncodeNet):
            vertMap = l(vertMap)
        mergeMap = torch.cat((horiMap, vertMap), dim=1)
        for i, l in enumerate(self.PoseDecodeNet):
            mergeMap = l(mergeMap)
        return mergeMap