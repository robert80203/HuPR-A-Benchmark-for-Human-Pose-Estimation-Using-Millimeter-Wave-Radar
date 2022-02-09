import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.layers import BasicBlock3D, BasicBlock2D, InceptionA3d
import math
from models.gcn_networks import *

class TemporalModel(nn.Module):
    def __init__(self, cfg):
        super(TemporalModel, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.chirpModel = cfg.MODEL.chirpModel
        if cfg.MODEL.backbone == 'resnet':
            self.main = nn.Sequential(
                BasicBlock3D(2 if self.chirpModel == 'maxpool' else self.numFilters, self.numFilters*2, 3, 1, 1, 1),
                nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
                BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
                nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
                BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
            )
        elif cfg.MODEL.backbone == 'resnet_deeper':
            self.main = nn.Sequential(
                BasicBlock3D(2 if self.chirpModel == 'maxpool' else self.numFilters, self.numFilters*2, 3, 1, 1, 1),
                nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
                BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
                nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
                BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
                nn.Upsample(size=(self.numGroupFrames//4, self.width//8, self.height//8), mode='trilinear', align_corners=True),
                BasicBlock3D(self.numFilters*8, self.numFilters*16, 3, 1, 1, 1),
            )
        elif cfg.MODEL.backbone == 'inception':
            self.main = nn.Sequential(
                BasicBlock3D(2 if self.chirpModel == 'maxpool' else self.numFilters, 32, 3, 1, 1),
                BasicBlock3D(32, 64, 3, 1, 1),
                BasicBlock3D(64, 80, 3, 1, 1),
                BasicBlock3D(80, 192, 3, 1, 1),
                nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
                InceptionA3d(192, pool_features=32),
                nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
                InceptionA3d(256, pool_features=64),
            )
        self.temporalMaxpool = nn.MaxPool3d((self.numGroupFrames//4, 1, 1), 1)
    def forward(self, maps):
        #maps = self.conv1(maps)
        #maps = self.conv2(maps)
        #maps = self.conv3(maps)
        maps = self.main(maps)
        maps = self.temporalMaxpool(maps)
        return maps.squeeze(2)

class TemporalModel2(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(TemporalModel2, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.layer1 = nn.Sequential(
            BasicBlock3D(self.numFilters, self.numFilters*2, 3, 1, 1, 1),
            #BasicBlock3D(self.numFilters*2, self.numFilters*2, 3, 1, 1, 1),
        )
        self.layer1_1x1 = nn.Conv3d(self.numFilters*2, self.numFilters*2, (self.numGroupFrames, 1, 1), 1, 0, bias=False)
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
            #BasicBlock3D(self.numFilters*4, self.numFilters*4, 3, 1, 1, 1),
        )
        self.layer2_1x1 = nn.Conv3d(self.numFilters*4, self.numFilters*4, (self.numGroupFrames//2, 1, 1), 1, 0, bias=False)
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
            #BasicBlock3D(self.numFilters*8, self.numFilters*8, 3, 1, 1, 1),
        )
        #self.temporalMerge = nn.MaxPool3d((self.numGroupFrames//4, 1, 1), 1)
        self.temporalMerge = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.numGroupFrames//4, 1, 1), 1, 0, bias=False)
    def encode(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        maps = self.temporalMerge(l3maps).squeeze(2)
        l1maps = self.layer1_1x1(l1maps).squeeze(2)
        l2maps = self.layer2_1x1(l2maps).squeeze(2)
        return l1maps, l2maps, maps

class TemporalModel3(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(TemporalModel3, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.layer1 = nn.Sequential(
            BasicBlock3D(self.numFilters if cfg.MODEL.chirpModel == 'mnet' else 2, self.numFilters*2, 3, 1, 1, 1),
            #BasicBlock3D(self.numFilters*2, self.numFilters*2, 3, 1, 1, 1),
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
            #BasicBlock3D(self.numFilters*4, self.numFilters*4, 3, 1, 1, 1),
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
            #BasicBlock3D(self.numFilters*8, self.numFilters*8, 3, 1, 1, 1),
        )
        #self.temporalMerge = nn.MaxPool3d((self.numGroupFrames//4, 1, 1), 1)
        self.temporalMerge = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.numGroupFrames//4, 1, 1), 1, 0, bias=False)
    def encode(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        maps = self.temporalMerge(l3maps).squeeze(2)
        return l1maps, l2maps, maps

class TemporalModel4(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(TemporalModel4, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.layer1 = nn.Sequential(
            BasicBlock3D(self.numFilters, 32, 3, 1, 1),
            BasicBlock3D(32, 64, 3, 1, 1),
            BasicBlock3D(64, 80, 3, 1, 1),
            BasicBlock3D(80, 192, 3, 1, 1),
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            InceptionA3d(192, pool_features=32),
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            InceptionA3d(256, pool_features=64),
        )
        self.layer1_1x1conv = BasicBlock3D(192, self.numFilters*2, 1, 1, 0)
        self.layer2_1x1conv = BasicBlock3D(256, self.numFilters*4, 1, 1, 0)
        #self.temporalMerge = nn.MaxPool3d((self.numGroupFrames//4, 1, 1), 1)
        self.temporalMerge = nn.Conv3d(288, self.numFilters*8, (self.numGroupFrames//4, 1, 1), 1, 0, bias=False)
    def encode(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        maps = self.temporalMerge(l3maps).squeeze(2)
        return self.layer1_1x1conv(l1maps), self.layer2_1x1conv(l2maps), maps

class TemporalModel5(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(TemporalModel5, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.layer1 = nn.Sequential(
            BasicBlock3D(self.numFilters if cfg.MODEL.chirpModel == 'mnet' else 2, self.numFilters*2, 3, 1, 1, 1)
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1)
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1)
        )
        self.temporalMerge = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.numGroupFrames//4, 1, 1), 1, 0, bias=False)
    def encode(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        maps = self.temporalMerge(l3maps).squeeze(2)
        return l1maps, l2maps, maps

class TemporalModel6(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU, fuseLast=False):
        super(TemporalModel6, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.fuseLast = fuseLast
        self.layer1 = nn.Sequential(
            #BasicBlock3D(self.numFilters, self.numFilters*2, 3, 1, 1, 1),
            nn.Conv3d(self.numFilters, self.numFilters*2, (5, 5, 5), (1, 1, 1), (2, 2, 2), bias=False),
            nn.BatchNorm3d(self.numFilters*2),
            nn.ReLU(),
            nn.Conv3d(self.numFilters*2, self.numFilters*2, (3, 3, 3), (2, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(self.numFilters*2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            #nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            #BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
            nn.Conv3d(self.numFilters*2, self.numFilters*4, (5, 5, 5), (1, 1, 1), (2, 2, 2), bias=False),
            nn.BatchNorm3d(self.numFilters*4),
            nn.ReLU(),
            nn.Conv3d(self.numFilters*4, self.numFilters*4, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=False),
            nn.BatchNorm3d(self.numFilters*4),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            #nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            #BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
            nn.Conv3d(self.numFilters*4, self.numFilters*8, (5, 5, 5), (1, 1, 1), (2, 2, 2), bias=False),
            nn.BatchNorm3d(self.numFilters*8),
            nn.ReLU(),
            nn.Conv3d(self.numFilters*8, self.numFilters*8, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=False),
            nn.BatchNorm3d(self.numFilters*8),
            nn.ReLU()
        )
        if fuseLast:
            self.temporalMerge = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.numGroupFrames//6, 1, 1), 1, 0, bias=False)

    def forward(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        if self.fuseLast:
            l3maps = self.temporalMerge(l3maps)
        return l1maps, l2maps, l3maps

class TemporalModel7(nn.Module):
    def __init__(self, cfg, fuseLast=False):
        super(TemporalModel7, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.chirpModel = cfg.MODEL.chirpModel
        self.fuseLast = fuseLast
        self.layer1 = BasicBlock3D(2 if self.chirpModel == 'identity' else self.numFilters, self.numFilters*2, 3, 1, 1, 1)
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1, 1),
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1, 1),
        )
        if fuseLast:
            self.temporalMerge = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.numGroupFrames//4, 1, 1), 1, 0, bias=False)

    def forward(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        if self.fuseLast:
            l3maps = self.temporalMerge(l3maps)
        return l1maps, l2maps, l3maps

class DynamicFilterNetGCN(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(DynamicFilterNetGCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.decoderLayer1 = nn.Sequential(
            BasicBlock2D(self.numFilters*8*2, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4*3, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer3 = nn.Sequential(
            BasicBlock2D(self.numFilters*2*3, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
        )
        self.dfNet2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters*4, self.numGroupFrames//2, (3, 3), (1, 1), (1, 1)),
        )
        self.dfNet1 = nn.Sequential(
            BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters*2, self.numGroupFrames, (3, 3), (1, 1), (1, 1)),
        )
        self.sigmoid = nn.Sigmoid()
        A = torch.tensor([
            # [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RightAnkle
            # [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RightKnee
            # [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RightHip
            # [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LeftHip
            # [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LeftKnee
            # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LeftAnkle
            # [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#Pelvis
            # [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],#chest
            # [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],#neck
            # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#head
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],#rightwrist
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],#rightelbow
            # [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],#rightshoulder
            # [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],#leftshoulder
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],#leftelbow
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],#leftwrist
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RHip
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RKnee
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RAnkle
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LHip
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#LKnee
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#LAnkle
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#Neck
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#Head
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],#LShoulder
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],#LElbow
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],#LWrist
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],#RShoulder
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],#RElbow
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]#RWrist
        ], dtype=torch.float).cuda()
        if 'gcn1' in self.gcnType:
            self.gcn = GCN1(cfg, A)
        elif 'gcn2' in self.gcnType:
            self.gcn = GCN2(cfg, A)
            
    def forward(self, hl1maps, hl2maps, vl1maps, vl2maps, maps):
        maps = self.decoderLayer1(maps)
        #l1jointfeat = self.level1JointFeat(maps).squeeze(2)
        dfmaps2 = self.dfNet2(maps).unsqueeze(1)
        hl2maps = torch.sum(dfmaps2 * hl2maps, dim=2)
        vl2maps = torch.sum(dfmaps2 * vl2maps, dim=2)
        maps = self.decoderLayer2(torch.cat((maps, hl2maps, vl2maps), 1))
        #l2jointfeat = self.level2JointFeat(maps).squeeze(2)
        dfmaps1 = self.dfNet1(maps).unsqueeze(1)
        hl1maps = torch.sum(dfmaps1 * hl1maps, dim=2)
        vl1maps = torch.sum(dfmaps1 * vl1maps, dim=2)
        output = self.decoderLayer3(torch.cat((maps, hl1maps, vl1maps), 1))
        #l3jointfeat = self.level3JointFeat(output).squeeze(2)

        gcnoutput = self.gcn(output)

        if self.modelType == "heatmapMSE":
            return output.unsqueeze(2)
        elif self.modelType == "heatmap_regress" or self.modelType == "heatmap_heatmap":
            return self.sigmoid(output).unsqueeze(2), gcnoutput
        elif self.modelType == "justregress":
            return gcnoutput
        else:
            return self.sigmoid(output).unsqueeze(2)   
    

class FusedNet(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(FusedNet, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.decoderLayer1 = nn.Sequential(
            BasicBlock2D(self.numFilters*8*2, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4*3, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer3 = nn.Sequential(
            BasicBlock2D(self.numFilters*2*3, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, l1maps, l2maps, maps):
        maps = self.decoderLayer1(maps)
        #print(maps.size(), l1maps.size(), l2maps.size())
        maps = self.decoderLayer2(torch.cat((maps, l2maps), 1))
        output = self.decoderLayer3(torch.cat((maps, l1maps), 1))
        return self.sigmoid(output).unsqueeze(2)      


class FusedNetGCN(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(FusedNetGCN, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.decoderLayer1 = nn.Sequential(
            BasicBlock2D(self.numFilters*8*2, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4*3, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer3 = nn.Sequential(
            BasicBlock2D(self.numFilters*2*3, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
        )
        self.sigmoid = nn.Sigmoid()
        A = torch.tensor([
            # [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RightAnkle
            # [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RightKnee
            # [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RightHip
            # [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LeftHip
            # [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LeftKnee
            # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LeftAnkle
            # [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#Pelvis
            # [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],#chest
            # [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],#neck
            # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#head
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],#rightwrist
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],#rightelbow
            # [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],#rightshoulder
            # [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],#leftshoulder
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],#leftelbow
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],#leftwrist
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RHip
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RKnee
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RAnkle
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LHip
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#LKnee
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#LAnkle
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#Neck
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#Head
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],#LShoulder
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],#LElbow
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],#LWrist
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],#RShoulder
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],#RElbow
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]#RWrist
        ], dtype=torch.float).cuda()
        if 'gcn1' in self.gcnType:
            self.gcn = GCN1(cfg, A)
        elif 'gcn2' in self.gcnType:
            self.gcn = GCN2(cfg, A)
    
    def forward(self, l1maps, l2maps, maps):
        maps = self.decoderLayer1(maps)
        maps = self.decoderLayer2(torch.cat((maps, l2maps), 1))
        output = self.decoderLayer3(torch.cat((maps, l1maps), 1))

        gcnoutput = self.gcn(output)

        if self.modelType == "heatmapMSE":
            return output.unsqueeze(2)
        elif self.modelType == "heatmap_regress" or self.modelType == "heatmap_heatmap":
            return self.sigmoid(output).unsqueeze(2), gcnoutput
        elif self.modelType == "justregress":
            return gcnoutput
        else:
            return self.sigmoid(output).unsqueeze(2) 

class DynamicFilterNet(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU, dfCh1=None, dfCh2=None):
        super(DynamicFilterNet, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.decoderLayer1 = nn.Sequential(
            BasicBlock2D(self.numFilters*8*2, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4*3, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer3 = nn.Sequential(
            BasicBlock2D(self.numFilters*2*3, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
            BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
        )
        self.sigmoid = nn.Sigmoid()
        self.dfNet2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters*4, self.numGroupFrames//2, (3, 3), (1, 1), (1, 1)),
        )
        self.dfNet1 = nn.Sequential(
            BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1)),
            BasicBlock2D(self.numFilters*2, self.numGroupFrames, (3, 3), (1, 1), (1, 1)),
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, hl1maps, hl2maps, vl1maps, vl2maps, maps):
        maps = self.decoderLayer1(maps)
        #dfmaps2 = self.softmax(self.dfNet2(maps)).unsqueeze(1)
        dfmaps2 = self.dfNet2(maps).unsqueeze(1)
        hl2maps = torch.sum(dfmaps2 * hl2maps, dim=2)
        vl2maps = torch.sum(dfmaps2 * vl2maps, dim=2)
        maps = self.decoderLayer2(torch.cat((maps, hl2maps, vl2maps), 1))
        #dfmaps1 = self.softmax(self.dfNet1(maps)).unsqueeze(1)
        dfmaps1 = self.dfNet1(maps).unsqueeze(1)
        hl1maps = torch.sum(dfmaps1 * hl1maps, dim=2)
        vl1maps = torch.sum(dfmaps1 * vl1maps, dim=2)
        output = self.decoderLayer3(torch.cat((maps, hl1maps, vl1maps), 1))
        if self.modelType == "heatmapMSE":
            return output.unsqueeze(2) 
        else:
            return self.sigmoid(output).unsqueeze(2)    
