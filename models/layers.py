import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcn_networks import PRGCN


class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True, activation=nn.ReLU):
        super(BasicBlock2D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            )
        self.relu = activation()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.main(x) + residual
        out = self.relu(out)
        return out

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True, activation=nn.ReLU):
        super(BasicBlock3D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(out_channels),
                activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            )
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            )
        self.relu = activation()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.main(x) + residual
        out = self.relu(out)
        return out

class MultiScaleCrossSelfAttentionPRGCN(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(MultiScaleCrossSelfAttentionPRGCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints

        self.decoderLayer3 = nn.Sequential(
            BasicBlock2D(self.numFilters*8*4, self.numFilters*8, 3, 1, 1, batchnorm, activation),
            BasicBlock2D(self.numFilters*8, self.numFilters*4, 3, 1, 1, batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4*5, self.numFilters*4, 3, 1, 1, batchnorm, activation),
            BasicBlock2D(self.numFilters*4, self.numFilters*2, 3, 1 ,1, batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer1 = nn.Sequential(
            BasicBlock2D(self.numFilters*2*5, self.numFilters*2, 3, 1, 1, batchnorm, activation),
            BasicBlock2D(self.numFilters*2, self.numFilters, 3, 1, 1, batchnorm, activation),
            nn.Conv2d(self.numFilters, self.numKeypoints, 1, 1, 0, bias=False),
        )

        A = torch.tensor([
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
        self.gcn = PRGCN(cfg, A)

        filterList = [self.numFilters*8, self.numFilters*4, self.numFilters*2]
        self.phi_cross_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_cross_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.phi_cross_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_cross_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.phi_self_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_self_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.phi_self_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_self_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.sigmoid = nn.Sigmoid()
    
    def attention(self, k, q, maps):
        b, c, h, w  = maps.size()
        k, q = k.view(b, c, h * w), q.view(b, c, h * w)
        spat_attn = torch.einsum('bij,bik->bjk', (k, q))
        maps = maps.view(b, c, h * w)
        maps = torch.einsum('bci,bik->bck', (maps, F.softmax(spat_attn, 1)))
        maps = maps.view(b, c, h, w)
        return maps

    def forward(self, ral1maps, ral2maps, ramaps, rel1maps, rel2maps, remaps):
        ramaps_res = ramaps
        remaps_res = remaps
        k3_c_hori = self.phi_cross_hori[0](ramaps)
        q3_c_vert = self.theta_cross_vert[0](remaps)
        k3_c_vert = self.phi_cross_vert[0](remaps)
        q3_c_hori = self.theta_cross_hori[0](ramaps)
        k3_hori = self.phi_self_hori[0](ramaps)
        q3_hori = self.theta_self_hori[0](ramaps)
        k3_vert = self.phi_self_vert[0](remaps)
        q3_vert = self.theta_self_vert[0](remaps)
        ramaps_cross = self.attention(k3_c_hori, q3_c_vert, ramaps) + ramaps_res
        ramaps_self = self.attention(k3_hori, q3_hori, ramaps)
        remaps_cross = self.attention(k3_c_vert, q3_c_hori, remaps) + remaps_res
        remaps_self = self.attention(k3_vert, q3_vert, remaps)
        maps = self.decoderLayer3(torch.cat((ramaps_cross, ramaps_self, remaps_cross, remaps_self), 1)) 

        ral2maps_res = ral2maps
        rel2maps_res = rel2maps
        k2_c_hori = self.phi_cross_hori[1](ral2maps)
        q2_c_vert = self.theta_cross_vert[1](rel2maps)
        k2_c_vert = self.phi_cross_vert[1](rel2maps)
        q2_c_hori = self.theta_cross_hori[1](ral2maps)
        k2_hori = self.phi_self_hori[1](ral2maps)
        q2_hori = self.theta_self_hori[1](ral2maps)
        k2_vert = self.phi_self_vert[1](rel2maps)
        q2_vert = self.theta_self_vert[1](rel2maps)
        ral2maps_cross = self.attention(k2_c_hori, q2_c_vert, ral2maps) + ral2maps_res
        ral2maps_self = self.attention(k2_hori, q2_hori, ral2maps)
        rel2maps_cross = self.attention(k2_c_vert, q2_c_hori, rel2maps) + rel2maps_res
        rel2maps_self = self.attention(k2_vert, q2_vert, rel2maps)
        maps = self.decoderLayer2(torch.cat((maps, ral2maps_cross, ral2maps_self, rel2maps_cross, rel2maps_self), 1)) 

        ral1maps_res = ral1maps
        rel1maps_res = rel1maps
        k1_c_hori = self.phi_cross_hori[2](ral1maps)
        q1_c_vert = self.theta_cross_vert[2](rel1maps)
        k1_c_vert = self.phi_cross_vert[2](rel1maps)
        q1_c_hori = self.theta_cross_hori[2](ral1maps)
        k1_hori = self.phi_self_hori[2](ral1maps)
        q1_hori = self.theta_self_hori[2](ral1maps)
        k1_vert = self.phi_self_vert[2](rel1maps)
        q1_vert = self.theta_self_vert[2](rel1maps)
        ral1maps_cross = self.attention(k1_c_hori, q1_c_vert, ral1maps) + ral1maps_res
        ral1maps_self = self.attention(k1_hori, q1_hori, ral1maps)
        rel1maps_cross = self.attention(k1_c_vert, q1_c_hori, rel1maps) + rel1maps_res
        rel1maps_self = self.attention(k1_vert, q1_vert, rel1maps)
        maps = self.decoderLayer1(torch.cat((maps, ral1maps_cross, ral1maps_self, rel1maps_cross, rel1maps_self), 1)) 
        gcn_output = self.gcn(maps)
        return maps, gcn_output

class Encoder3D(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(Encoder3D, self).__init__()
        #self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.layer1 = nn.Sequential(
            nn.Conv3d(self.numFilters, self.numFilters*2, 3, 1, 1),
            BasicBlock3D(self.numFilters*2, self.numFilters*2, 3, 1, 1),
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*2, self.numFilters*4, 3, 1, 1),
            BasicBlock3D(self.numFilters*4, self.numFilters*4, 3, 1, 1),
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True),
            BasicBlock3D(self.numFilters*4, self.numFilters*8, 3, 1, 1),
            BasicBlock3D(self.numFilters*8, self.numFilters*8, 3, 1, 1),
        )
        self.l1temporalMerge = nn.Conv3d(self.numFilters*2, self.numFilters*2, (self.numGroupFrames, 1, 1), 1, 0, bias=False)
        self.l2temporalMerge = nn.Conv3d(self.numFilters*4, self.numFilters*4, (self.numGroupFrames//2, 1, 1), 1, 0, bias=False)
        self.temporalMerge = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.numGroupFrames//4, 1, 1), 1, 0, bias=False)
    
    def forward(self, maps):
        l1maps = self.layer1(maps)
        l2maps = self.layer2(l1maps)
        l3maps = self.layer3(l2maps)
        maps = self.temporalMerge(l3maps).squeeze(2)
        return self.l1temporalMerge(l1maps).squeeze(2), self.l2temporalMerge(l2maps).squeeze(2), maps