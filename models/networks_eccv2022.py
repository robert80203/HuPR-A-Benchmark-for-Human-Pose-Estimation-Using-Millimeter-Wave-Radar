import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_networks import TemporalModel6, TemporalModel7
from models.chirp_networks import *
from models.eccv2022.gcn_networks import STGCN, STGCN2, STGCN3, GCN
from models.eccv2022.layers import RadarDecoder, RadarConv, TemporalDecoder, TemporalDecoder2, get_embedder
from models.eccv2022.hrnet import HRNet
from models.layers import BasicDecoder
from misc.utils import generateTarget, soft_argmax, get_coords_using_integral, point_sample
from misc.metrics import get_max_preds

BN_MOMENTUM = 0.1

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
        #self.mode = cfg.DATASET.mode
        #self.useVelocity = cfg.DATASET.useVelocity
        self.numFilters = cfg.MODEL.numFilters
        self.frontModel = cfg.MODEL.frontModel
        self.backbone = cfg.MODEL.backbone
        self.chirpModel = cfg.MODEL.chirpModel
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType
        self.tempinfo = self.numGroupFrames // 4 // 2 #self.numFrames * self.numGroupFrames // 4 // 2

        ###################################
        #model for preprocessing
        ###################################
        if self.chirpModel == 'maxpool':
            self.mnet = None
        elif self.chirpModel == 'mnet':
            self.mnet = MNet(self.numFrames, self.numFilters)
        elif self.chirpModel == 'identity':
            self.mnet = Identity()
        else:
            raise RuntimeError(F"chirpModel should be maxpool/mnet, not {self.chirpModel}.")
        
        ###################################
        #main model architecture
        ###################################
        if self.frontModel == 'stgcn':
            self.temporalHoriNet = TemporalModel6(cfg, batchnorm=False, activation=nn.PReLU)
            self.temporalVertNet = TemporalModel6(cfg, batchnorm=False, activation=nn.PReLU)
            self.fusedNet = STGCN(cfg, batchnorm=False, activation=nn.PReLU)
        elif self.frontModel == 'stgcn2_stage1':
            self.temporalHoriNet = TemporalModel7(cfg, fuseLast=True)
            self.temporalVertNet = TemporalModel7(cfg, fuseLast=True)
            self.fusedNet = BasicDecoder(cfg, batchnorm=False, activation=nn.PReLU)
        elif self.frontModel == 'stunet_df':
            self.temporalHoriNet = TemporalModel7(cfg, fuseLast=False)
            self.temporalVertNet = TemporalModel7(cfg, fuseLast=False)
            if self.backbone == 'temp2':
                self.fusedNet = TemporalDecoder2(cfg, batchnorm=False, activation=nn.PReLU)
            else:
                self.fusedNet = TemporalDecoder(cfg, batchnorm=False, activation=nn.PReLU)
            if 'stgcn2' in self.gcnType:
                self.gcn = STGCN2(cfg, batchnorm=False, activation=nn.PReLU)
            if 'keyfeat' in self.gcnType:
                self.embed, _ = get_embedder(10)
        elif self.frontModel == 'hrnet':
            self.hrnet = HRNet(cfg)
            if self.gcnType == 'gcn':
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
                self.gcn = GCN(cfg, A)
        else:
            raise RuntimeError(F"activation should be stgcn, not {self.frontModel}.")

    def forward(self, horiMap, vertMap):

        batchSize = horiMap.size(0)
        #if self.frontModel == 'stunet_df':
        #    horiMap = horiMap.view(batchSize, 2, self.numGroupFrames * self.numFrames, self.rangeSize, self.azimuthSize)
        #    vertMap = vertMap.view(batchSize, 2, self.numGroupFrames * self.numFrames, self.rangeSize, self.azimuthSize)
        #else:
        horiMap = horiMap.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize)
        vertMap = vertMap.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize)
        horiMap = self.mnet(horiMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
        vertMap = self.mnet(vertMap).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize)
        horiMap = horiMap.permute(0, 2, 1, 3, 4) # (b, channel=2, # of frames=30, 64, 8)
        vertMap = vertMap.permute(0, 2, 1, 3, 4)
        
        if self.frontModel == 'stgcn':
            hl1maps, hl2maps, hl3maps = self.temporalHoriNet(horiMap)
            vl1maps, vl2maps, vl3maps = self.temporalVertNet(vertMap)
            l1maps = torch.cat((hl1maps, vl1maps), dim=1)
            l2maps = torch.cat((hl2maps, vl2maps), dim=1)
            l3maps = torch.cat((hl3maps, vl3maps), dim=1)
            heatmap = self.fusedNet(l1maps, l2maps, l3maps)
        elif self.frontModel == 'stgcn2_stage1':
            hl1maps, hl2maps, hl3maps = self.temporalHoriNet(horiMap)
            vl1maps, vl2maps, vl3maps = self.temporalVertNet(vertMap)
            l3maps = torch.cat((hl3maps, vl3maps), dim=1)
            heatmap, _ = self.fusedNet(l3maps.squeeze(2))
        elif self.frontModel == 'stunet_df':
            hl1maps, hl2maps, hl3maps = self.temporalHoriNet(horiMap)
            vl1maps, vl2maps, vl3maps = self.temporalVertNet(vertMap)
            fusedMap = torch.cat((hl3maps, vl3maps), 1)
            fusedMap, finalfeature = self.fusedNet(hl1maps, hl2maps, vl1maps, vl2maps, fusedMap)
            
            if 'stgcn' in self.gcnType:
                if 'conf' in self.gcnType:
                    gcninput = F.interpolate(fusedMap, scale_factor=(1.0, 0.5, 0.5), mode='trilinear', align_corners=True)
                    gcninput = gcninput.view(batchSize, self.numKeypoints, self.numGroupFrames//4, -1).permute(0, 3, 2, 1)
                    gcnoutput = self.gcn(gcninput)
                    gcnoutput = gcnoutput.view(batchSize, self.numKeypoints, self.height//2, self.width//2)
                    gcnoutput = F.interpolate(gcnoutput, scale_factor=2.0, mode='bilinear', align_corners=True)
                elif 'coord' in self.gcnType:
                    #gcninput = soft_argmax(fusedMap)
                    for i in range(fusedMap.size(2)):
                        if i == 0:
                            gcninput = get_coords_using_integral(torch.sigmoid(fusedMap[:,:,i,:,:])).unsqueeze(2)/self.height
                        else:
                            gcninput = torch.cat((gcninput, get_coords_using_integral(torch.sigmoid(fusedMap[:,:,i,:,:])).unsqueeze(2)/self.height), 2)
                    gcnoutput = self.gcn(gcninput.permute(0, 3, 2, 1)) # output range: 0~1
                elif 'keyfeat' in self.gcnType:
                    for i in range(fusedMap.size(2)):
                        coord, _, = get_max_preds(torch.sigmoid(fusedMap[:,:,i,:,:]).detach().cpu().numpy())
                        coord = torch.FloatTensor(coord).cuda()
                        embedding = self.embed(coord)
                        coord[:, :, 0] = coord[:, :, 0] / self.width - 1
                        coord[:, :, 1] = coord[:, :, 1] / self.height - 1
                        feat = point_sample(finalfeature[:,:,i,:,:], coord)
                        feat = torch.cat((feat, embedding.permute(0, 2, 1)), 1)
                        if i == 0:
                            gcninput = feat.unsqueeze(2)
                        else:
                            gcninput = torch.cat((gcninput, feat.unsqueeze(2)), 2)
                    gcnoutput = self.gcn(gcninput)

            #heatmap = (fusedMap[:, :, self.tempinfo-1, :, :]+fusedMap[:, :, self.tempinfo, :, :])/2
            temp = fusedMap[:, :, self.tempinfo-1, :, :]
            heatmap = torch.sigmoid(temp).unsqueeze(2)
        elif self.frontModel == 'hrnet':
            heatmap = self.hrnet(torch.cat((horiMap, vertMap), 1))
            if 'gcn' in self.gcnType:
                gcnoutput = self.gcn(heatmap)
            heatmap = torch.sigmoid(heatmap).unsqueeze(2)
        if self.modelType == 'heatmap':
            return heatmap
        elif self.modelType == 'heatmap_heatmap':
            return heatmap, torch.sigmoid(gcnoutput).unsqueeze(1)
        elif self.modelType == 'heatmap_regress' or self.modelType == 'heatmap_refine':
            #gcnoutput = get_coords_using_integral(torch.sigmoid(temp))/self.height
            return heatmap, gcnoutput
        #return heatmap


class RefineModel(nn.Module):
    def __init__(self, cfg):
        super(RefineModel, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.numFilters = cfg.MODEL.numFilters
        self.gcnType = cfg.MODEL.gcnType
        self.hsize = cfg.DATASET.heatmapSize
        self.isize = cfg.DATASET.imgSize
        if 'stgcn2' in cfg.MODEL.frontModel:
            self.refineNet = STGCN2(cfg)
        elif 'stgcn3' in cfg.MODEL.frontModel:
            self.refineNet = STGCN3(cfg)
        elif 'conv2d' in cfg.MODEL.frontModel:
            self.refineNet = Convolution2D(cfg)
        else:
            raise RuntimeError(F"activation should contain stgcn2/stgcn3, not {cfg.MODEL.frontModel}.")

    def forward(self, x, sigmas=None):
        if self.gcnType == 'conf':
            x = (x + 1)*(self.hsize/2)
            for i in range(x.size(0)):
                for j in range(x.size(2)):
                    out, _ = generateTarget(x[i, :, j, :].permute(1, 0), self.numKeypoints, self.hsize, self.isize, sigmas=sigmas[i, j].cpu().numpy())
                    out = torch.from_numpy(out).cuda().unsqueeze(0)
                    out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=True)
                    if j == 0:
                        tempf = out
                    else:
                        tempf = torch.cat((tempf, out), dim=0)
                if i == 0:
                    heatmap = tempf.unsqueeze(0)
                else:
                    heatmap = torch.cat((heatmap, tempf.unsqueeze(0)), 0)
                
            heatmap = heatmap.permute(0, 3, 4, 1, 2).contiguous()
            heatmap = heatmap.view(x.size(0), -1, self.numGroupFrames, self.numKeypoints)
            refinemap = self.refineNet(heatmap)
            refinemap = refinemap.reshape(-1, 32, 32, self.numKeypoints)
            refinemap = refinemap.permute(0, 3, 1, 2)
            refinemap = F.interpolate(refinemap, scale_factor=2.0, mode='bilinear', align_corners=True)
            return torch.sigmoid(refinemap).unsqueeze(2)
        elif self.gcnType == 'coord_var':
            x = torch.cat((x, sigmas.unsqueeze(1) * 10), dim=1)
            return self.refineNet(x)
        else:
            return self.refineNet(x)

class RadarModel(nn.Module):
    def __init__(self, cfg):
        super(RadarModel, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numChirps = cfg.DATASET.numFrames
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.gcnType = cfg.MODEL.gcnType
        self.modelType = cfg.MODEL.type
        self.layer1 = RadarConv(self.numChirps*2, self.numFilters)
        self.layer2 = RadarConv(self.numFilters*2, self.numFilters*4)
        self.tempinfo = ((self.numGroupFrames+1)//2+1)//2
        self.tempfuse = nn.Conv3d(self.numFilters*8, self.numFilters*8, (self.tempinfo, 1, 1), 1, 0, bias=False)
        self.decode1 = RadarDecoder(self.numFilters*8*2, self.numFilters*4)
        self.decode2 = RadarDecoder(self.numFilters*4, self.numKeypoints)
        if 'gcn' in self.gcnType:
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
            self.gcn = GCN(cfg, A)

    def forward(self, horiMap, vertMap):
        b = horiMap.size(0)#(b, # of frames, 1, # of channels, h, w)
        horireal = horiMap[:, :, :, 0, :, :].view(b, -1, self.numGroupFrames, self.height, self.width)
        horiimag = horiMap[:, :, :, 1, :, :].view(b, -1, self.numGroupFrames, self.height, self.width)
        hori = torch.cat((horireal, horiimag), 1)
        vertreal = vertMap[:, :, :, 0, :, :].view(b, -1, self.numGroupFrames, self.height, self.width)
        vertimag = vertMap[:, :, :, 1, :, :].view(b, -1, self.numGroupFrames, self.height, self.width)
        vert = torch.cat((vertreal, vertimag), 1)
        feat3x3, feat5x5 = self.layer1(hori, vert)
        feat3x3, feat5x5 = self.layer2(feat3x3, feat5x5)
        feat3x3 = self.tempfuse(feat3x3).squeeze(2)
        feat5x5 = self.tempfuse(feat5x5).squeeze(2)
        feat = torch.cat((feat3x3, feat5x5), 1)
        feat = self.decode1(feat)
        output = self.decode2(feat)
        if 'gcn' in self.gcnType:
            gcnoutput = self.gcn(output)
        
        if self.modelType == 'heatmap':
            return torch.sigmoid(output).unsqueeze(2)
        elif self.modelType == 'heatmap_heatmap':
            return torch.sigmoid(output).unsqueeze(2), gcnoutput

# class HRNet(nn.Module):
#     def __init__(self, cfg):
#         super(HRNet, self).__init__()
#         self.numFrames = cfg.DATASET.numFrames
#         self.numGroupFrames = cfg.DATASET.numGroupFrames
#         self.numKeypoints = cfg.DATASET.numKeypoints
#         self.rangeSize = cfg.DATASET.rangeSize
#         self.azimuthSize = cfg.DATASET.azimuthSize
#         self.width = cfg.DATASET.heatmapSize
#         self.height = cfg.DATASET.heatmapSize
#         self.numFilters = cfg.MODEL.numFilters
#         self.frontModel = cfg.MODEL.frontModel
#         self.backbone = cfg.MODEL.backbone
#         self.gcnType = cfg.MODEL.gcnType
#         self.tempinfo = ((self.numGroupFrames+1)//2+1)//2
#         self.preconv = nn.Sequential(
#             nn.Conv3d(self.numFilters*2 if cfg.MODEL.chirpModel == 'mnet' else 2 * 2, self.numFilters, 3, (2, 1, 1) , 1, bias=False),
#             nn.BatchNorm3d(self.numFilters),
#             nn.ReLU(),
#             nn.Conv3d(self.numFilters, self.numFilters, 3, (2, 1, 1) , 1, bias=False),
#             nn.BatchNorm3d(self.numFilters),
#             nn.ReLU()
#         )
#         self.tempfuse = nn.Conv3d(self.numFilters, self.numFilters, 1, (self.tempinfo, 1, 1) , 0, bias=False)
#         self.numLayers = 2 # at least 2
#         self.numStages = 3
#         self.stage = [[nn.ModuleList() for i in range(self.numStages)],
#                       [nn.ModuleList() for i in range(self.numStages-1)],
#                       [nn.ModuleList() for i in range(self.numStages-2)]]
#         self.transition = [[nn.ModuleList() for i in range(self.numStages-1)],
#                            [nn.ModuleList() for i in range(self.numStages-2)],
#                            [None]]
        
#         channellist = [
#             [[numFilters, numFilters*2], [numFilters*2, numFilters*2]],
#             [[numFilters*2, numFilters*4], [numFilters*4, numFilters*4]],
#             [[numFilters*4, numFilters*8], [numFilters*8, numFilters*8]],
#         ]
        
#         for k in range(self.numStages):
#             for i in range(self.numStages-k):
#                 for j in range(self.numLayers):
#                     self.stage[i].append(
#                         nn.Conv2d(channellist[i][j][0], channellist[i][j][1], 3, 1, 1, bias=False),
#                         nn.BatchNorm2d(channellist[i][j][1]),
#                         nn.ReLU(),
#                     )
#                 if i < self.numStages-1:
#                     self.transition[i].append(
#                         nn.Conv2d(channellist[i][-1][0], channellist[i][-1][1], 3, (2, 1), 1, bias=False),
#                         nn.BatchNorm2d(channellist[i][-1][1]),
#                         nn.ReLU()
#                     )

#     def forward(self, x):
#         tempfeat = self.preconv(x)
#         x = self.tempfuse(tempfeat)

#         x_list = []
#         for i in range(self.numStages):
#             x = self.stage[0][i](x)
#             if i < self.numStages -1 :
#                 x_list.append(self.transition[0][i](x))


class Convolution2D(nn.Module):
    def __init__(self, cfg):
        super(Convolution2D, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.main = nn.Sequential(
            nn.Conv2d(2, self.numFilters*2, (3, 3), (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(self.numFilters*2),
            nn.ReLU(),
            nn.Conv2d(self.numFilters*2, self.numFilters*4, (3, 3), (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(self.numFilters*4),
            nn.ReLU(),
            nn.Conv2d(self.numFilters*4, self.numFilters*8, (3, 3), (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(self.numFilters*8),
            nn.ReLU(),
        )
        self.temporalMerge = nn.Conv2d(self.numFilters*8, 2, (self.numGroupFrames//8, 1), 1, 0, bias=False)
    def forward(self, x):
        x = self.main(x)
        x = self.temporalMerge(x).squeeze(2).permute(0, 2, 1)
        return x