import torch
import torch.nn as nn
import torch.nn.functional as F
from models.chirp_networks import MNet
from models.layers import Encoder3D, MultiScaleCrossSelfAttentionPRGCN

class HuPRNet(nn.Module):
    def __init__(self, cfg):
        super(HuPRNet, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.rangeSize = cfg.DATASET.rangeSize
        self.heatmapSize = cfg.DATASET.heatmapSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.elevationSize = cfg.DATASET.elevationSize
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.RAchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.REchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.RAradarEncoder = Encoder3D(cfg)
        self.REradarEncoder = Encoder3D(cfg)
        self.radarDecoder = MultiScaleCrossSelfAttentionPRGCN(cfg, batchnorm=False, activation=nn.PReLU)

    def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
        batchSize = VRDAEmaps_hori.size(0)
        # Shrink elevation dimension
        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
        VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

        RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        RAmaps = RAmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        REmaps = self.REchirpNet(VRDAmaps_vert.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        REmaps = REmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        return RAmaps, REmaps
    
    def forward(self, VRDAEmaps_hori, VRDAEmaps_vert):
        RAmaps, REmaps = self.forward_chirp(VRDAEmaps_hori, VRDAEmaps_vert)
        RAl1feat, RAl2feat, RAfeat = self.RAradarEncoder(RAmaps)
        REl1feat, REl2feat, REfeat = self.REradarEncoder(REmaps)
        output, gcn_heatmap = self.radarDecoder(RAl1feat, RAl2feat, RAfeat, REl1feat, REl2feat, REfeat)
        heatmap = torch.sigmoid(output).unsqueeze(2)
        return heatmap, gcn_heatmap