import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class TemporalDecoder(nn.Module): #UNet + Dynamic filter
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(TemporalDecoder, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        #self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.decoderLayer1 = nn.Sequential(
            BasicBlock3D(self.numFilters*8*2, self.numFilters*8, 3, 1, 1, 1, batchnorm, activation),
            BasicBlock3D(self.numFilters*8, self.numFilters*4, 3, 1, 1, 1, batchnorm, activation),
            nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock3D(self.numFilters*4*2, self.numFilters*4, 3, 1, 1, 1, batchnorm, activation),
            BasicBlock3D(self.numFilters*4, self.numFilters*2, 3, 1 ,1, 1, batchnorm, activation),
            nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True),
        )
        # self.decoderLayer3 = nn.Sequential(
        #     BasicBlock3D(self.numFilters*2*2, self.numFilters*2, 3, 1, 1, 1, batchnorm, activation),
        #     BasicBlock3D(self.numFilters*2, self.numKeypoints, 3, 1, 1, 1, batchnorm, activation)
        # )
        self.decoderLayer3 = nn.Sequential(
            BasicBlock3D(self.numFilters*2*2, self.numFilters*2, 3, 1, 1, 1, batchnorm, activation)
        )
        self.finalLayer = nn.Sequential(
            BasicBlock3D(self.numFilters*2, self.numKeypoints, 3, 1, 1, 1, batchnorm, activation)
        )
        self.dfNet2 = nn.Sequential(
            BasicBlock3D(self.numFilters*4, self.numFilters*4, 3, 1, 1),
            BasicBlock3D(self.numFilters*4, 1, 3, 1, 1),
            #nn.Conv3d(self.numFilters*4, 1, 3, 2, 1, bias=False),
        )
        self.dfNet1 = nn.Sequential(
            BasicBlock3D(self.numFilters*2, self.numFilters*2, 3, 1, 1),
            BasicBlock3D(self.numFilters*2, 1, 3, 1, 1),
            #nn.Conv3d(self.numFilters*2, 1, 3, 2, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, hl1maps, hl2maps, vl1maps, vl2maps, maps):
        maps = self.decoderLayer1(maps)
        dfmaps2 = self.dfNet2(maps)
        hl2maps = F.interpolate(hl2maps, scale_factor=(0.5, 1, 1), mode='trilinear', align_corners=True)
        vl2maps = F.interpolate(vl2maps, scale_factor=(0.5, 1, 1), mode='trilinear', align_corners=True)
        #print(dfmaps2.size(), hl2maps.size(), maps.size())
        l2maps = dfmaps2 * (hl2maps + vl2maps)
        maps = self.decoderLayer2(torch.cat((maps, l2maps), 1))
        dfmaps1 = self.dfNet1(maps)
        hl1maps = F.interpolate(hl1maps, scale_factor=(0.25, 1, 1), mode='trilinear', align_corners=True)
        vl1maps = F.interpolate(vl1maps, scale_factor=(0.25, 1, 1), mode='trilinear', align_corners=True)
        #print(dfmaps1.size(), hl1maps.size(), maps.size())
        #hl1maps = dfmaps1 * hl1maps
        #vl1maps = dfmaps1 * vl1maps
        l1maps = dfmaps1 * (hl1maps + vl1maps)
        finalfeature = self.decoderLayer3(torch.cat((maps, l1maps), 1))
        output = self.finalLayer(finalfeature)
        #output = self.decoderLayer3(torch.cat((maps, l1maps), 1))
        return output, finalfeature


class TemporalDecoder2(nn.Module): #UNet + temporal + spatial attention
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(TemporalDecoder2, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames # for 60
        #self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.decoderLayer1 = nn.Sequential(
            BasicBlock3D(self.numFilters*8*2, self.numFilters*8, 3, 1, 1, 1, batchnorm, activation),
            BasicBlock3D(self.numFilters*8, self.numFilters*4, 3, 1, 1, 1, batchnorm, activation),
            nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock3D(self.numFilters*4*2, self.numFilters*4, 3, 1, 1, 1, batchnorm, activation),
            BasicBlock3D(self.numFilters*4, self.numFilters*2, 3, 1 ,1, 1, batchnorm, activation),
            nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True),
        )
        self.decoderLayer3 = nn.Sequential(
            BasicBlock3D(self.numFilters*2*2, self.numFilters*2, 3, 1, 1, 1, batchnorm, activation)
        )
        self.finalLayer = nn.Sequential(
            BasicBlock3D(self.numFilters*2, self.numKeypoints, 3, 1, 1, 1, batchnorm, activation)
        )
        self.phi2 = BasicBlock3D(self.numFilters*4, 1, 3, 1, 1)
        self.theta2 = BasicBlock3D(self.numFilters*4, 1, 3, 1, 1)
        self.phi1 = BasicBlock3D(self.numFilters*2, 1, 3, 1, 1)
        self.theta1 = BasicBlock3D(self.numFilters*2, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def attention(self, k, q, maps):
        k, q = k.squeeze(1), q.squeeze(1)
        b, c, f, h, w  = maps.size()
        k, q = k.view(b, f, h * w), q.view(b, f, h * w)
        # temporal attention
        temp_attn = torch.einsum('bij,bkj->bik', (k, q))
        maps = torch.einsum('bcihw,bik->bckhw', (maps, temp_attn))
        # spatial attention
        spat_attn = torch.einsum('bij,bik->bjk', (k, q))
        maps = maps.view(b, c, f, h * w)
        maps = torch.einsum('bcfi,bik->bcfk', (maps, spat_attn))
        maps = maps.view(b, c, f, h, w)
        return maps



    def forward(self, hl1maps, hl2maps, vl1maps, vl2maps, maps):
        # use decoder ouput as the input signal
        maps = self.decoderLayer1(maps)
        k2 = self.phi2(maps)
        q2 = self.theta2(maps)
        hl2maps = F.interpolate(hl2maps, scale_factor=(0.5, 1, 1), mode='trilinear', align_corners=True)
        vl2maps = F.interpolate(vl2maps, scale_factor=(0.5, 1, 1), mode='trilinear', align_corners=True)
        l2maps = self.attention(k2, q2, (hl2maps + vl2maps))
        maps = self.decoderLayer2(torch.cat((maps, l2maps), 1))
        k1 = self.phi1(maps)
        q1 = self.theta1(maps)
        hl1maps = F.interpolate(hl1maps, scale_factor=(0.25, 1, 1), mode='trilinear', align_corners=True)
        vl1maps = F.interpolate(vl1maps, scale_factor=(0.25, 1, 1), mode='trilinear', align_corners=True)
        l1maps = self.attention(k1, q1, (hl1maps + vl1maps))
        finalfeature = self.decoderLayer3(torch.cat((maps, l1maps), 1))
        output = self.finalLayer(finalfeature)
        return output, finalfeature
        # use hori/vert encoder intermediates as the input signal 


class RadarDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RadarDecoder, self).__init__()
        # self.main = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
        #     nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
        #     nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        # )
        self.main = nn.Sequential(
            BasicBlock2D(in_ch, out_ch, 3, 1, 1, batchnorm=False, activation=nn.PReLU),
            BasicBlock2D(out_ch, out_ch, 3, 1, 1, batchnorm=False, activation=nn.PReLU),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
    def forward(self, feat):
        return self.main(feat)


class RadarConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RadarConv, self).__init__()
        self.gconv3x3 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 2, 1, groups=2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            BasicBlock3D(out_ch, out_ch, 3, 1, 1, groups=2)
        )
        self.gconv5x5 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 5, 2, 2, groups=2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            BasicBlock3D(out_ch, out_ch, 5, 1, 2, groups=2)
        )
        self.conv3x3 = BasicBlock3D(out_ch, out_ch, 3, 1, 1)
        self.conv5x5 = BasicBlock3D(out_ch, out_ch, 5, 1, 2)
        # self.horiGCONV3x3 = nn.Sequential(
        #     nn.Conv3d(in_ch, out_ch, 3, 2, 1, groups=2, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.vertGCONV3x3 = nn.Sequential(
        #     nn.Conv3d(in_ch, out_ch, 3, 2, 1, groups=2, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.horiGCONV5x5 = nn.Sequential(
        #     nn.Conv3d(in_ch, out_ch, 5, 2, 2, groups=2, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.vertGCONV5x5 = nn.Sequential(
        #     nn.Conv3d(in_ch, out_ch, 5, 2, 2, groups=2, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.realCONV3x3 = nn.Sequential(
        #     nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.realCONV5x5 = nn.Sequential(
        #     nn.Conv3d(out_ch, out_ch, 5, 1, 2, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.imagCONV3x3 = nn.Sequential(
        #     nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.imagCONV5x5 = nn.Sequential(
        #     nn.Conv3d(out_ch, out_ch, 5, 1, 2, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.realCONV1x1 = nn.Sequential(
        #     nn.Conv3d(out_ch*2, out_ch, 1, 1, 0, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        # self.imagCONV1x1 = nn.Sequential(
        #     nn.Conv3d(out_ch*2, out_ch, 1, 1, 0, bias=False),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU()
        # )
        self.half = out_ch//2
    def forward(self, hori, vert):
        hori3x3 = self.gconv3x3(hori)
        vert3x3 = self.gconv3x3(vert)
        hori5x5 = self.gconv5x5(hori)
        vert5x5 = self.gconv5x5(vert)
        real3x3 = torch.cat((hori3x3[:, :self.half, :, :, :], vert3x3[:, :self.half, :, :, :]), 1)
        real5x5 = torch.cat((hori5x5[:, :self.half, :, :, :], vert5x5[:, :self.half, :, :, :]), 1)
        imag3x3 = torch.cat((hori3x3[:, self.half:, :, :, :], vert3x3[:, self.half:, :, :, :]), 1)
        imag5x5 = torch.cat((hori5x5[:, self.half:, :, :, :], vert5x5[:, self.half:, :, :, :]), 1)
        real3x3 = self.conv3x3(real3x3)
        imag3x3 = self.conv3x3(imag3x3)
        real5x5 = self.conv5x5(real5x5)
        imag5x5 = self.conv5x5(imag5x5)
        feat3x3 = torch.cat((real3x3, imag3x3), dim=1)
        feat5x5 = torch.cat((real5x5, imag5x5), dim=1)
        return feat3x3, feat5x5

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, batchnorm=True, activation=nn.ReLU):
        super(BasicBlock3D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm3d(out_channels),
                activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, groups=groups, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            )
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, groups=groups, bias=False),
            )
        self.relu = activation()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.main(x) + residual
        out = self.relu(out)
        return out