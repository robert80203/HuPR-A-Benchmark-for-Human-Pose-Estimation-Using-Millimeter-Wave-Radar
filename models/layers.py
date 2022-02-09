import torch
import torch.nn as nn
from models.gcn_networks import *

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

class Block2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True, activation=nn.ReLU):
        super(Block2D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                activation(),
            )
        else:   
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
            )
    def forward(self, x):
        return self.main(x)

class DeconvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation=nn.PReLU):
        super(DeconvBlock2D, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            activation(),
        )
    def forward(self, x):
        return self.main(x)

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BasicConv3d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

#same naive inception module
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#same naive inception module
class InceptionA3d(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv3d(input_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv3d(input_channels, 48, kernel_size=1),
            BasicConv3d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv3d(input_channels, 64, kernel_size=1),
            BasicConv3d(64, 96, kernel_size=3, padding=1),
            BasicConv3d(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            BasicConv3d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)


class BasicDecoder(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU, backbone=""):
        super(BasicDecoder, self).__init__()
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.numFilters = cfg.MODEL.numFilters
        if "deeper" in backbone:
            self.decoder = nn.Sequential(
                BasicBlock2D(self.numFilters*2*16, self.numFilters*2*16, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2*16, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*8, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
            )
        else:
            self.decoder = nn.Sequential(
                BasicBlock2D(self.numFilters*2*8, self.numFilters*2*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
            )

        self.sigmoid = nn.Sigmoid()
    def forward(self, fusedMap):
        return self.sigmoid(self.decoder(fusedMap)).unsqueeze(2)


class BasicDecoderGCN(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU, backbone=""):
        super(BasicDecoderGCN, self).__init__()
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.numFilters = cfg.MODEL.numFilters
        self.gcnType = cfg.MODEL.gcnType
        if "deeper" in backbone:
            self.decoder = nn.Sequential(
                BasicBlock2D(self.numFilters*2*16, self.numFilters*2*16, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2*16, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*8, self.numFilters*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2, self.numKeypoints, (3, 3), (1, 1), (1, 1), batchnorm, activation)
            )
        else:
            self.decoder = nn.Sequential(
                BasicBlock2D(self.numFilters*2*8, self.numFilters*2*8, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*2*8, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*4, self.numFilters*4, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                BasicBlock2D(self.numFilters*4, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                BasicBlock2D(self.numFilters*2, self.numFilters*2, (3, 3), (1, 1), (1, 1), batchnorm, activation),
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
    
    def forward(self, fusedMap):
        output = self.decoder(fusedMap)
        gcnoutput = self.gcn(output)
        return self.sigmoid(output).unsqueeze(2), gcnoutput



if __name__ == '__main__':
    layer1 = InceptionA3d(192, pool_features=32)
    layer2 = InceptionA3d(256, pool_features=64)
    #layer3 = InceptionA(288, pool_features=64)
    x = torch.randn((2, 192, 5, 32, 32))
    out = layer1(x)
    out = layer2(out)