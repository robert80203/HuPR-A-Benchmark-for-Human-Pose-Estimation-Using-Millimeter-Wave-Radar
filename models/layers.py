import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ResidualBlock2D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.residualConv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        residual = self.residualConv1x1(x)
        x = self.main(x) + residual
        return x

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

class Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True, activation=nn.ReLU):
        super(Block3D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(out_channels),
                activation(),
            )
        else:   
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
            )
    def forward(self, x):
        return self.main(x)

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