import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ResidualBlock3D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.residualConv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0)),
            nn.BatchNorm3d(out_channels),
        )
    def forward(self, x):
        residual = self.residualConv1x1(x)
        x = self.main(x) + residual
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