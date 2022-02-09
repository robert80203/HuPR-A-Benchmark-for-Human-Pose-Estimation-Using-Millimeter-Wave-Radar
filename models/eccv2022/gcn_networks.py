import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.layers import BasicBlock3D, BasicBlock2D

class GCN_layers(nn.Module):
    def __init__(self, in_features, out_features, numKeypoints, bias=True):
        super(GCN_layers, self).__init__()
        self.bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, numKeypoints))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, adj)
        output = torch.matmul(self.weight, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN_layers_res(nn.Module):
    def __init__(self, in_features, out_features, numKeypoints, bias=True):
        super(GCN_layers_res, self).__init__()
        self.bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, numKeypoints), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.mask = nn.Parameter(nn.Parameter(torch.randn((numKeypoints, numKeypoints)), requires_grad=True))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        adj = adj + self.mask
        support = torch.matmul(input, adj)
        output = torch.matmul(self.weight, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim, joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous() 



class GCN(nn.Module): # output: heatmap
    def __init__(self, cfg, A):
        super(GCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType
        self.featureSize = (self.height//2) * (self.width//2)
        self.L1 = GCN_layers_res(self.featureSize, self.featureSize, self.numKeypoints)
        self.L2 = GCN_layers_res(self.featureSize, self.featureSize, self.numKeypoints)
        self.L3 = GCN_layers_res(self.featureSize,self.featureSize, self.numKeypoints)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if 'trainA'in self.gcnType:
            self.A = nn.Parameter(A, requires_grad=True)
        else:
            self.A = A

    def generate_node_feature(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = x.reshape(-1, self.numKeypoints, self.featureSize).permute(0, 2, 1)
        return x
    
    def gcn_forward(self, x):
        #x: (B, numFilters, numkeypoints)
        x2 = self.relu(self.L1(x, self.A))
        # if self.training:
        #     x2 = F.dropout(x2, p=0.1)
        x3 = self.relu(self.L2(x2, self.A))
        # if self.training:
        #     x3 = F.dropout(x3, p=0.1)
        keypoints = self.L3(x3, self.A)
        return keypoints.permute(0, 2, 1)
    
    def forward(self, x):
        nodeFeat = self.generate_node_feature(x)
        heatmap = self.gcn_forward(nodeFeat).reshape(-1, self.numKeypoints, (self.height//2), (self.width//2))
        heatmap = F.interpolate(heatmap, scale_factor=2.0, mode='bilinear', align_corners=True)
        return self.sigmoid(heatmap).unsqueeze(2)

class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        
        self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer
        
        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )       
        
        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        return x


class STGCN(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(STGCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.nFrameIdx = [-(self.numGroupFrames//-2), 
                                  -(self.numGroupFrames//-4), 
                                  -(self.numGroupFrames//-8)]
        self.toHeatmap1 = BasicBlock3D(self.numFilters * 2 * 2, self.numKeypoints, 1, 1, 0)
        self.toHeatmap2 = BasicBlock3D(self.numFilters * 4 * 2, self.numKeypoints, 1, 1, 0)
        self.toHeatmap3 = BasicBlock3D(self.numFilters * 8 * 2, self.numKeypoints, 1, 1, 0)
        self.toHeatmapFinal = nn.Conv3d(sum(self.nFrameIdx), 1,  1, 1, 0, bias=False)

        self.stgcnl1 = ST_GCNN_layer(in_channels=64 * 64,
                                     out_channels=32 * 32,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     time_dim=self.nFrameIdx[0],
                                     joints_dim=self.numKeypoints,
                                     dropout=0.2)
        self.stgcnl2 = ST_GCNN_layer(in_channels=32 * 32,
                                     out_channels=32 * 32,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     time_dim=self.nFrameIdx[1],
                                     joints_dim=self.numKeypoints,
                                     dropout=0.2)
        self.stgcnl3 = ST_GCNN_layer(in_channels=16 * 16,
                                     out_channels=32 * 32,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     time_dim=self.nFrameIdx[2],
                                     joints_dim=self.numKeypoints,
                                     dropout=0.2)

    def forward(self, l1maps, l2maps, l3maps):
        gcninput1 = self.toHeatmap1(l1maps)
        gcninput1 = gcninput1.permute(0, 3, 4, 2, 1).view(-1, 64 * 64, self.nFrameIdx[0], self.numKeypoints)
        stmaps1 = self.stgcnl1(gcninput1).permute(0, 2, 3, 1).view(-1, self.nFrameIdx[0], self.numKeypoints, 32, 32)
        
        gcninput2 = self.toHeatmap2(l2maps)
        gcninput2 = gcninput2.permute(0, 3, 4, 2, 1).view(-1, 32 * 32, self.nFrameIdx[1], self.numKeypoints)
        stmaps2 = self.stgcnl2(gcninput2).permute(0, 2, 3, 1).view(-1, self.nFrameIdx[1], self.numKeypoints, 32, 32)


        gcninput3 = self.toHeatmap3(l3maps)
        gcninput3 = gcninput3.permute(0, 3, 4, 2, 1).view(-1, 16 * 16, self.nFrameIdx[2], self.numKeypoints)
        stmaps3 = self.stgcnl3(gcninput3).permute(0, 2, 3, 1).view(-1, self.nFrameIdx[2], self.numKeypoints, 32, 32)

        output = self.toHeatmapFinal(torch.cat((stmaps1, stmaps2, stmaps3), dim=1)).squeeze(1)
        output = F.interpolate(output, scale_factor=2.0)
    
        if self.modelType == "heatmapMSE":
            return output.unsqueeze(2)
        elif self.modelType == "heatmap_regress" or self.modelType == "heatmap_heatmap":
            return self.sigmoid(output).unsqueeze(2), gcnoutput
        elif self.modelType == "justregress":
            return gcnoutput
        else:
            return torch.sigmoid(output).unsqueeze(2)

class STGCN2(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(STGCN2, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numGroupFrames = self.numGroupFrames//4
        self.numFilters = cfg.MODEL.numFilters
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.stgcn = nn.ModuleList()
        # if self.gcnType == 'coord':
        #     self.mergeTemporal = nn.Conv2d(2, 2, (self.numGroupFrames//8, 1), 1, 0, bias=False)
        # elif self.gcnType == 'conf':
        #     self.mergeTemporal = nn.Conv2d(32 * 32, 32 * 32, (self.numGroupFrames//8, 1), 1, 0, bias=False)
        # else:
        #     raise RuntimeError(F"gcnType should be coord/conf, not {self.gcnType}.")
        # self.in_out_T_channels = [[2 if self.gcnType == 'coord' else (32 * 32), self.numFilters * 4, self.numGroupFrames],
        #                         [self.numFilters * 4, self.numFilters * 8, self.numGroupFrames//2],
        #                         [self.numFilters * 8, 2 if self.gcnType == 'coord' else (32 * 32), self.numGroupFrames//4]]
        if 'coord' in self.gcnType :
            in_ch = 2
            out_ch = 2
        elif 'conf' in self.gcnType:
            in_ch = 32 * 32
            out_ch = 32 * 32
        elif 'coord_var' in self.gcnType:
            in_ch = 3
            out_ch = 2
        elif 'keyfeat' in self.gcnType:
            in_ch = self.numFilters*2+42
            out_ch = 2
        else:
            raise RuntimeError(F"gcnType should be coord/conf/coord_var, not {self.gcnType}.")
        
        tempinfo = self.numGroupFrames
        self.in_out_T_channels = [[in_ch, self.numFilters * 4, 0],
                                [self.numFilters * 4, self.numFilters * 8, 0],
                                [self.numFilters * 8, out_ch, 0]]
        #self.mergeTemporal = nn.Conv2d(out_ch, out_ch, (1 if self.numGroupFrames < 8 else self.numGroupFrames//8, 1), 1, 0, bias=False)
        for i in range(len(self.in_out_T_channels)):
            self.in_out_T_channels[i][2] = tempinfo
            tempinfo = (tempinfo + 1) // 2
            self.stgcn.append(nn.Sequential(
                ST_GCNN_layer(in_channels=self.in_out_T_channels[i][0],
                                        out_channels=self.in_out_T_channels[i][1],
                                        kernel_size=(3, 3),
                                        stride=1,
                                        time_dim=self.in_out_T_channels[i][2],
                                        joints_dim=self.numKeypoints,
                                        dropout=0.2),
                nn.Conv2d(self.in_out_T_channels[i][1], self.in_out_T_channels[i][1], (3, 1), (2, 1), (1 ,0), bias=False),
                nn.BatchNorm2d(self.in_out_T_channels[i][1])))
                #nn.ReLU()))
        self.mergeTemporal = nn.Conv2d(out_ch, out_ch, (tempinfo, 1), 1, 0, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        #x shape: (B, C, T, V)
        for i in range(len(self.in_out_T_channels)):
            x = self.stgcn[i](x)
            if i+1 < len(self.in_out_T_channels):
                x = self.relu(x)
        x = self.mergeTemporal(x).squeeze(2).permute(0, 2, 1)
        return x
        # if self.modelType == "justregress":
        #     return x
        # else:
        #     return torch.sigmoid(x)

# all stgcn, with variance as input
class STGCN3(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(STGCN3, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType

        self.stgcn = nn.ModuleList()
        if self.gcnType == 'coord':
            in_ch = 2
            out_ch = 2
        elif self.gcnType == 'conf':
            in_ch = 32 * 32
            out_ch = 32 * 32
        elif self.gcnType == 'coord_var':
            in_ch = 3
            out_ch = 2
        else:
            raise RuntimeError(F"gcnType should be coord/conf, not {self.gcnType}.")
        #self.mergeTemporal = nn.Conv2d(in_ch, out_ch, (self.numGroupFrames//8, 1), 1, 0, bias=False)
        self.in_out_T_channels = [[in_ch, self.numFilters * 4, self.numGroupFrames],
                                [self.numFilters * 4, self.numFilters * 4, self.numGroupFrames//2],
                                [self.numFilters * 4, self.numFilters * 8, self.numGroupFrames//4],
                                [self.numFilters * 8, out_ch, self.numGroupFrames//8]]
        
        for i in range(len(self.in_out_T_channels)):
            self.stgcn.append(nn.Sequential(
                ST_GCNN_layer(in_channels=self.in_out_T_channels[i][0],
                                        out_channels=self.in_out_T_channels[i][1],
                                        kernel_size=(3, 3),
                                        stride=1,
                                        time_dim=self.in_out_T_channels[i][2],
                                        joints_dim=self.numKeypoints,
                                        dropout=0.2),
                nn.Conv2d(self.in_out_T_channels[i][1], self.in_out_T_channels[i][1], (3, 1), (2, 1), (1 ,0), bias=False)
                if i < (len(self.in_out_T_channels)-1) else
                nn.Conv2d(self.in_out_T_channels[i][1], self.in_out_T_channels[i][1], (self.numGroupFrames//8, 1), (1, 1), (0 ,0), bias=False),
                nn.BatchNorm2d(self.in_out_T_channels[i][1]),
                nn.ReLU())
                )

    def forward(self, x):
        #x shape: (B, C, T, V)
        for i in range(len(self.in_out_T_channels)):
            x = self.stgcn[i](x)
        x = x.squeeze(2).permute(0, 2, 1)
        return x