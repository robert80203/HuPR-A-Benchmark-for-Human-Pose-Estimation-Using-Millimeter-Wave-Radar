import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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

class GCN1(nn.Module): # output: heatmap
    def __init__(self, cfg, A):
        super(GCN1, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType
        self.featureSize = (self.height//2) * (self.width//2)
        self.L1 = GCN_layers(self.featureSize, self.featureSize, self.numKeypoints)
        self.L2 = GCN_layers(self.featureSize, self.featureSize, self.numKeypoints)
        self.L3 = GCN_layers(self.featureSize,self.featureSize, self.numKeypoints)
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
        x3 = self.relu(self.L2(x2, self.A))
        keypoints = self.L3(x3, self.A)
        return keypoints.permute(0, 2, 1)
    
    def forward(self, x):
        nodeFeat = self.generate_node_feature(x)
        heatmap = self.gcn_forward(nodeFeat).reshape(-1, self.numKeypoints, (self.height//2), (self.width//2))
        heatmap = F.interpolate(heatmap, scale_factor=2.0, mode='bilinear', align_corners=True)
        return self.sigmoid(heatmap).unsqueeze(2)


class GCN2(nn.Module): # output: keypoint coordinate
    def __init__(self, cfg, A):
        super(GCN2, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.modelType = cfg.MODEL.type
        self.gcnType = cfg.MODEL.gcnType
        self.featureSize = (self.height//2) * (self.width//2)
        self.L1 = GCN_layers(self.featureSize, 128, self.numKeypoints)
        self.L2 = GCN_layers(128, 64, self.numKeypoints)
        self.L3 = GCN_layers(64, 2, self.numKeypoints)
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
        x3 = self.relu(self.L2(x2, self.A))
        keypoints = self.L3(x3, self.A)
        return keypoints.permute(0, 2, 1)
    
    def forward(self, x):
        nodeFeat = self.generate_node_feature(x)
        keypoints = self.gcn_forward(nodeFeat)
        return keypoints #211028_2 use sigmoid
        #return keypoints

