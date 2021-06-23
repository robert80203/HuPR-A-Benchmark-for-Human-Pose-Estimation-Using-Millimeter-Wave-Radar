import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def plotHumanPose(heatmaps, visDir, idx):
    # matplotlib
    heatmap = np.max(heatmaps, axis = 0)
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(heatmap)
    fileName = '%09d.png' % idx
    fig.savefig(os.path.join(visDir, fileName))

def plotHumanPoseRGBWithGT(preds, heatmapsGT, visDir, idx, upsamplingSize=(128, 128)):
    
    rgbPath = '../frames/20210609/single_6/processed/images/' + ('%09d.jpg' % idx)
    rgbImg = Image.open(rgbPath).convert('RGB')
    transforms_fn = transforms.Compose([
        transforms.Resize(upsamplingSize),
        transforms.ToTensor(),
    ])
    rgbImg = transforms_fn(rgbImg)

    # torchvision
    preds = torch.from_numpy(preds)
    heatmapsGT = torch.from_numpy(heatmapsGT)
    
    pred, _ = preds.max(dim=0, keepdim=True)
    heatmapGT, _ = heatmapsGT.max(dim=0, keepdim=True)
    
    pred = torch.cat((pred, torch.zeros(2, pred.size(1), pred.size(2))))
    heatmapGT = torch.cat((heatmapGT, torch.zeros(2, heatmapGT.size(1), heatmapGT.size(2))))
    
    img = torch.cat((pred.unsqueeze(0), heatmapGT.unsqueeze(0)), 0)
    img = F.interpolate(img, size=upsamplingSize, mode='bilinear', align_corners=True)
    img = torch.cat((img, rgbImg.unsqueeze(0)), 0)
    
    imgGrid = make_grid(img, nrow=2)
    fileName = '%09d.png' % idx
    save_image(imgGrid, os.path.join(visDir, fileName))