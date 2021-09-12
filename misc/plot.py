import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def assignRGBcolor(pred, gt):
    colorGroup = [
        (128, 0, 0),#RightAnkle
        (255, 0, 0),#RightKnee
        (255, 127,80),#RightHip
        (255,165,0),#LeftHip
        (218,165,32),#LeftKnee
        (128,128,0),#LeftAnkle
        (0,128,0),#Pelvis
        (0,255,255),#Chest
        (70,130,180),#Neck
        (30,144,255),#head
        (0,0,255),#RightWrist
        (138,43,226),#RightElbow
        (139,0,139),#RightShoulder
        (216,191,216),#LeftShoulder
        (255,105,180),#LeftElbow
        (245,222,179),#LeftWrist
        (0,0,0)
    ]
    colorGroup = torch.FloatTensor(colorGroup)
    preds = torch.zeros(3, 1, pred.size(1), pred.size(2))
    gts = torch.zeros(3, 1, gt.size(1), gt.size(2))

    for i in range(17):#16 keypoints + 1 none
        preds[0][pred == i] = colorGroup[i][0]
        preds[1][pred == i] = colorGroup[i][1]
        preds[2][pred == i] = colorGroup[i][2]
        gts[0][gt == i] = colorGroup[i][0]
        gts[1][gt == i] = colorGroup[i][1]
        gts[2][gt == i] = colorGroup[i][2]
    
    preds = preds.squeeze(1) / 255.
    gts = gts.squeeze(1) / 255.

    #print(gts[:,0,0])

    return preds, gts

def plotHumanPoseRGBWithGT(preds, heatmapsGT, isEval, cfg,
                           visDir, idx, upsamplingSize=(128, 128)):

    # only plot 1 specific images of a directory
    # rgbPath = os.path.join('../frames', cfg.LOGGER.plotImgDir, 
    #                         cfg.DATASET.testName[0] if isEval else cfg.DATASET.valName[0], 
    #                         'processed/images', '%09d.jpg' % idx)
    rgbPath = os.path.join('../frames', cfg.LOGGER.plotImgDir, 
                            cfg.DATASET.testName[0][0] if isEval else cfg.DATASET.valName[0][0], 
                            'processed/images', '%09d.jpg' % idx)


    rgbImg = Image.open(rgbPath).convert('RGB')
    transforms_fn = transforms.Compose([
        transforms.Resize(upsamplingSize),
        transforms.ToTensor(),
    ])
    rgbImg = transforms_fn(rgbImg)
    # torchvision
    preds = torch.from_numpy(preds)
    heatmapsGT = torch.from_numpy(heatmapsGT)
    
    pred, predIdx = preds.max(dim=0, keepdim=True)
    heatmapGT, gtIdx = heatmapsGT.max(dim=0, keepdim=True)

    #if isColormap:
    #print(predIdx.size())
    predIdx[pred <= 0.5] = 16
    gtIdx[heatmapGT <= 0.5] = 16
    predColor, heatmapGTColor = assignRGBcolor(predIdx, gtIdx)
    #else:
    pred = torch.cat((pred, torch.zeros(2, pred.size(1), pred.size(2))))
    heatmapGT = torch.cat((heatmapGT, torch.zeros(2, heatmapGT.size(1), heatmapGT.size(2))))
    
    img = torch.cat((pred.unsqueeze(0), heatmapGT.unsqueeze(0), predColor.unsqueeze(0)), 0)
    img = F.interpolate(img, size=upsamplingSize, mode='bilinear', align_corners=True)
    img = torch.cat((img, rgbImg.unsqueeze(0)), 0)
    
    imgGrid = make_grid(img, nrow=2, padding=10, pad_value=255.0)
    fileName = '%09d.png' % idx
    save_image(imgGrid, os.path.join(visDir, fileName))