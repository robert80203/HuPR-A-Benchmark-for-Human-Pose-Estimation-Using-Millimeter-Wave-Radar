import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision
import math
import cv2
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
    rgbPath = os.path.join('../frames', cfg.TEST.plotImgDir, 
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
    
    img = torch.cat((pred.unsqueeze(0), heatmapGTColor.unsqueeze(0), predColor.unsqueeze(0)), 0)
    img = F.interpolate(img, size=upsamplingSize, mode='bilinear', align_corners=True)
    img = torch.cat((img, rgbImg.unsqueeze(0)), 0)
    
    imgGrid = make_grid(img, nrow=2, padding=10, pad_value=255.0)
    fileName = '%09d.png' % idx
    save_image(imgGrid, os.path.join(visDir, fileName))


#def plotHumanPose(batch_joints, cfg, visDir, idx, bbox=None, upsamplingSize=(256, 256), nrow=8, padding=2):
def plotHumanPose(batch_joints, cfg, visDir, imageIdx, bbox=None, upsamplingSize=(256, 256), nrow=8, padding=2):
    for j in range(len(batch_joints)):
        namestr = '%09d'%imageIdx[j].item()
        imageDir = os.path.join(visDir, 'single_%d'%int(namestr[:4]))
        if not os.path.isdir(imageDir):
            os.mkdir(imageDir)
        imagePath = os.path.join(imageDir, '%09d.png'%int(namestr[-4:]))
        rgbPath = os.path.join('../frames', cfg.TEST.plotImgDir, 'single_%d'%int(namestr[:4]), 'processed/images', '%09d.jpg'%int(namestr[-4:]))
        rgbImg = Image.open(rgbPath).convert('RGB')
        transforms_fn = transforms.Compose([
            transforms.Resize(upsamplingSize),
            transforms.ToTensor(),
        ])
        batch_image = transforms_fn(rgbImg).unsqueeze(0)
        s_joints = np.expand_dims(batch_joints[j], axis=0)
        
        grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        ndarr = ndarr.copy()

        nmaps = batch_image.size(0)
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height = int(batch_image.size(2) + padding)
        width = int(batch_image.size(3) + padding)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                #joints = batch_joints[k]
                joints = s_joints[k]
                for joint in joints:
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                k = k + 1
        joints_edges = [[(int(joints[0][0]), int(joints[0][1])), (int(joints[1][0]), int(joints[1][1]))],
                        [(int(joints[1][0]), int(joints[1][1])), (int(joints[2][0]), int(joints[2][1]))],
                        [(int(joints[0][0]), int(joints[0][1])), (int(joints[3][0]), int(joints[3][1]))],
                        [(int(joints[3][0]), int(joints[3][1])), (int(joints[4][0]), int(joints[4][1]))],
                        [(int(joints[4][0]), int(joints[4][1])), (int(joints[5][0]), int(joints[5][1]))],
                        [(int(joints[0][0]), int(joints[0][1])), (int(joints[6][0]), int(joints[6][1]))],
                        [(int(joints[3][0]), int(joints[3][1])), (int(joints[6][0]), int(joints[6][1]))],
                        [(int(joints[6][0]), int(joints[6][1])), (int(joints[7][0]), int(joints[7][1]))],
                        [(int(joints[6][0]), int(joints[6][1])), (int(joints[8][0]), int(joints[8][1]))],
                        [(int(joints[6][0]), int(joints[6][1])), (int(joints[11][0]), int(joints[11][1]))],
                        [(int(joints[8][0]), int(joints[8][1])), (int(joints[9][0]), int(joints[9][1]))],
                        [(int(joints[9][0]), int(joints[9][1])), (int(joints[10][0]), int(joints[10][1]))],
                        [(int(joints[11][0]), int(joints[11][1])), (int(joints[12][0]), int(joints[12][1]))],
                        [(int(joints[12][0]), int(joints[12][1])), (int(joints[13][0]), int(joints[13][1]))],
        ]
        for joint_edge in joints_edges:
            ndarr = cv2.line(ndarr, joint_edge[0], joint_edge[1], [255, 0, 0], 1)

        if bbox is not None:
            topleft = (int(bbox[j][0].item()), int(bbox[j][1].item()))
            topright = (int(bbox[j][0].item() + bbox[j][2].item()), int(bbox[j][1]))
            botleft = (int(bbox[j][0].item()), int(bbox[j][1].item() + bbox[j][3].item()))
            botright = (int(bbox[j][0].item() + bbox[j][2].item()), int(bbox[j][1].item() + bbox[j][3].item()))
            ndarr = cv2.line(ndarr, topleft, topright, [0, 255, 0], 1)
            ndarr = cv2.line(ndarr, topleft, botleft, [0, 255, 0], 1)
            ndarr = cv2.line(ndarr, topright, botright, [0, 255, 0], 1)
            ndarr = cv2.line(ndarr, botleft, botright, [0, 255, 0], 1)

        ndarr = cv2.putText(ndarr, cfg.TEST.outputName, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(imagePath, ndarr[:, :, [2, 1, 0]])