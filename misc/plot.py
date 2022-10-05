
import os
import cv2
import math
import torch
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

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

        # ndarr = cv2.putText(ndarr, cfg.TEST.outputName, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(imagePath, ndarr[:, :, [2, 1, 0]])