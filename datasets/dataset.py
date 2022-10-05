import os
import random
import json
import torch
import numpy as np
from PIL import Image
from random import sample
import torch.nn.functional as F
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.base import BaseDataset, generateGTAnnot

def getDataset(phase, cfg, args, random=True):
    return HuPR3D_horivert(phase, cfg, args, random)

class HuPR3D_horivert(BaseDataset):
    def __init__(self, phase, cfg, args, random=True):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HuPR3D_horivert, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numChirps = cfg.DATASET.numChirps
        self.r = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.h = cfg.DATASET.elevationSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.sampling_ratio = args.sampling_ratio
        self.dirRoot = cfg.DATASET.dataDir
        self.idxToJoints = cfg.DATASET.idxToJoints
        self.random = random

        generateGTAnnot(cfg, phase)
        self.gtFile = os.path.join(self.dirRoot, '%s_gt.json' % phase)
        self.coco = COCO(self.gtFile)
        self.imageIds = self.coco.getImgIds()
        self.VRDAEPaths_hori = []
        self.VRDAEPaths_vert = []
        for name in self.imageIds:
            namestr = '%09d' % name
            self.VRDAEPaths_hori.append(os.path.join(self.dirRoot, 'single_%d/hori/%09d.npy'%(int(namestr[:4]), int(namestr[-4:]))))
            self.VRDAEPaths_vert.append(os.path.join(self.dirRoot, 'single_%d/vert/%09d.npy'%(int(namestr[:4]), int(namestr[-4:]))))
        self.annots = self._load_coco_keypoint_annotations()
        self.transformFunc = self.getTransformFunc(cfg)

    def evaluateEach(self, loadDir):
        res_file = os.path.join(loadDir, "%s_results.json"% self.phase)
        anns = json.load(open(res_file))
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        keypoint_list = []
        for i in range(self.numKeypoints):
            coco_eval.evaluate(i)
            coco_eval.accumulate()
            coco_eval.summarize()
            info_str = []
            for ind, name in enumerate(stats_names):
                info_str.append((name, coco_eval.stats[ind]))
            keypoint_list.append(info_str[0][1])
        for i in range(self.numKeypoints):
            print('%s: %.3f' % (self.idxToJoints[i], keypoint_list[i]))
        return info_str[0][1] # return the value of AP
    
    def evaluate(self, loadDir):
        res_file = os.path.join(loadDir, "%s_results.json"% self.phase)
        anns = json.load(open(res_file))
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        
        for idx_metric in range(10):
            print("%s:\t%.3f\t"%(info_str[idx_metric][0], info_str[idx_metric][1]), end='')
            if (idx_metric+1) % 5 == 0:
                print()
        return info_str[0][1] # return the value of AP

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.imageIds:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)
        rec = []
        for obj in objs:
            joints_2d = np.zeros((self.numKeypoints, 2), dtype=np.float)
            joints_2d_vis = np.zeros((self.numKeypoints, 2), dtype=np.float)
            for ipt in range(self.numKeypoints):
                joints_2d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_2d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_2d_vis[ipt, 0] = t_vis
                joints_2d_vis[ipt, 1] = t_vis
            rec.append({
                'joints': joints_2d,
                'joints_vis': joints_2d_vis,
                'bbox': obj['bbox'], # x, y, w, h
                'imageId': obj['image_id']
            })
        return rec
    def __getitem__(self, index):
        if self.random:
            index = index * random.randint(1, self.sampling_ratio)
        else:
            index = index * self.sampling_ratio
        # collect past frames and furture frames for the center target frame
        padSize = index % self.duration
        idx = index - self.numGroupFrames//2 - 1
        
        VRDAEmaps_hori = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.r, self.w, self.h))
        VRDAEmaps_vert = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.r, self.w, self.h))
        
        for j in range(self.numGroupFrames):
            if (j + padSize) <= self.numGroupFrames//2:
                idx = index - padSize
            elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                idx = index + (self.duration - 1 - padSize)
            else:
                idx += 1
            VRDAEPath_hori = self.VRDAEPaths_hori[idx]
            VRDAEPath_vert = self.VRDAEPaths_vert[idx]
            VRDAERealImag_hori = np.load(VRDAEPath_hori)
            VRDAERealImag_vert = np.load(VRDAEPath_vert)

            idxSampleChirps = 0
            for idxChirps in range(self.numChirps//2 - self.numFrames//2, self.numChirps//2 + self.numFrames//2):
                VRDAEmaps_hori[j, idxSampleChirps, 0, :, :, :] = self.transformFunc(VRDAERealImag_hori[idxChirps].real).permute(1, 2, 0)
                VRDAEmaps_hori[j, idxSampleChirps, 1, :, :, :] = self.transformFunc(VRDAERealImag_hori[idxChirps].imag).permute(1, 2, 0)
                VRDAEmaps_vert[j, idxSampleChirps, 0, :, :, :] = self.transformFunc(VRDAERealImag_vert[idxChirps].real).permute(1, 2, 0)
                VRDAEmaps_vert[j, idxSampleChirps, 1, :, :, :] = self.transformFunc(VRDAERealImag_vert[idxChirps].imag).permute(1, 2, 0)
                idxSampleChirps += 1

        joints = torch.LongTensor(self.annots[index]['joints'])
        bbox = torch.FloatTensor(self.annots[index]['bbox'])
        imageId = self.annots[index]['imageId']
        return {'VRDAEmap_hori': VRDAEmaps_hori,
                'VRDAEmap_vert': VRDAEmaps_vert,
                'imageId': imageId,
                'jointsGroup': joints,
                'bbox': bbox}
    
    def __len__(self):
        return len(self.VRDAEPaths_hori)//self.sampling_ratio