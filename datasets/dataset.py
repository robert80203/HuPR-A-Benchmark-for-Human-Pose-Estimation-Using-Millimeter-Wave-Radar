import os
import random
from random import sample
from datasets.base import BaseDataset, generateGTAnnot
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json



def getDataset(phase, cfg, sampling_ratio):
    #return HumanRadarRDA(phase, cfg, sampling_ratio) if cfg.DATASET.useVelocity else HumanRadarRA(phase, cfg, sampling_ratio)
    if "stage2" in cfg.MODEL.frontModel:
        return HuPRRefine(phase, cfg, sampling_ratio)
    else:
        return HuPR(phase, cfg, sampling_ratio)
    #return HumanRadarRA(phase, cfg, sampling_ratio)

###################################
# follow the setup of coco dataset
# Refine the 2d skeletons
###################################
class HuPRRefine(BaseDataset):
    def __init__(self, phase, cfg, sampling_ratio=1):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HuPRRefine, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numChirps = cfg.DATASET.numChirps
        #self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.heatmapSize
        self.w = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.chirpModel = cfg.MODEL.chirpModel
        self.sampling_ratio = sampling_ratio
        self.dirRoot = cfg.DATASET.dataDir #'/work/robert80203/Radar_detection/radar_skeleton_estimation/data/20211106_Pad_CltRm_RDA'
        self.pretrainDir = cfg.DATASET.pretrainDir

        if os.path.exists(os.path.join(cfg.DATASET.dataDir,'%s_gt.json'%phase)):
            print('File %s exists'%os.path.join(cfg.DATASET.dataDir,'%s_gt.json'%phase))
        else:
            generateGTAnnot(cfg, phase)
        self.gtFile = os.path.join(self.dirRoot, '%s_gt.json' % phase)
        self.coco = COCO(self.gtFile)
        self.imageIds = self.coco.getImgIds()
        
        #with open(os.path.join("./logs/eccv2022/stgcn2_stage1/%s_results.json" % phase), "r") as fp:
        #with open(os.path.join("./logs/test/%s_results.json" % phase), "r") as fp:
        with open(os.path.join(self.pretrainDir, "%s_results.json" % phase), "r") as fp:
            self.jointsData = json.load(fp)
        self.annots = self._load_coco_keypoint_annotations()
        self.transformFunc = self.getTransformFunc(cfg)

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

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.imageIds:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db
    def _load_coco_keypoint_annotation_kernal(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        #width = im_ann['width']
        #height = im_ann['height']

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
                'imageId': obj['image_id'],
            })
        return rec
    def __getitem__(self, index):
        index = index * random.randint(1, self.sampling_ratio)
        padSize = index % self.duration
        idx = index - self.numGroupFrames//2 - 1

        jointsdata = torch.zeros((self.numGroupFrames, self.numKeypoints * 3))
        sigma = torch.zeros((self.numGroupFrames, self.numKeypoints))

        for j in range(self.numGroupFrames):
            if (j + padSize) <= self.numGroupFrames//2:
                idx = index - padSize
            elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                idx = index + (self.duration - 1 - padSize)
            else:
                idx += 1

            datablock = self.jointsData[idx]
            jointsdata[j] = torch.FloatTensor(datablock["keypoints"])
            sigma[j] = torch.FloatTensor(datablock['sigma'])

        joints = torch.LongTensor(self.annots[index]['joints'])
        bbox = torch.FloatTensor(self.annots[index]['bbox'])
        imageId = self.annots[index]['imageId']
        imageId2 = self.jointsData[index]["image_id"]
        jointsdata = jointsdata.view(self.numGroupFrames, self.numKeypoints, 3).permute(2, 0, 1)
        jointsdata[0, :, :] = jointsdata[0, :, :]/(self.w/2) - 1
        jointsdata[1, :, :] = jointsdata[1, :, :]/(self.h/2) - 1

        return {'jointsdata': jointsdata,
                'imageId': imageId,
                'imageId2': imageId2,
                'jointsGroup': joints,
                'bbox': bbox,
                'sigma': sigma}
    def __len__(self):
        return len(self.jointsData)//self.sampling_ratio

###################################
# follow the setup of coco dataset
###################################
class HuPR(BaseDataset):
    def __init__(self, phase, cfg, sampling_ratio=1):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HuPR, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numChirps = cfg.DATASET.numChirps
        #self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.chirpModel = cfg.MODEL.chirpModel
        self.sampling_ratio = sampling_ratio
        self.dirRoot = cfg.DATASET.dataDir #'/work/robert80203/Radar_detection/radar_skeleton_estimation/data/20211106_Pad_CltRm_RDA'

        if os.path.exists(os.path.join(cfg.DATASET.dataDir,'%s_gt.json'%phase)):
            print('File %s exists'%os.path.join(cfg.DATASET.dataDir,'%s_gt.json'%phase))
        else:
            generateGTAnnot(cfg, phase)
        self.gtFile = os.path.join(self.dirRoot, '%s_gt.json' % phase)
        #self.gtFile = os.path.join(self.dirRoot, 'train_gt.json')
        self.coco = COCO(self.gtFile)
        self.imageIds = self.coco.getImgIds()
        
        self.horiPaths = []
        self.vertPaths = []
        for name in self.imageIds:
            namestr = '%09d' % name
            self.horiPaths.append(os.path.join(self.dirRoot, 'single_%d/hori/%09d.npy'%(int(namestr[:4]), int(namestr[-4:]))))
            self.vertPaths.append(os.path.join(self.dirRoot, 'single_%d/verti/%09d.npy'%(int(namestr[:4]), int(namestr[-4:]))))

        self.annots = self._load_coco_keypoint_annotations()
        self.transformFunc = self.getTransformFunc(cfg)

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

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.imageIds:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db
    def _load_coco_keypoint_annotation_kernal(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        #width = im_ann['width']
        #height = im_ann['height']

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
        #index = index * self.sampling_ratio
        index = index * random.randint(1, self.sampling_ratio)
        #if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
        # collect past frames and furture frames for the center target frame
        padSize = index % self.duration
        idx = index - self.numGroupFrames//2 - 1
        
        horiImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
        vertImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
        
        for j in range(self.numGroupFrames):
            if (j + padSize) <= self.numGroupFrames//2:
                idx = index - padSize
            elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                idx = index + (self.duration - 1 - padSize)
            else:
                idx += 1

            horiPath = self.horiPaths[idx]
            vertPath = self.vertPaths[idx]
            
            horiRealImag = np.load(horiPath)
            vertRealImag = np.load(vertPath)

            idxSampleChirps = 0
            #for idxChirps in range(self.numFrames):
            #print(horiRealImag.shape, horiImgss.shape, horiPath)
            #print(vertRealImag.shape, vertImgss.shape, vertPath)
            for idxChirps in range(0, self.numChirps, self.numChirps//self.numFrames):
                horiImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(horiRealImag[idxChirps].real)
                horiImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(horiRealImag[idxChirps].imag)
                vertImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(vertRealImag[idxChirps].real)
                vertImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(vertRealImag[idxChirps].imag)
                idxSampleChirps += 1
        
        if self.chirpModel == 'maxpool':
            #print(horiImgss.size())
            horiImgss = F.max_pool3d(horiImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)
            vertImgss = F.max_pool3d(vertImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)

        joints = torch.LongTensor(self.annots[index]['joints'])
        bbox = torch.FloatTensor(self.annots[index]['bbox'])
        imageId = self.annots[index]['imageId']

        return {'horiImg': horiImgss, # shape: (# of frames, # of chirps, # of channel=2, h, w)
                'vertImg': vertImgss,  # shape: (# of frames, # of chirps, # of channel=2, h, w)
                'imageId': imageId,
                'jointsGroup': joints,
                'bbox': bbox}
    def __len__(self):
        return len(self.horiPaths)//self.sampling_ratio

class HumanRadarRA(BaseDataset):
    def __init__(self, phase, cfg, sampling_ratio=1):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HumanRadarRA, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        #self.numSamples = cfg.DATASET.numSamples
        self.numChirps = cfg.DATASET.numChirps
        self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.chirpModel = cfg.MODEL.chirpModel
        self.sampling_ratio = sampling_ratio
        
        if phase == 'test':
             dirGroup = cfg.DATASET.testName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        elif phase == 'val':
             dirGroup = cfg.DATASET.valName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        else:
             dirGroup = cfg.DATASET.trainName
             dataDirGroup = cfg.DATASET.dataDir
        
        #numFramesEachClip = self.duration * self.numFrames
        numFramesEachClip = self.duration
        idxFrameGroup = [('%09d' % i) for i in range(numFramesEachClip)]
        self.horiPaths = self.getPaths(dataDirGroup, dirGroup, 'hori', idxFrameGroup)
        self.vertPaths = self.getPaths(dataDirGroup, dirGroup, 'verti', idxFrameGroup)
        #self.annots = self.getAnnots(dataDirGroup, dirGroup, 'annot', 'hrnet_annot.json')
        self.annots = self.getAnnots(dataDirGroup, dirGroup, 'annot', 'mmlab.json')
        self.transformFunc = self.getTransformFunc(cfg)

    def __getitem__(self, index):
        #index = index * self.sampling_ratio
        index = index * random.randint(1, self.sampling_ratio)
        if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            # collect past frames and furture frames for the center target frame
            padSize = index % self.duration
            idx = index - self.numGroupFrames//2 - 1
            
            horiImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
            vertImgss = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.h, self.w))
            
            for j in range(self.numGroupFrames):
                if (j + padSize) <= self.numGroupFrames//2:
                    idx = index - padSize
                elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                    idx = index + (self.duration - 1 - padSize)
                else:
                    idx += 1

                horiPath = self.horiPaths[idx]
                vertPath = self.vertPaths[idx]
                
                horiRealImag = np.load(horiPath)
                vertRealImag = np.load(vertPath)

                idxSampleChirps = 0
                #for idxChirps in range(self.numFrames):
                #print(horiRealImag.shape, horiImgss.shape, horiPath)
                #print(vertRealImag.shape, vertImgss.shape, vertPath)
                for idxChirps in range(0, self.numChirps, self.numChirps//self.numFrames):
                    horiImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(horiRealImag[idxChirps].real)
                    horiImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(horiRealImag[idxChirps].imag)
                    vertImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(vertRealImag[idxChirps].real)
                    vertImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(vertRealImag[idxChirps].imag)
                    idxSampleChirps += 1
            
            if self.chirpModel == 'maxpool':
                #print(horiImgss.size())
                horiImgss = F.max_pool3d(horiImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)
                vertImgss = F.max_pool3d(vertImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)

            joints = torch.LongTensor(self.annots[index]['joints'])
            bbox = torch.FloatTensor(self.annots[index]['bbox'])

            return {'horiImg': horiImgss, # shape: (# of frames, # of chirps, # of channel=2, h, w)
                    'horiPath': horiPath,
                    'vertImg': vertImgss,  # shape: (# of frames, # of chirps, # of channel=2, h, w)
                    'vertPath': vertPath,
                    'jointsGroup': joints,
                    'bbox': bbox}
    def __len__(self):
        return len(self.horiPaths)//self.sampling_ratio

class HumanRadarRDA(BaseDataset):
    def __init__(self, phase, cfg, sampling_ratio=1):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HumanRadarRDA, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 30 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        #self.numSamples = cfg.DATASET.numSamples
        self.numChirps = cfg.DATASET.numChirps
        self.mode = cfg.DATASET.mode
        self.h = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.chirpModel = cfg.MODEL.chirpModel
        self.sampling_ratio = sampling_ratio
        
        if phase == 'test':
             dirGroup = cfg.DATASET.testName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
             #self.sampling_ratio = 1
        elif phase == 'val':
             dirGroup = cfg.DATASET.valName
             dataDirGroup = [cfg.DATASET.dataDir[0]]
        else:
             dirGroup = cfg.DATASET.trainName
             dataDirGroup = cfg.DATASET.dataDir
             #self.sampling_ratio = 100
        
        #numFramesEachClip = self.duration * self.numFrames
        numFramesEachClip = self.duration
        idxFrameGroup = [('%09d' % i) for i in range(numFramesEachClip)]
        self.horiPaths = self.getPaths(dataDirGroup, dirGroup, 'hori', idxFrameGroup)
        self.vertPaths = self.getPaths(dataDirGroup, dirGroup, 'verti', idxFrameGroup)
        self.horiVPaths = self.getPaths(dataDirGroup, dirGroup, 'horiv', idxFrameGroup)
        self.vertVPaths = self.getPaths(dataDirGroup, dirGroup, 'vertv', idxFrameGroup)
        self.annots = self.getAnnots(dataDirGroup, dirGroup, 'annot', 'hrnet_annot.json')
        self.transformFunc = self.getTransformFunc(cfg)

    def __getitem__(self, index):
        #index = index * self.sampling_ratio
        index = index * random.randint(1, self.sampling_ratio)
        if self.mode == 'multiFramesChirps' or self.mode == 'multiFrames':
            # collect past frames and furture frames for the center target frame
            padSize = index % self.duration
            idx = index - self.numGroupFrames//2 - 1
            
            horiImgss = torch.zeros((self.numGroupFrames, self.numFrames, 3, self.h, self.w))
            vertImgss = torch.zeros((self.numGroupFrames, self.numFrames, 3, self.h, self.w))
            
            for j in range(self.numGroupFrames):
                if (j + padSize) <= self.numGroupFrames//2:
                    idx = index - padSize
                elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                    idx = index + (self.duration - 1 - padSize)
                else:
                    idx += 1

                horiPath = self.horiPaths[idx]
                vertPath = self.vertPaths[idx]
                horiVPath = self.horiVPaths[idx]
                vertVpath = self.vertVPaths[idx]
                
                horiRealImag = np.load(horiPath)
                vertRealImag = np.load(vertPath)
                horiV = np.load(horiVPath)
                vertV = np.load(vertVpath)

                idxSampleChirps = 0
                for idxChirps in range(0, self.numChirps, self.numChirps//self.numFrames):
                    horiImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(horiRealImag[idxChirps].real)
                    horiImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(horiRealImag[idxChirps].imag)
                    horiImgss[j, idxSampleChirps, 2, :, :] = self.transformFunc(horiV[idxChirps])
                    vertImgss[j, idxSampleChirps, 0, :, :] = self.transformFunc(vertRealImag[idxChirps].real)
                    vertImgss[j, idxSampleChirps, 1, :, :] = self.transformFunc(vertRealImag[idxChirps].imag)
                    vertImgss[j, idxSampleChirps, 2, :, :] = self.transformFunc(vertV[idxChirps])

                    idxSampleChirps += 1
            
            if self.chirpModel == 'maxpool':
                #print(horiImgss.size())
                horiImgss = F.max_pool3d(horiImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)
                vertImgss = F.max_pool3d(vertImgss.permute(0, 2, 1, 3, 4), kernel_size=(self.numFrames, 1, 1)).squeeze(2)

            joints = torch.LongTensor(self.annots[index]['joints'])

            return {'horiImg': horiImgss, # shape: (# of frames, # of chirps, # of channel=3, h, w)
                    'vertImg': vertImgss,  # shape: (# of frames, # of chirps, # of channel=3, h, w)
                    'jointsGroup': joints}
    def __len__(self):
        return len(self.horiPaths)//self.sampling_ratio