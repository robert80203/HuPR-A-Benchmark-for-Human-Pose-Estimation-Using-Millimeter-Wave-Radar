import os
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.txt']

class Normalize(object):
    def __init__(self, std_mean=False, use_log=False):
        self.std_mean = std_mean
        self.use_log = use_log
    
    def __call__(self, radarData):
        assert len(radarData.size()) == 3, 'input shape must be 3D'
        c = radarData.size(0)
        minValues = torch.min(radarData.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataZero = radarData - minValues

        if self.use_log:
            radarDataNorm = torch.log(radarDataZero + 1e-7)
        else:
            maxValues = torch.max(radarDataZero.view(c, -1), 1)[0].view(c, 1, 1)
            radarDataNorm = radarDataZero / maxValues
        
        if self.std_mean:
            std, mean = torch.std_mean(radarDataNorm.view(c, -1), 1)
            return (radarDataNorm - mean) / std
        else:
            return radarDataNorm 

def generateGTAnnot(cfg, phase='train'):
    annot = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    annot["info"] = {
        "description": "HuPR dataset",
        "url": "",
        "version": "1.0",
        "year": 2021,
        "contributor": "UW-NYCU-AI-Labs",
        "date_created": "2021/11/06"
    }
    annot["categories"] = [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", 
                "L_Ankle", "Neck", "Head", "L_Shoulder", "L_Elbow", 
                "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"
            ],
            "skeleton": [
                [14,13],[13,12],[11,10],[10,9],[9,7],[12,9],[8,7],[7,1],[7,4],[6,5],[5,4],[3,2],[2,1]
            ]
    }]
    group_idx = eval('cfg.DATASET.'+phase+'Name')
    for idx in group_idx:
        with open(os.path.join(cfg.DATASET.dataDir, 'single_%d/annot/hrnet_annot.json'% idx)) as fp:
        #with open(os.path.join(cfg.DATASET.dataDir, 'single_%d/annot/mmlab.json'% idx)) as fp:
            mmlab = json.load(fp)
        for block in mmlab:
            image_id = int(block['image'][:9]) + idx * 100000 #image_id = id
            joints = np.array(block["joints"])
            bbox = block["bbox"]
            visIdx = np.ones((14 , 1)) + 1.0 #2 is labeled and visible
            joints = np.concatenate((joints, visIdx), axis=1).reshape(len(joints) * 3).tolist()
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 2
            annot["annotations"].append({
                "num_keypoints": 14,
                "area": area,
                "iscrowd": 0,
                "keypoints": joints,
                "image_id": image_id,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "category_id": 1,
                "id": image_id,
            })
            annot["images"].append({
                "license": -1,
                "file_name": block['image'],
                "coco_url": "None",
                "height": 256,
                "width": 256,
                "date_captured": "None",
                "flickr_url": "None",
                "id": image_id
            })
        print('Generate GTs for single_%d for %s stage'%(idx, phase), end='\r')

    with open(os.path.join(cfg.DATASET.dataDir,'%s_gt.json'%phase), 'w') as fp:
        json.dump(annot, fp)

class BaseDataset(data.Dataset):
    def __init__(self, phase):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))

        super(BaseDataset, self).__init__()
        #self.resizeShape = resizeShape
        #self.cropShape = cropShape
        self.phase = phase
        #self.transformFunc = self.getTransformFunc()

    def getTransformFunc(self, cfg):
        if self.phase == 'train':
            transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize(std_mean=cfg.DATASET.stdmean, use_log=cfg.DATASET.log)
            ])
        else:
            transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize(std_mean=cfg.DATASET.stdmean, use_log=cfg.DATASET.log)
            ])
        return transformFunc
    
    def isImageFile(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def getPaths(self, dataDirGroup, dirGroup, mode, frameGroup):
        num = len(dataDirGroup)
        images = []
        for i in range(num):
            for dirName in dirGroup[i]:
                for frame in frameGroup:
                    path = os.path.join(dataDirGroup[i], dirName, mode, frame + '.npy')
                    images.append(path)
        return images
        # for dirName in dirGroup:
        #     for frame in frameGroup:
        #         path = os.path.join(dataDir, dirName, mode, frame + '.npy')
        #         images.append(path)
        # return images

    def getAnnots(self, dataDirGroup, dirGroup, mode, fileName):
        num = len(dataDirGroup)
        annots = []
        for i in range(num):
            for dirName in dirGroup[i]:
                path = os.path.join(dataDirGroup[i], dirName, mode, fileName)
                with open(path, 'r') as fp:
                    annot = json.load(fp)
                annots.extend(annot)
        return annots

        # for dirName in dirGroup:
        #     path = os.path.join(dataDir, dirName, mode, fileName)
        #     with open(path, 'r') as fp:
        #         annot = json.load(fp)
        #     annots.extend(annot)
        # return annots

    def __getitem__(self, idx):
        raise NotImplementedError('Subclass of BaseDataset must implement __getitem__')

    def __len__(self):
        raise NotImplementedError('Subclass of BaseDataset must implement __len__')