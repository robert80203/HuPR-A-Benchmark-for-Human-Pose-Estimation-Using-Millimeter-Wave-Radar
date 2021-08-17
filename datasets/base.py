import os
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from PIL import Image
import json

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.txt']

class Normalize(object):
    def __init__(self, std_mean=False):
        self.std_mean = std_mean
    def __call__(self, radarData):
        assert len(radarData.size()) == 3, 'input shape must be 3D'
        c = radarData.size(0)
        minValues = torch.min(radarData.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataZero = radarData - minValues
        maxValues = torch.max(radarDataZero.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataNorm = radarDataZero / maxValues
        if self.std_mean:
            std, mean = torch.std_mean(radarDataNorm.view(c, -1), 1)
            return (radarDataNorm - mean) / std
        else:
            return radarDataNorm

class BaseDataset(data.Dataset):
    def __init__(self, dataDir, phase):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))

        super(BaseDataset, self).__init__()
        self.dataDir = dataDir
        #self.resizeShape = resizeShape
        #self.cropShape = cropShape
        self.phase = phase
        self.transformFunc = self.getTransformFunc()

    def getTransformFunc(self):
        if self.phase == 'train':
            transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize(std_mean=True)
            ])
        else:
            transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize(std_mean=True)
            ])
        return transformFunc
    
    def isImageFile(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def getPaths(self, dataDir, dirGroup, mode, frameGroup):
        images = []
        for dirName in dirGroup:
            for frame in frameGroup:
                path = os.path.join(dataDir, dirName, mode, frame + '.npy')
                images.append(path)
        return images

    def getAnnots(self, dataDir, dirGroup, mode, fileName):
        annots = []
        for dirName in dirGroup:
            path = os.path.join(dataDir, dirName, mode, fileName)
            with open(path, 'r') as fp:
                annot = json.load(fp)
            annots.extend(annot)
        return annots

    def __getitem__(self, idx):
        raise NotImplementedError('Subclass of BaseDataset must implement __getitem__')

    def __len__(self):
        raise NotImplementedError('Subclass of BaseDataset must implement __len__')