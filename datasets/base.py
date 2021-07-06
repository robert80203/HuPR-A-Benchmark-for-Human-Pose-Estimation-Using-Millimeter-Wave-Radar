import os
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from PIL import Image
import json

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.txt']


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
                #transforms.Resize(self.resize_shape),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                #transforms.CenterCrop(self.crop_shape),
                transforms.ToTensor()
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transformFunc = transforms.Compose([
                #transforms.Resize(self.resize_shape),
                #transforms.CenterCrop(self.crop_shape),
                transforms.ToTensor()
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return transformFunc

    def isImageFile(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def getPaths(self, dataDir, dirGroup, mode, frameGroup):
        images = []
        for dirName in dirGroup:
            for frame in frameGroup:
                path = os.path.join(dataDir, dirName, mode, frame + '.txt')
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