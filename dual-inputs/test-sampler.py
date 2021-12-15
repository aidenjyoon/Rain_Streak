import os
import itertools
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

import cv2
from skimage import io
from skimage.util import img_as_float

import argparse
import os
import random
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from math import log10
from PIL import Image


class sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, invalid_idx):
        self.data_source = data_source
        self.invalid_idx = invalid_idx
        
    def __iter__(self):
        indices = torch.arange(len(self.data_source))
        paired_indices = indices.unfold(0,2,1)
        paired_indices = torch.stack(
            [paired_indices[i] for i in range(len(paired_indices))]
        )
        paired_indices = paired_indices[torch.randperm(len(paired_indices))]
        indices = paired_indices.view(-1)
        
        return iter(indices.tolist())
        
    def __len__(self):
        return len(self.data_source)

class rain_dataset(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 rain_transform=None,
                 real=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.rain_transform = rain_transform
        self.real = real
        
        if real:
            self.ids = sorted(os.listdir(root))
        else:
            self.ids = sorted(os.listdir(os.path.join(root)))
        
    def __getitem__(self, index):
        
        # print("THIS IS THE INDEX", index)
        
        img = self.ids[index]
        if self.real:
            input = Image.open(os.path.join(self.root, img)).convert('RGB')
            if self.transform is not None:
                input = self.transform(input)
            return input
        else:
            # input = Image.open(os.path.join(self.root, 'I', img)).convert('RGB')        # Image
            # target = Image.open(os.path.join(self.root, 'B', img)).convert('RGB')       # Background
            # target_rain = Image.open(os.path.join(self.root, 'R', img)).convert('RGB')  # Rain

            input1 = Image.open(os.path.join(self.root, img)).convert('RGB')        # Image
            input2 = Image.open(os.path.join(self.root, img)).convert('RGB')        # Image
            
            extracted_rain1 = img_as_float(input1) - img_as_float(input2)
            extracted_rain2 = img_as_float(input2) - img_as_float(input1)
            
            rain1_mask = extracted_rain1 >= 0
            rain2_mask = extracted_rain2 >= 0
            rain1 = extracted_rain1 * rain1_mask
            rain2 = extracted_rain2 * rain2_mask
            background1 = img_as_float(input1) - img_as_float(rain1)
            background2 = img_as_float(input2) - img_as_float(rain2)
            
            if self.transform is not None:
                input1 = self.transform(input1)
                input2 = self.transform(input2)
            if self.target_transform is not None:
                target1 = self.target_transform(background1)
                target2 = self.target_transform(background2)
            if self.rain_transform is not None:
                target_rain1 = self.rain_transform(rain1)
                target_rain2 = self.rain_transform(rain2)
            
            return input1, input2, target1, target2, target_rain1, target_rain2
            
    def __len__(self):
        return len(self.ids)
    
    

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', 
                        help='path to dataset',
                        default='../rain_generator/train_horse/',
                        type=str)
parser.add_argument('--real', 
                        help='test real images',
                        action='store_true')
parser.add_argument('--workers', 
                        help='number of data loading workers', 
                        default=2,
                        type=int)
parser.add_argument('--batchSize', 
                        help='input batch size',
                        default=8,
                        type=int)

args = parser.parse_args()

transform = transforms.Compose([
    # transforms.Scale(opt.imageSize),
    # transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = rain_dataset(
    args.dataroot,
    transform=transform,
    target_transform=transform,
    rain_transform=transform,
    real=args.real)
assert dataset

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batchSize,
    shuffle=False,
    num_workers=int(args.workers),
    sampler=sampler)

print('====================================')
print("LENGTH OF DATALOADER  -  ", len(dataloader))
print('====================================')

for i, data in enumerate(dataloader, 1):
    print(i, data)