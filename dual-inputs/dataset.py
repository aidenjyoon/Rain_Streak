import os
import itertools
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

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
        
        print("THIS IS THE INDEX", index)
        
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
            
            mask1 = input1 - input2
            mask2 = input2 - input1
            target1_mask = mask1 >= 0
            target2_mask = mask2 >= 0
            target1 = mask1[target1_mask]
            target2 = mask2[target2_mask]
            target_rain1 = mask1[target2_mask]
            target_rain2 = mask2[target1_mask]
            
            if self.transform is not None:
                input1 = self.transform(input1)
                input2 = self.transform(input2)
            if self.target_transform is not None:
                target1 = self.target_transform(target1)
                target2 = self.target_transform(target2)
            if self.rain_transform is not None:
                target_rain1 = self.rain_transform(target_rain1)
                target_rain2 = self.rain_transform(target_rain2)
            
            return input1, input2, target1, target2, target_rain1, target_rain2
            
    def __len__(self):
        return len(self.ids)
