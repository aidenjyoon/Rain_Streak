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
            self.ids = sorted(os.listdir(os.path.join(root,"fake")))
        
    def __getitem__(self, index):
        img = self.ids[index]
        if self.real:
            input = Image.open(os.path.join(self.root, img)).conver('RGB')
            if self.transform is not None:
                input = self.transform(input)
            return input
        else:
            input = Image.open(os.path.join(self.root, 'I', img)).convert('RGB')        # Image
            target = Image.open(os.path.join(self.root, 'B', img)).convert('RGB')       # Background
            target_rain = Image.open(os.path.join(self.root, 'R', img)).convert('RGB')  # Rain
            
            if self.transform is not None:
                input = self.transform(input)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.rain_transform is not None:
                target_rain = self.rain_transform(target_rain)
            
            return input, target, target_rain
            
    def __len__(self):
        return len(self.ids)