import cv2
from skimage import io
from skimage.util import img_as_float
import sklearn
import numpy as np
import argparse
import os
import random
import time
from collections import OrderedDict
import itertools
from functools import reduce
from math import log10
from PIL import Image

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

        

class sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=2):
        self.data_source = data_source
        self.batch_size = batch_size
        
        # names and idicies
        self.indices = torch.arange(len(self.data_source))
        self.img_names = self.data_source.ids # ie. img_n112_0deg_rc200_2.jpg

        # number of desired dataset length
        self.n = len(self.data_source)

    def get_dict(self):
        '''
        create a dictionary of synthesized images
        key - which image was used for synthesis in string integers ('0', '1', '2',..)
        value - list of image file names synthesized with key image
            (ie. if key = 1, value = [img_n1_0deg_rc200_2.jpg, img_n1_33deg_rc200_2.jpg])
        '''
        
        # go through img-names and create a dictionary of item: count
        img_n = ''
        img_files = []
        dict = {}
        for name in self.img_names:
            temp = ''
            # iterate through .jpg name
            for idx, char in enumerate(name):
                # check if at img_n
                if idx > 4:
                    # check if char isn't surpassing n..#.._
                    if char != '_':
                        # get the img_n
                        temp = temp + char
                    else:
                        break
    
            if img_n == '':
                img_n = temp
                dict[img_n] = 0
                
            # check if img_n changed or not
            if img_n == temp:   # if same
                img_files.append(name)
                dict[img_n] = img_files
            else:               # if diff
                # refresh
                img_n = ''
                img_files = []
                
        # print('=========================')
        # print('DICT:', dict['0'], '\n', dict['1'])
        # print('count:', len(dict['0']), len(dict['3']))
        # print('=========================')
        return dict
        
    
    def __iter__(self):
        
        imgs_dict = self.get_dict()
        paired_imgs_list = []
        paired_indices_list = []


        # for each image background
        for i in range(len(imgs_dict)):
            
            # until we have enough pairs
            while (len(paired_indices_list) < self.n):
                img_files_arr = imgs_dict[str(i)]
                indices = np.arange(len(img_files_arr))

                # shuffle and pair images
                img_files_arr, indices = sklearn.utils.shuffle(img_files_arr, indices)
                paired_imgs = [( img_files_arr[i], img_files_arr[i+1] ) for i in range(len(img_files_arr) - 1)]
                paired_indices = [( indices[i], indices[i+1] ) for i in range(len(indices) - 1)]
                
                paired_imgs_list.append(paired_imgs)
                paired_indices_list.append(paired_indices)
                if len(paired_indices_list) > 1:
                    print(paired_indices_list)
                    paired_indices_list = reduce(lambda x,y: x+y, paired_indices_list)

            break
        # print(single_list)
        # print(paired_indices)
        # print(paired_indices[0][0])
        return iter(paired_indices_list)
        
    def __len__(self):
        return len(self.data_source)


class rain_dataset(torch.utils.data.Dataset):
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
        
        img1 = self.ids[index[0]]
        img2 = self.ids[index[1]]
        if self.real:
            input = Image.open(os.path.join(self.root, img)).convert('RGB')
            if self.transform is not None:
                input = self.transform(input)
            return input
        else:
            # input = Image.open(os.path.join(self.root, 'I', img)).convert('RGB')        # Image
            # target = Image.open(os.path.join(self.root, 'B', img)).convert('RGB')       # Background
            # target_rain = Image.open(os.path.join(self.root, 'R', img)).convert('RGB')  # Rain
            
            print('img1 name: ', img1)
            print('img2 name: ', img2)
            
            input1 = Image.open(os.path.join(self.root, img1)).convert('RGB')        # Image
            input2 = Image.open(os.path.join(self.root, img2)).convert('RGB')        # Image
            
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
            
            return input1, input2#, target1, target2, target_rain1, target_rain2
            
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
parser.add_argument('--imageSize',
                        help='the height / width of the input image to network',
                        default=256,
                        type=int)

args = parser.parse_args()

print(args.dataroot)


transform = transforms.Compose([
    # transforms.Resize(args.imageSize),
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

sampler = sampler(dataset)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batchSize,
    shuffle=False,
    num_workers=int(args.workers),
    sampler=sampler
    )


# print('====================================')
# print("LENGTH OF DATALOADER  -  ", len(dataloader))
# print('====================================')

for i, data in enumerate(dataloader):
    
    input1, input2 = data
    print(i, input1.shape)
    break