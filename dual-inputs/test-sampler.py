# import cv2
# from skimage import io
# from skimage.util import img_as_float
# import sklearn
# import numpy as np
# import argparse
# import os
# import random
# import time
# from collections import OrderedDict
# import itertools
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
# from torch.autograd import Variable

        


# class mySampler(torch.utils.data.Sampler):
#     def __init__(self, data_source):
#         self.data_source = data_source
        
#         # names and idicies
#         self.indices = torch.arange(len(self.data_source))
#         self.img_names = self.data_source.ids # ie. img_n112_0deg_rc200_2.jpg

#         # number of desired dataset length
#         self.n = 1

#     def get_dict(self):
#         '''
#         create a dictionary of synthesized images
#         key - which image was used for synthesis in string integers ('0', '1', '2',..)
#         value - list of image file names synthesized with key image
#             (ie. if key = 1, value = [img_n1_0deg_rc200_2.jpg, img_n1_33deg_rc200_2.jpg])
#         '''
        
#         # go through img-names and create a dictionary of item: count
#         img_n = ''
#         img_files = []
#         dict = {}
#         for name in self.img_names:
#             temp = ''
#             # iterate through .jpg name
#             for idx, char in enumerate(name):
#                 # check if at img_n
#                 if idx > 4:
#                     # check if char isn't surpassing n..#.._
#                     if char != '_':
#                         # get the img_n
#                         temp = temp + char
#                     else:
#                         break
    
#             if img_n == '':
#                 img_n = temp
#                 dict[img_n] = 0
                
#             # check if img_n changed or not
#             if img_n == temp:   # if same
#                 img_files.append(name)
#                 dict[img_n] = img_files
#             else:               # if diff
#                 # refresh
#                 img_n = ''
#                 img_files = []
#         # print('=========================')
#         # print('DICT:', dict['0'], '\n', dict['1'])
#         # print('count:', len(dict['0']), len(dict['3']))
#         # print('=========================')
#         return dict
        
    
#     def __iter__(self):
        
#         imgs_dict = self.get_dict()
#         paired_imgs_list = []
#         paired_indices_list = []
#         prev_imgs_count = 0

#         print('==========================')
#         print('THIS IS DICTIONARY LENGTH', len(imgs_dict))
#         print('==========================')
#         # for each image background
#         for i in range(len(imgs_dict)):
#             n_pairs = 0
#             self.n = len(imgs_dict[str(i)])
            
            
#             # until we have enough pairs
#             while (n_pairs < self.n):
                
#                 print('LENGTH OF LIST: ', len(paired_indices_list))
#                 img_files_arr = imgs_dict[str(i)]
#                 indices = np.arange(prev_imgs_count, len(img_files_arr) + prev_imgs_count)
#                 print('==========================')
#                 print('IMG-FILES-ARR', img_files_arr)
#                 print('indices: ', indices)
#                 print('prev_imgs_count: ', prev_imgs_count)
#                 print('==========================')

#                 # shuffle and pair images
#                 img_files_arr, indices = sklearn.utils.shuffle(img_files_arr, indices)
#                 paired_imgs = [( img_files_arr[i], img_files_arr[i+1] ) for i in range(len(img_files_arr) - 1)]
#                 paired_indices = [( indices[i], indices[i+1] ) for i in range(len(indices) - 1)]
                
                
#                 n_pairs += len(paired_indices)
#                 paired_imgs_list += paired_imgs
#                 paired_indices_list += paired_indices
                
#             prev_imgs_count += self.n
                                                 
#         # print(paired_indices_list)
#         return iter(paired_indices_list)
        
#     def __len__(self):
#         return len(self.data_source)


# class rain_dataset(torch.utils.data.Dataset):
#     def __init__(self,
#                  root,
#                  transform=None,
#                  target_transform=None,
#                  rain_transform=None,
#                  real=False):
#         self.root = root
#         self.transform = transform
#         self.target_transform = target_transform
#         self.rain_transform = rain_transform
#         self.real = real
        
#         if real:
#             self.ids = sorted(os.listdir(root))
#         else:
#             self.ids = sorted(os.listdir(os.path.join(root)))
        
#     def __getitem__(self, index):
                
#         img1 = self.ids[index[0]]
#         img2 = self.ids[index[1]]
#         if self.real:
#             input = Image.open(os.path.join(self.root, img1)).convert('RGB')
#             if self.transform is not None:
#                 input = self.transform(input)
#             return input
#         else:
#             print('img1 name: ', img1)
#             print('img2 name: ', img2)
            
#             input1 = Image.open(os.path.join(self.root, img1)).convert('RGB')        # Image
#             input2 = Image.open(os.path.join(self.root, img2)).convert('RGB')        # Image
            
#             extracted_rain1 = img_as_float(input1) - img_as_float(input2)
#             extracted_rain2 = img_as_float(input2) - img_as_float(input1)
            
#             rain1_mask = extracted_rain1 >= 0
#             rain2_mask = extracted_rain2 >= 0
#             rain1 = extracted_rain1 * rain1_mask
#             rain2 = extracted_rain2 * rain2_mask
#             background1 = img_as_float(input1) - img_as_float(rain1)
#             background2 = img_as_float(input2) - img_as_float(rain2)
            
#             if self.transform is not None:
#                 input1 = self.transform(input1)
#                 input2 = self.transform(input2)
#             if self.target_transform is not None:
#                 target1 = self.target_transform(background1)
#                 target2 = self.target_transform(background2)
#             if self.rain_transform is not None:
#                 target_rain1 = self.rain_transform(rain1)
#                 target_rain2 = self.rain_transform(rain2)
            
#             return input1, input2, target1, target2, target_rain1, target_rain2
            
#     def __len__(self):
#         return len(self.ids)
    
    

# parser = argparse.ArgumentParser()

# parser.add_argument('--dataroot', 
#                         help='path to dataset',
#                         default='../rain_generator/train_horse/',
#                         type=str)
# parser.add_argument('--real', 
#                         help='test real images',
#                         action='store_true')
# parser.add_argument('--workers', 
#                         help='number of data loading workers', 
#                         default=2,
#                         type=int)
# parser.add_argument('--batchSize', 
#                         help='input batch size',
#                         default=8,
#                         type=int)
# parser.add_argument('--imageSize',
#                         help='the height / width of the input image to network',
#                         default=256,
#                         type=int)
# parser.add_argument('--gpu',
#                         help='gpu',
#                         default='0',
#                         type=str)

# args = parser.parse_args()

# print(args.dataroot)


# transform = transforms.Compose([
#     transforms.Resize(256),
#     # transforms.CenterCrop(opt.imageSize),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# dataset = rain_dataset(
#     args.dataroot,
#     transform=transform,
#     target_transform=transform,
#     rain_transform=transform,
#     real=args.real)
# assert dataset

# mySampler = mySampler(dataset)

# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=args.batchSize,
#     shuffle=False,
#     num_workers=int(args.workers),
#     sampler=mySampler
#     )

# if len(args.gpu) > 1:
#     # number of gpu into array
#     str_ids = args.gpu.split(',')
#     args.gpu = []
#     for str_id in str_ids:
#         id = int(str_id)
#         if id >= 0:
#             args.gpu.append(id)
# else:
#     args.gpu = [int(args.gpu)]
    
# print("THIS PRINTS GPU: ", args.gpu)
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', index=args.gpu[0])
    
# input_real1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input_real2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input_real1 = input_real1.to(device)
# input_real2 = input_real2.to(device)
# input1 = input1.to(device)
# input2 = input2.to(device)

# # print('====================================')
# # print("LENGTH OF DATALOADER  -  ", len(dataloader))
# # print('====================================')

# for i, data in enumerate(dataloader):
    
#     input1, input2, target1, target2, target_rain1, target_rain2 = data
#     # print(i, input1.shape, input2.shape, target1.shape, target2.shape, target_rain1.shape, target_rain2.shape)

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

from dataset import rain_dataset, mySampler
import network
from vutil import save_image



parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', 
                        help='path to dataset',
                        default='../rain_generator/train_horse',
                        required=False)

parser.add_argument('--workers', 
                        help='number of data loading workers', 
                        default=1,
                        type=int)
parser.add_argument('--batchSize', 
                        help='input batch size',
                        default=2,
                        type=int)
parser.add_argument('--which_model_netG',
                        help='selects model to use for netG',
                        default='cascade_unet',
                        type=str)
parser.add_argument('--ns',
                        help='number of blocks for each module',
                        default='5,5,5',
                        type=str)
parser.add_argument('--netG',  
                        help="path to netG (to continue training)",
                        default='')
parser.add_argument('--norm',
                        help='instance normalization or batch normalization',
                        default='batch',
                        type=str)
parser.add_argument('--use_dropout',
                        help='use dropout for the generator',
                        action='store_true')
parser.add_argument('--imageSize',
                        help='the height / width of the input image to network',
                        default=256,
                        type=int)
parser.add_argument('--epochs',
                        help='number of epochs for training',
                        default=1,
                        type=int)
parser.add_argument('--gpu',
                        help='cuda:_x_ number',
                        default='0',
                        type=str)

parser.add_argument('--outf',
                        help='folder to output images and model checkpoints',
                        default='.')
parser.add_argument('--outf1',
                        help='folder to output images and model checkpoints',
                        default='.')
parser.add_argument('--outf2',
                        help='folder to output images and model checkpoints',
                        default='.')

parser.add_argument('--real', 
                        help='test real images',
                        action='store_true')
parser.add_argument('--iteration', 
                        help='number of iterative updates',
                        default=0,
                        type=int)
parser.add_argument('--n_outputs',
                        help='number of images to save',
                        default=0,
                        type=int)


args = parser.parse_args()

# number of downsampling into array
str_ids = args.ns.split(',')
args.ns = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.ns.append(id)
        
if len(args.gpu) > 1:
    # number of gpu into array
    str_ids = args.gpu.split(',')
    args.gpu = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu.append(id)
else:
    args.gpu = [int(args.gpu)]

try:
    os.makedirs(args.outf)
except OSError:
    pass

nc = 3
ngf = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', index=args.gpu[0])

netG = network.define_G(nc, nc, ngf, args.which_model_netG, args.ns, args.norm,
                        args.use_dropout, args.gpu, args.iteration)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))

transform = transforms.Compose([
    # transforms.Resize(args.imageSize),
    # transforms.CenterCrop(args.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = rain_dataset(
    args.dataroot,
    transform=transform,
    target_transform=transform,
    rain_transform=transform,
    real=args.real
)
assert dataset


mySampler = mySampler(dataset)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batchSize,
    shuffle=False,
    num_workers=int(args.workers),
    sampler=mySampler
)

# input_real1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input_real2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
# input_real1 = input_real1.to(device)
# input_real2 = input_real2.to(device)
# input1 = input1.to(device)
# input2 = input2.to(device)
# netG.to(device)

# criterion = nn.MSELoss()
# criterion.to(device)

# lr = 0.0001
# beta = [0.9, 0.999]
# optimizer = optim.Adam(netG.parameters(), 
#                        lr=lr, 
#                        betas=beta[:2])

print('start training...')
# netG.train()
for epoch in range(args.epochs):
    print(f'start epoch {epoch}...')
    
    for i, data in enumerate(dataloader, start=1):
        print(i, len(dataloader))