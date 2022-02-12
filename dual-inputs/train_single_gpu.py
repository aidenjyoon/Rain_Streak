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
                        required=True)

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
                        default='5',
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
                        default=5,
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

# creating placeholders
input_real1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
input_real2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
input1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
input2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
target_B1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
target_B2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
target_R1 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
target_R2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)

# to gpu            
input_real1 = input_real1.to(device)
input_real2 = input_real2.to(device)
input1 = input1.to(device)
input2 = input2.to(device)
target_B1 = target_B1.to(device)
target_B2 = target_B2.to(device)
target_R1 = target_R1.to(device)
target_R2 = target_R2.to(device)

netG.to(device)

criterion = nn.MSELoss()
criterion.to(device)

lr = 0.0001
beta = [0.9, 0.999]
optimizer = optim.Adam(netG.parameters(), 
                       lr=lr, 
                       betas=beta[:2])

print('start training...')
netG.train()
for epoch in range(args.epochs):
    print(f'start epoch {epoch}...')
    
    for i, data in enumerate(dataloader, start=1):
        if args.real:
            input_cpu = data
            category = 'real'
            
            input_real1.resize_(input_cpu.size()).copy_(input_cpu)
            input_real2.resize_(input_cpu.size()).copy_(input_cpu)
            
            if args.which_model_netG.startswith('cascade'):
                res1, res2 = netG(input_real1, input_real2)
                if len(res1) % 2 == 1:
                    output_B1, output_R1 = res1[-1], res1[-2]
                else:
                    output_B1, output_R1 = res1[-2], res1[-1]


        else:
            input_data1, input_data2, target_B_data1, target_B_data2, target_R_data1, target_R_data2 = data

            # input data
            input_real1.resize_(input_data1.size()).copy_(input_data1)
            input_real2.resize_(input_data2.size()).copy_(input_data2)
            # cleaned target images
            target_B1.resize_(target_B_data1.size()).copy_(target_B_data1)
            target_B2.resize_(target_B_data2.size()).copy_(target_B_data2)
            target_R1.resize_(target_R_data1.size()).copy_(target_R_data1)
            target_R2.resize_(target_R_data2.size()).copy_(target_R_data2)
            
            if args.which_model_netG.startswith('cascade'):
                res1, res2 = netG(input_real1, input_real2)
                
                if len(res1) % 2 == 1:
                    output_B1, output_R1 = res1[-1], res1[-2]
                else:
                    output_B1, output_R1 = res1[-2], res1[-1]
                    
                if len(res2) % 2 == 1:
                    output_B2, output_R2 = res2[-1], res2[-2]
                else:
                    output_B2, output_R2 = res2[-2], res2[-1]
                                    
            else:
                raise NotImplementedError('requires stating which model type to use')

            cuda_check1 = target_B1.is_cuda
            cuda_check2 = input_data1.is_cuda
            
            

            print("target_B1: ", target_B1.get_device())
            print("input_data1: ", input_data1.get_device())               
            
            # should net clean background
            potential_B1 = input_data1 - output_R1.to(device)
            potential_B2 = input_data2 - output_R2.to(device)

            # should = Rain
            potential_R1 = input_data1 - output_B1.to(device)
            potential_R2 = input_data2 - output_B2.to(device)
            
 

            # mse error
            errB1 = criterion(output_B1, target_B1)
            errB2 = criterion(output_B2, target_B2)
            errR1 = criterion(output_R1, target_R1)
            errR2 = criterion(output_R2, target_R2)
            errPB1 = criterion(potential_B1, target_B1)
            errPB2 = criterion(potential_B2, target_B2)
            errPR1 = criterion(potential_R1, target_R1)
            errPR2 = criterion(potential_R2, target_R2)

            
            # loss
            loss = errB1 + errB2 + errR1 + errR2 + errPB1 + errPB2 + errPR1 + errPR2
            loss = loss / .0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Output training stats
        if i % 500 == 0:
            print(f'{i}/{len(dataloader)}\tLoss_B1: {errB1}/tLoss_R1: {errR1}\tLoss_B2: {errB2}/tLoss_R2: {errR2}\n')

        # save trained image
        if i % 10000 == 0:
            if args.n_outputs == 0 or i <= args.n_outputs:
                save_image(output_B1 / 2 + 0.5, f'../trained_imgs/{args.outf}/B1_{i}.png')
                if args.which_model_netG.startswith('cascade'):
                    save_image(output_R1 / 2 + 0.5, f'../trained_imgs/{args.outf}/R1_{i}.png')
            
                save_image(output_B2 / 2 + 0.5, f'../trained_imgs/{args.outf}/B2_{i}.png')
                if args.which_model_netG.startswith('cascade'):
                    save_image(output_R2 / 2 + 0.5, f'../trained_imgs/{args.outf}/R2_{i}.png')
    
    # save model
    torch.save(netG, f'./model_save/epoch{epoch}_model_save.pth')