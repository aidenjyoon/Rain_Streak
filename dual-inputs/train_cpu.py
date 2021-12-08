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
from vutil import save_image

from dataset import rain_dataset
import network



parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
    '--batchSize', type=int, default=8, help='input batch size')
parser.add_argument(
    '--which_model_netG',
    type=str,
    default='cascade_unet',
    help='selects model to use for netG')
parser.add_argument(
    '--ns', type=str, default='5', help='number of blocks for each module')
parser.add_argument(
    '--netG', default='', help="path to netG (to continue training)")
parser.add_argument(
    '--norm',
    type=str,
    default='batch',
    help='instance normalization or batch normalization')
parser.add_argument(
    '--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument(
    '--imageSize',
    type=int,
    default=256,
    help='the height / width of the input image to network')
parser.add_argument(
    '--outf',
    default='.',
    help='folder to output images and model checkpoints')
parser.add_argument(
    '--outf1',
    default='.',
    help='folder to output images and model checkpoints')
parser.add_argument(
    '--outf2',
    default='.',
    help='folder to output images and model checkpoints')
parser.add_argument('--real', action='store_true', help='test real images')
parser.add_argument(
    '--iteration', type=int, default=0, help='number of iterative updates')
parser.add_argument(
    '--n_outputs', type=int, default=0, help='number of images to save')


opt = parser.parse_args()

# number of downsampling into array
str_ids = opt.ns.split(',')
opt.ns = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.ns.append(id)
        
try:
    os.makedirs(opt.outf)
except OSError:
    pass

nc = 3
ngf = 64
device = torch.device('cpu')

netG = network.define_G(nc, nc, ngf, opt.which_model_netG, opt.ns, opt.norm,
                        opt.use_dropout, [], opt.iteration)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

transform = transforms.Compose([
    # transforms.Scale(opt.imageSize),
    # transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = rain_dataset(
    opt.dataroot,
    transform=transform,
    target_transform=transform,
    rain_transform=transform,
    real=opt.real)
assert dataset

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))



input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input1 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input2 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_real = input_real.to(device)
input1 = input1.to(device)
input2 = input2.to(device)
netG.to(device)
netG.eval()

criterion = nn.MSELoss()
criterion.to(device)


lr = 0.0001
beta1 = 0.9
beta2 = 0.999
optimizer = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

print('this is device ', device)

for i, data in enumerate(dataloader, 1):
    if opt.real:
        input_cpu = data
        category = 'real'
        
        input_real.resize_(input_cpu.size()).copy_(input_cpu)
        if opt.which_model_netG.startswith('cascade'):
            res = netG(input_real)
            if len(res) % 2 == 1:
                output_B1, output_R1 = res[-1], res[-2]
            else:
                output_B2, output_R2 = res[-2], res[-1]
        else:
            # don't use atm
            output_B = netG(input)

    else:
        input_cpu1, input_cpu2, target_B_cpu1, target_B_cpu2, target_R_cpu1, target_R_cpu2 = data
        category = 'test'
        
        input1.resize_(input_cpu1.size()).copy_(input_cpu1)
        input2.resize_(input_cpu2.size()).copy_(input_cpu2)
        if opt.which_model_netG.startswith('cascade'):
            res1, res2 = netG(input1, input2)
            
            # Output training stats

            if len(res1) % 2 == 1:
                output_B1, output_R1 = res1[-1], res1[-2]
            else:
                output_B1, output_R1 = res1[-2], res1[-1]
                
            if len(res2) % 2 == 1:
                output_B2, output_R2 = res2[-1], res2[-2]
            else:
                output_B2, output_R2 = res2[-2], res2[-1]
        else:
            # don't use atm
            output_B = netG(input)
        
        target_B1 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        target_B2 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        target_R1 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        target_R2 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

        target_B1.resize_(target_B_cpu1.size()).copy_(target_B_cpu1)
        target_B2.resize_(target_B_cpu2.size()).copy_(target_B_cpu2)
        target_R1.resize_(target_R_cpu1.size()).copy_(target_R_cpu1)
        target_R2.resize_(target_R_cpu2.size()).copy_(target_R_cpu2)

        
        target_B1 = target_B1.to(device)
        target_B2 = target_B2.to(device)
        target_R1 = target_R1.to(device)
        target_R2 = target_R2.to(device)
        target_B1.resize_(target_B_cpu1.size()).copy_(target_B_cpu1)
        target_B2.resize_(target_B_cpu2.size()).copy_(target_B_cpu2)
        target_R1.resize_(target_R_cpu1.size()).copy_(target_R_cpu1)
        target_R2.resize_(target_R_cpu2.size()).copy_(target_R_cpu2)

        # error
        errB1 = criterion(output_B1, target_B1)
        errB2 = criterion(output_B2, target_B2)
        errR1 = criterion(output_R1, target_R1)
        errR2 = criterion(output_R2, target_R2)

        errN = errB1 + errB2 + errR1 + errR2
        
        errN.backward()

        optimizer.step()
        
    # Output training stats
    if i % 50 == 0:
        print(f'{i}/{len(dataloader)}\tLoss_B1: {errB1}/tLoss_R1: {errR1}\tLoss_B2: {errB2}\tLoss_R2: {errR2}')

    # save trained image
    if i % 1000 == 0:
        if opt.n_outputs == 0 or i <= opt.n_outputs:
            save_image(output_B1 / 2 + 0.5, f'../trained_imgs/{opt.outf}/B1_{i}.png')
            if opt.which_model_netG.startswith('cascade'):
                save_image(output_R1 / 2 + 0.5, f'../trained_imgs/{opt.outf}/R1_{i}.png')

            save_image(output_B2 / 2 + 0.5, f'../trained_imgs/{opt.outf}/B2_{i}.png')
            if opt.which_model_netG.startswith('cascade'):
                save_image(output_R2 / 2 + 0.5, f'../trained_imgs/{opt.outf}/R2_{i}.png')
            
            
            
            # # we don't want dedicated directories for images
            # if opt.outf1 == '.' and opt.outf2 == '.':
            #     save_image(output_B1 / 2 + 0.5, f'../trained_imgs/{opt.outf}/B1_{i}.png')
            #     if opt.which_model_netG.startswith('cascade'):
            #         save_image(output_R1 / 2 + 0.5, f'../trained_imgs/{opt.outf}/R1_{i}.png')

            #     save_image(output_B2 / 2 + 0.5, f'../trained_imgs/{opt.outf}/B2_{i}.png')
            #     if opt.which_model_netG.startswith('cascade'):
            #         save_image(output_R2 / 2 + 0.5, f'../trained_imgs/{opt.outf}/R2_{i}.png')

            # else:    
            #     save_image(output_B1 / 2 + 0.5, f'../trained_imgs/{opt.outf1}/B1_{i}.png')
            #     if opt.which_model_netG.startswith('cascade'):
            #         save_image(output_R1 / 2 + 0.5, f'../trained_imgs/{opt.outf1}/R1_{i}.png')

            #     save_image(output_B2 / 2 + 0.5, f'../trained_imgs/{opt.outf2}/B2_{i}.png')
            #     if opt.which_model_netG.startswith('cascade'):
            #         save_image(output_R2 / 2 + 0.5, f'../trained_imgs/{opt.outf2}/R2_{i}.png')