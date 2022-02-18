#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import torchvision.models as models
from torch.autograd import Variable

from deform_conv_v2 import * 

class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
            
        self._block4m = nn.Sequential(
            nn.Conv2d(240, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))264+144 = 408

        self._block5m = nn.Sequential(
            nn.Conv2d(288, 144, 3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()
        

        # ### Directional Dilated Convolution ###
        # # weights1 = torch.Tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).unsqueeze(0)
        # # weights1.requires_grad = True
        # self.ddc1= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # # with torch.no_grad():
        # #    self.ddc1.weight = nn.Parameter(weights1*self.ddc1.weight)
        # self.bnd1=nn.BatchNorm2d(48)

        # # weights2 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).unsqueeze(0)
        # # weights2.requires_grad = True
        # self.ddc2= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # # with torch.no_grad():
        # #    self.ddc2.weight = nn.Parameter(weights2*self.ddc2.weight)
        # self.bnd2=nn.BatchNorm2d(48)

        # # weights3 = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).unsqueeze(0)
        # # weights3.requires_grad = True
        # self.ddc3= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # # with torch.no_grad():
        # #    self.ddc3.weight = nn.Parameter(weights2*self.ddc2.weight)
        # self.bnd3=nn.BatchNorm2d(48)

        # # weights4 = torch.Tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).unsqueeze(0)
        # # weights4.requires_grad = True
        # self.ddc4= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # # with torch.no_grad():
        # #    self.ddc4.weight = nn.Parameter(weights1*self.ddc1.weight)
        # self.bnd4=nn.BatchNorm2d(48)

        # # weights5 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).unsqueeze(0)
        # # weights5.requires_grad = True
        # self.ddc5= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # # with torch.no_grad():
        # #    self.ddc5.weight = nn.Parameter(weights2*self.ddc2.weight)
        # self.bnd5=nn.BatchNorm2d(48)

        # # weights6 = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).unsqueeze(0)
        # # weights6.requires_grad = True
        # self.ddc6= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # # with torch.no_grad():
        # #    self.ddc6.weight = nn.Parameter(weights2*self.ddc2.weight)
        # self.bnd6=nn.BatchNorm2d(48)

        # ### IRNN ### 
        self.dsc1 = DSC_Module(48, 48,alpha=0.8)
        self.bns1=nn.BatchNorm2d(48)
        self.dsc2 = DSC_Module(48, 48,alpha=0.8)
        self.bns2=nn.BatchNorm2d(48)

        # self.defc1=DeformableConv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bne1=nn.BatchNorm2d(48)
        # self.defc2=DeformableConv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bne2=nn.BatchNorm2d(48)



    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        #dsc
        #mege2=F.relu(self.bns1(self.dsc1(pool2)))
        #mege4=F.relu(self.bns2(self.dsc2(pool4)))
        #defc
        #mege2=F.relu(self.bne1(self.defc1(pool2)))
        #mege4=F.relu(self.bne2(self.defc2(pool4)))

        #ddc
        mege21=F.relu(self.bnd1(self.ddc1(pool2)))
        mege22=F.relu(self.bnd2(self.ddc2(pool2)))
        mege23=F.relu(self.bnd3(self.ddc3(pool2)))

        mege41=F.relu(self.bnd4(self.ddc4(pool4)))
        mege42=F.relu(self.bnd5(self.ddc5(pool4)))
        mege43=F.relu(self.bnd6(self.ddc6(pool4)))

        # Decoder
        upsample5 = self._block3(pool5)

        concat5 = torch.cat((upsample5, pool4, mege41, mege42, mege43), dim=1)#
        upsample4 = self._block4m(concat5)

        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)

        concat3 = torch.cat((upsample3, pool2, mege21, mege22, mege23), dim=1)#
        upsample2 = self._block5m(concat3)

        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        final = self._block6(concat1) + x
        # Final activation
        return final
