import torch
import torch.nn as nn
import torch.nn.parallel
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn import init


#######################################
### Network ###
#######################################

def weights_init(net, init_type):
    '''
    initialize the model's weights
    '''
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.2)
            else:
                raise NotImplementedError(f'weightt initializattion method {init_type} is not implementetd')
        elif classname.find('Norm') != -1:  # finds both batch and instance norms
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        
    # apply weight init
    net.apply(init_func)
    
def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc,
             output_nc,
             ngf,
             netG_model,
             ns,
             norm='batch',
             use_dropout=False,
             gpu_ids=[],
             iteration=0,
             padding_type='zero',
             upsample_type='transpose',
             init_type='normal'):
             
    netG = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    
    
    norm_layer = get_norm_layer(norm_type=norm)

    if netG_model == 'cascade_unet':
        netG = Generator_cascade(
            input_nc,
            output_nc,
            'unet',
            ns,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            gpu_ids=gpu_ids,
            iteration=iteration
        )
    else:
        raise Exception(f'Model name {netG_model} is not recognized')    
    
    
    # Handle multi-gpu if desired
    if len(gpu_ids) > 1:
        netG =netG.to('cuda')
        netG = nn.DataParallel(netG, list(range(len(gpu_ids))))
    elif len(gpu_ids) == 1:
        netG = netG.to(device=f"cuda:{gpu_ids[0]}")
        
    
    weights_init(netG, init_type)
    
    return netG
        
#######################################
### Layers ###
#######################################

class Generator_cascade(nn.Module):
    '''
    Params:
    input_nc
    output_nc
    base_model
    ns: number of downsampling; ns.type = array
    
    '''
    def __init__(self,
                 input_nc,
                 output_nc,
                 base_model,
                 ns,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 gpu_ids=[],
                 iteration=0,
                 padding_type='zero',
                 upsample_type='transpose'):
        super(Generator_cascade,self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.iteration = iteration
        
        if base_model == 'unet':
            self.model1 = UnetGenerator(
                input_nc,
                output_nc,
                ns[0],
                ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                gpu_ids=gpu_ids
            )
            self.model2 = UnetGenerator(
                input_nc * 2,
                output_nc,
                ns[1],
                ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                gpu_ids=gpu_ids
            )
            if self.iteration > 0:
                self.model3 = UnetGenerator(
                    input_nc * 2,
                    output_nc,
                    ns[2],
                    ngf,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    gpu_ids=gpu_ids
                )
            
    def forward(self, input1, input2=None):
        # model1 output with input 1
        x1 = self.model1(input1)
        
        # img 1
        res1 = [x1]
        # img 2
        res2 = []

        for i in range(self.iteration + 1):
            if i % 2 == 0:
                xy = torch.cat([x1, input1], 1)
                z = self.model2(xy)
                res1 += [z]
            else:
                zy = torch.cat([z, input1], 1)
                x1 = self.model3(zy)
                res1 += [x1]
        
        # for second image
        if input2 != None:                  # incase i implement single image training
            x2 = self.model1(input2)
            res2 = [x2]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = torch.cat([x2, input2], 1)
                    z = self.model2(xy)
                    res2 += [z]
                else:
                    zy = torch.cat([z, input2], 1)
                    x2 = self.model3(zy)
                    res2 += [x2]
                    
        return res1, res2


class UnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 gpu_ids=[]):
        
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # submodule
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            norm_layer = norm_layer,
            innermost=True,
            use_dropout=use_dropout
        )
        
        # building network
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout
            )
            
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, 
            unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, 
            unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, 
            unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, unet_block, 
            outermost=True, 
            norm_layer=norm_layer, 
            outermost_input_nc=input_nc
        )
        
        self.model = unet_block
    
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
            
# | downsampling | submodule | upsampling |
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 outermost_input_nc=-1):
    
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        ### downsampling ###
        
        if outermost and outermost_input_nc > 0:
            # for the last layer of the unet
            
            downconv = nn.Conv2d(
                outermost_input_nc,
                inner_nc,
                kernel_size=4,
                stride=2,
                padding=1
            )
        else:
            downconv = nn.Conv2d(
                outer_nc, 
                inner_nc, 
                kernel_size=4, 
                stride=2, 
                padding=1
            )
        
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        
        ### upsampling ###
        
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        x1 = self.model(x)
        diff_h = x.size()[2] - x1.size()[2]
        diff_w = x.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_w // 2, 
                        diff_w - diff_w // 2, 
                        diff_h // 2,
                        diff_h - diff_h // 2))

        if self.outermost:
            return x1
        else:
            return torch.cat([x1, x], 1)