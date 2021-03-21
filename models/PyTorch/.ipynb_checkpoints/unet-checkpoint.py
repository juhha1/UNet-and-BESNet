# Original paper: U-Net: Convolutional Networks for Biomedical Image Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_components import BaseConv, DownBlock, UpBlock

class UNet(nn.Module):
    def __init__(self, input_channel, skip_method = 'cat', use_bilinear = False):
        """ Construct a UNet model (have similar structure as the original paper)
        Parameters:
            input_channel (int): input dimension (1 for B/W, 2 for RGB)
            skip_method (str): "sum" for summation skip connection, 'cat' for concatenating skip connection
            use_bilinear (bool): True for bilinear upsampling, False for convolutional transpose for upsampling
        """
        super(UNet, self).__init__()
        self.input_channel = input_channel
        self.skip_method = skip_method
        self.use_bilinear = use_bilinear
        
        # initial block
        self.init_block = BaseConv(input_channel, 64)
        # Down Blocks
        self.down_block1 = DownBlock(64, 128)
        self.down_block2 = DownBlock(128, 256)
        self.down_block3 = DownBlock(256, 512)
        self.down_block4 = DownBlock(512, 512, [1024])
        # Up Blocks
        self.up_block1 = UpBlock(512, 256, skip_method = skip_method, use_bilinear = use_bilinear)
        self.up_block2 = UpBlock(256, 128, skip_method = skip_method, use_bilinear = use_bilinear)
        self.up_block3 = UpBlock(128, 64, skip_method = skip_method, use_bilinear = use_bilinear)
        self.up_block4 = UpBlock(64, 64, skip_method = skip_method, use_bilinear = use_bilinear)
        # Output
        self.output_layer = nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        
    def forward(self, x):
        x_skip1 = self.init_block(x)
        # Down
        x_skip2 = self.down_block1(x_skip1)
        x_skip3 = self.down_block2(x_skip2)
        x_skip4 = self.down_block3(x_skip3)
        x = self.down_block4(x_skip4)
        # Up
        x = self.up_block1(x_skip4, x)
        x = self.up_block2(x_skip3, x)
        x = self.up_block3(x_skip2, x)
        x = self.up_block4(x_skip1, x)
        # Output
        x = self.output_layer(x)
        x = nn.Sigmoid()(x)
        return x