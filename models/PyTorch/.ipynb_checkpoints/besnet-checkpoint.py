# Original paper: https://link.springer.com/chapter/10.1007%2F978-3-030-00934-2_26

import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_components import BaseConv, MDPBlock, BDPBlock, DownBlock

class BESNet(nn.Module):
    def __init__(self, input_channel):
        """ Construct a BESNet
        Parameters:
            input_channel (int): input dimension (1 for B/W, 2 for RGB)
        """
        super(BESNet, self).__init__()
        self.input_channel = input_channel
        
        # initial block
        self.init_block = BaseConv(input_channel, 64, [32])
        # Down Blocks
        self.down_block1 = DownBlock(64, 128)
        self.down_block2 = DownBlock(128, 256)
        self.down_block3 = DownBlock(256, 512)
        self.down_block4 = DownBlock(512, 512, [1024])
        # Up Blocks for Boundary Decoding Path (BDP)
        self.bdp_block1 = BDPBlock(512, 256)
        self.bdp_block2 = BDPBlock(256, 128)
        self.bdp_block3 = BDPBlock(128, 64)
        self.bdp_block4 = BDPBlock(64, 64)
        self.bdp = nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        # Up Blocks for Main Decoding Path (MDP)
        self.mdp_block1 = MDPBlock(512, 256, 512)
        self.mdp_block2 = MDPBlock(256 + 256, 128, 256)
        self.mdp_block3 = MDPBlock(128 + 128, 64, 128)
        self.mdp_block4 = MDPBlock(64 + 64, 64, 64)
        self.mdp = nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        
    def forward(self, x):
        # Encoding
        x_skip_enp1 = self.init_block(x)
        x_skip_enp2 = self.down_block1(x_skip_enp1)
        x_skip_enp3 = self.down_block2(x_skip_enp2)
        x_skip_enp4 = self.down_block3(x_skip_enp3)
        x = self.down_block4(x_skip_enp4)
        # BDP
        x_skip_bdp1 = self.bdp_block1(x_skip_enp4, x)
        x_skip_bdp2 = self.bdp_block2(x_skip_enp3, x_skip_bdp1)
        x_skip_bdp3 = self.bdp_block3(x_skip_enp2, x_skip_bdp2)
        x_skip_bdp4 = self.bdp_block4(x_skip_enp1, x_skip_bdp3)
        x_bdp = self.bdp(x_skip_bdp4)
        x_bdp = nn.Sigmoid()(x_bdp)
        # MDP
        x = self.mdp_block1(x_skip_enp4, x)
        x = torch.cat([x_skip_bdp1, x], dim = 1)
        x = self.mdp_block2(x_skip_enp3, x)
        x = torch.cat([x_skip_bdp2, x], dim = 1)
        x = self.mdp_block3(x_skip_enp2, x)
        x = torch.cat([x_skip_bdp3, x], dim = 1)
        x = self.mdp_block4(x_skip_enp1, x)
        x_mdp = self.mdp(x)
        x_mdp = nn.Sigmoid()(x_mdp)
        return x_bdp, x_mdp