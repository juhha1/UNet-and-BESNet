import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, middle_channels = [], kernel_size = 3, padding = 1):
        super().__init__()
        if middle_channels: middle_channels.append(out_channel)
        list_channels = [in_channel]
        for middle_channel in middle_channels:
            list_channels.append(middle_channel)
        list_channels.append(out_channel)
        layers = []
        for in_c, out_c in zip(list_channels[:-1], list_channels[1:]):
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size = kernel_size, padding = padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace = True)
            ])
        self.base_conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.base_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, middle_channels = [], kernel_size = 3, padding = 1):
        super().__init__()
        self.base_conv = BaseConv(in_channel, out_channel, middle_channels, kernel_size, padding)
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.downsample(x)
        return self.base_conv(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, middle_channels = [], kernel_size = 3, padding = 1, use_bilinear = True, skip_method = 'sum'):
        super().__init__()
        self.skip_method = skip_method
        if use_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        else:
            self.upsample = nn.ConvTranspose2d(in_channel, in_channel, kernel_size = 2, stride = 2)
        if skip_method == 'sum':
            self.base_conv = BaseConv(in_channel, out_channel, middle_channels, kernel_size, padding)
        else:
            self.base_conv = BaseConv(in_channel*2, out_channel, middle_channels, kernel_size, padding)
    def forward(self, x_skip, x):
        x = self.upsample(x)
        h_pad = x_skip.size()[2] - x.size()[2]
        w_pad = x_skip.size()[3] - x.size()[3]
        x = F.pad(x, [h_pad // 2, h_pad - h_pad // 2,
                      w_pad // 2, w_pad - w_pad // 2])
        x = x_skip + x if self.skip_method == 'sum' else torch.cat([x_skip, x], dim = 1)
        x = self.base_conv(x)
        return x
    
class BDPBlock(nn.Module):
    def __init__(self, in_channel,  out_channel, middle_channels = [], kernel_size = 3, padding = 1, use_bilinear = False):
        super().__init__()
        if use_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        else:
            self.upsample = nn.ConvTranspose2d(in_channel, in_channel, kernel_size = 2, stride = 2)
        self.base_conv = BaseConv(in_channel, out_channel, middle_channels, kernel_size, padding)
    def forward(self, x_skip, x):
        x = self.upsample(x)
        h_pad = x_skip.size()[2] - x.size()[2]
        w_pad = x_skip.size()[3] - x.size()[3]
        x = F.pad(x, [h_pad // 2, h_pad - h_pad // 2,
                      w_pad // 2, w_pad - w_pad // 2])
        x = x_skip + x
        x = self.base_conv(x)
        return x
    
class MDPBlock(nn.Module):
    def __init__(self, in_channel,  out_channel, upsample_channel, kernel_size = 3, padding = 1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, upsample_channel, kernel_size = 2, stride = 2)
        self.base_conv = BaseConv(upsample_channel, out_channel, [], kernel_size, padding)
    def forward(self, x_skip, x, skip = True):
        x = self.upsample(x)
        h_pad = x_skip.size()[2] - x.size()[2]
        w_pad = x_skip.size()[3] - x.size()[3]
        x = F.pad(x, [h_pad // 2, h_pad - h_pad // 2,
                      w_pad // 2, w_pad - w_pad // 2])
        x = x_skip + x
        x = self.base_conv(x)
        return x