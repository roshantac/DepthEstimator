""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        y1 = self.inc(x) #------------
        x2 = self.down1(x1)
        y2 = self.down1(x1) #------------
        x3 = self.down2(x2)
        y3 = self.down2(y2) #------------ 
        x4 = self.down3(x3)
        y4 = self.down3(y3) #------------
        x5 = self.down4(x4)
        y5 = self.down5(y4) #------------
        x = self.up1(x5, x4)
        y = self.up1(y5, y4) #------------
        x = self.up2(x, x3)
        y = self.up2(y, y3) #------------
        x = self.up3(x, x2)
        y = self.up1(y, y2) #------------
        x = self.up4(x, x1)
        y = self.up1(y, y1) #------------
        rt_depth = self.outc(x)
        rt_mask = self.outc(y) #------------
        return rt_depth, rt_mask
