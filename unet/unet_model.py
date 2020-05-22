""" Full assembly of the parts to form the complete network """
#unet_model
import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
    # mask
        self.inc_m = DoubleConv(n_channels, 16)
        self.down1_m = Down(16,32)
        self.down2_m = Down(32, 64)
        self.down3_m = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4_m = Down(128, 256 // factor)
        self.up1_m = Up(256, 128 // factor, bilinear)
        self.up2_m = Up(128, 64 // factor, bilinear)
        self.up3_m = Up(64, 32 // factor, bilinear)
        self.up4_m = Up(32, 16, bilinear)
        self.outc_m = OutConv(16, n_classes)
    # depth
        self.inc_d = DoubleConv(n_channels, 32)
        self.down1_d = Down(32, 64)
        self.down2_d = Down(64, 128)
        self.down3_d = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4_d = Down(256, 512 // factor)
        self.up1_d = Up(512, 256 // factor, bilinear)
        self.up2_d = Up(256, 128 // factor, bilinear)
        self.up3_d = Up(128, 64 // factor, bilinear)
        self.up4_d = Up(64, 32, bilinear)
        self.outc_d = OutConv(32, n_classes)

    def forward(self, data):
        dat1 = data['fgbg']
        dat2 = data['bg']
        x = torch.cat([dat1,dat2], dim = 1)
        y = x
        x1 = self.inc_d(x)
        x2 = self.down1_d(x1)
        x3 = self.down2_d(x2)
        x4 = self.down3_d(x3)
        x5 = self.down4_d(x4)
        x = self.up1_d(x5, x4)
        x = self.up2_d(x, x3)
        x = self.up3_d(x, x2)
        x = self.up4_d(x, x1)
        rt_depth = self.outc_d(x)

        y1 = self.inc_m(y) 
        y2 = self.down1_m(y1) 
        y3 = self.down2_m(y2)  
        y4 = self.down3_m(y3) 
        y5 = self.down4_m(y4) 
        y = self.up1_m(y5, y4) 
        y = self.up2_m(y, y3) 
        y = self.up3_m(y, y2) 
        y = self.up4_m(y, y1) 
        rt_mask = self.outc_m(y) 
        return rt_depth, rt_mask 
