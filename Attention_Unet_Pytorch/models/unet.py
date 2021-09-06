""" Full assembly of the parts to form the complete network """

import torch.nn as nn

import Attention_Unet_Pytorch.models.utils_model as um


class Unet(nn.Module):
    def __init__(self, filters, drop_r=0.5, attention=True):
        super(Unet, self).__init__()
        self.down1 = um.Down_Block(1, filters)
        self.down2 = um.Down_Block(filters, filters * 2, drop_r)
        self.down3 = um.Down_Block(filters * 2, filters * 4, drop_r)
        self.down4 = um.Down_Block(filters * 4, filters * 8, drop_r)

        self.bridge = um.Bridge(filters * 8, filters * 16, drop_r)

        self.up1 = um.Up_Block(filters * 16, filters * 8, drop_r, attention)
        self.up2 = um.Up_Block(filters * 8, filters * 4, drop_r, attention)
        self.up3 = um.Up_Block(filters * 4, filters * 2, drop_r, attention)
        self.up4 = um.Up_Block(filters * 2, filters, drop_r, attention)

        self.outc = um.OutConv(filters, 1)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        _, x = self.up1(bridge, c4)
        _, x = self.up2(x, c3)
        att, x = self.up3(x, c2)
        _, x = self.up4(x, c1)
        mask = self.outc(x)
        return mask, att
