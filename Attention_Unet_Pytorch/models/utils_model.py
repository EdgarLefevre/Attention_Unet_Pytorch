""" Parts of the U-Net model """

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Block(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(drop))

    def forward(self, x):
        c = self.conv(x)
        return c, self.down(c)


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.Dropout(drop)
        )

    def forward(self, x):
        return self.conv(x)


class Up_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5, attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.Dropout(p=drop)
        )
        self.attention = attention
        if attention:
            self.gating = GatingSignal(in_channels, out_channels)
            self.att_gat = Attention_Gate(out_channels)

    def forward(self, x, conc):
        x1 = self.up(x)
        if self.attention:
            gat = self.gating(x)
            map, att = self.att_gat(conc, gat)
            x = torch.cat([x1, att], dim=1)
            return map, self.conv(x)
        else:
            x = torch.cat([conc, x1], dim=1)
            return None, self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class GatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(GatingSignal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)


class Attention_Gate(nn.Module):
    def __init__(self, in_channels):
        super(Attention_Gate, self).__init__()
        self.conv_theta_x = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(2, 2)
        )
        self.conv_phi_g = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x, gat):
        theta_x = self.conv_theta_x(x)
        phi_g = self.conv_phi_g(gat)
        res = torch.add(phi_g, theta_x)
        res = self.att(res)
        # print(res.size(), x.size())
        return res, torch.mul(res, x)
