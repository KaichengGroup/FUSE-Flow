import math
import torch
from torch import nn
from torch.nn import init

from .concat_elu import ConcatELU


class ChannelAttention(nn.Module):
    def __init__(self, channel, attn_red_ratio):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, math.ceil(channel * attn_red_ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(math.ceil(channel * attn_red_ratio), channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    """
    CBAM block from 'CBAM' Convolutional Block Attention Module, https://arxiv.org/pdf/1807.06521.pdf.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    """
    def __init__(self, channel, attn_red_ratio, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel, attn_red_ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class CBAMGatedConv(nn.Module):
    """
    This module applies a two-layer convolutional ResNet block with input gate and a CBAM block

    Parameters
    ----------
    c_in : int
        Number of channels of the input.
    c_hid : int
        Number of hidden dimensions we want to model (usually similar to c_in).
    """

    def __init__(self, c_in, c_hid, attn_red_ratio):
        super().__init__()

        # GatedCov
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2 * c_hid, 2 * c_in, kernel_size=1),
        )

        self.cbam = CBAMBlock(c_in, attn_red_ratio, kernel_size=3)

    def forward(self, x):
        out1 = self.net(x)
        val1, gate1 = out1.chunk(2, dim=1)
        out2 = val1 * torch.sigmoid(gate1)
        out3 = self.cbam(out2)
        return x + out3
