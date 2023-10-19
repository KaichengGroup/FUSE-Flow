import math
import torch
from torch import nn

from .concat_elu import ConcatELU


class SEGatedConv(nn.Module):
    """
    This module applies a two-layer convolutional ResNet block with input gate and a Squeeze-and-Excitation block
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters
    ----------
    c_in : int
        Number of channels of the input.
    c_hid : int
        Number of hidden dimensions we want to model (usually similar to c_in).
    attn_red_ratio : float  # default 16
        Minimum value = 1, Maximum value = c_in, set reduction from 1 to c_in using attn_red_ratio
        Smaller attn_red_ratio --> fewer Parameters
        Hyperparameter to vary capacity and computational cost of SE blocks in the network.
    """
    def __init__(self, c_in, c_hid, attn_red_ratio):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2 * c_hid, 2 * c_in, kernel_size=1),
        )

        # SE Block: global pooling, fully connected layer, RELU, fully connected layer, Sigmoid
        channels = c_in
        mid_channels = math.ceil(channels * attn_red_ratio)

        self.SEBlock = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, groups=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.net(x)
        val1, gate1 = out1.chunk(2, dim=1)
        out2 = val1 * torch.sigmoid(gate1)
        out3 = self.SEBlock(out2)
        out4 = out2*out3
        return x + out4
