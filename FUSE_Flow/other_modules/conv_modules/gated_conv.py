import torch
from torch import nn

from .concat_elu import ConcatELU


class GatedConv(nn.Module):
    """This module applies a two-layer convolutional ResNet block with input gate

    Parameters
    ----------
    c_in : int
        Number of channels of the input.
    c_hid : int
        Number of hidden dimensions we want to model (usually similar to c_in).
    """

    def __init__(self, c_in, c_hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2 * c_hid, 2 * c_in, kernel_size=1),
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)
