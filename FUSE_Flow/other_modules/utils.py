import os
from enum import Enum

import torch
from torch import nn

PRETRAIN_PATH = os.path.join('models', 'pretrain_unet', 'weights.pth')


ae_losses = {
    'bce': nn.BCELoss(),
    'l1': nn.L1Loss(),
    'l2': nn.MSELoss()
}


class AEInit(str, Enum):
    zero = 'zero'
    xavier = 'xavier'

    @classmethod
    def get_values(cls):
        return tuple(map(lambda c: c.value, cls))


class DequantizationType(str, Enum):
    var = 'var'
    basic = 'basic'
    none = 'none'

    @classmethod
    def get_values(cls):
        return tuple(map(lambda c: c.value, cls))


class SBPosition(str, Enum):
    first = 'first'
    middle = 'middle'
    last = 'last'

    @classmethod
    def get_values(cls):
        return tuple(map(lambda c: c.value, cls))


class AttentionType(str, Enum):
    """Types of attention mechanism implemented.
    se: Squeeze-and-Excitation https://arxiv.org/abs/1709.01507
    cbam: Convolutional Block Attention Module https://arxiv.org/abs/1807.06521
    """
    se = 'se'
    cbam = 'cbam'
    none = 'none'

    @classmethod
    def get_values(cls):
        return tuple(map(lambda c: c.value, cls))


def quantize(x, quantums):
    """Converts continuous real values to discrete integers.

    Parameters
    ----------
    x : Tensor
    quantums : int
        Number of discrete levels in x

    Returns
    -------
    y : Tensor
    """
    return torch.floor(x*quantums).clamp(min=0, max=quantums-1).to(torch.uint8)
