import torch
from torch import nn
import torch.nn.functional as f


class ConcatELU(nn.Module):
    """Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input
    (important for final convolution).
    """

    def forward(self, x):
        return torch.cat([f.elu(x), f.elu(-x)], dim=1)
