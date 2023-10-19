import numpy as np
import torch
import torch.nn.functional
from torch import nn


class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 Convolutional layer in Flow Step.
    Square weight matrix for simple expression of log-determinant of Jacobian matrix.

    Based on the paper:
    "Glow: Generative Flow with Invertible 1×1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Parameters
    ----------
    c_in : int
        Number of input channels.
    """

    def __init__(self, c_in):
        super().__init__()
        self.w_shape = [c_in, c_in]
        w_init = np.linalg.qr(np.random.randn(*self.w_shape))[0].astype(np.float32)
        self.register_parameter('weight', nn.Parameter(torch.Tensor(w_init)))

    def forward(self, x, reverse):
        if not reverse:
            return self._forward_flow(x)
        else:
            return self._reverse_flow(x)

    def _forward_flow(self, x):
        """Multiply input by weight.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        h = x.size(2)
        w = x.size(3)
        weight = self.weight.view(*self.w_shape, 1, 1)
        y = nn.functional.conv2d(x, weight)
        ldj = h * w * torch.slogdet(self.weight)[1]
        return y, ldj

    def _reverse_flow(self, y):
        """Multiply input by inverse of weight.

        Parameters
        ----------
        y : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        h = y.size(2)
        w = y.size(3)
        inverse_weight = torch.inverse(self.weight.double()).float().view(*self.w_shape, 1, 1)
        x = nn.functional.conv2d(y, inverse_weight)
        ldj = -1 * h * w * torch.slogdet(self.weight)[1]
        return x, ldj
