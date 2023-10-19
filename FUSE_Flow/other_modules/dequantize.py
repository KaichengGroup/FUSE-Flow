import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from FUSE_Flow.other_modules.utils import quantize, DequantizationType


class Dequantization(nn.Module):
    """Convert discrete distribution into a continuous distribution by adding noise.
    Prevents a degenerate solution that places all probability mass on discrete
    data points (Uria et al., 2013).
    Adds noise from complex distribution to better approximate smooth continuous distribution
    instead of simple uniform distribution.

    Based on the paper:
    "Flow++: Improving Flow-Based Generative Models with
    Variational Dequantization and Architecture Design"
    by Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel
    (https://arxiv.org/abs/1902.00275).

    Parameters
    ----------
    alpha : float
        Small constant that is used to scale the original input.
        Prevents dealing with values very close to 0 and 1 when inverting the sigmoid.
        Breaks invertibility. Set to 0 to maintain invertibility.
    quantums : int
        Number of possible discrete values (usually 256 for 8-bit image).
    """

    def __init__(self, flow, downsample, perturbation_type, quantums, alpha=1e-5):
        super().__init__()
        self.alpha = alpha
        self.quantums = quantums
        self.perturbation_type = perturbation_type
        if perturbation_type == DequantizationType.var:
            self.flow = flow
            self.downsample = downsample

    def forward(self, x, reverse=False):
        if not reverse:
            return self._forward_flow(x)
        else:
            return self._reverse_flow(x)

    def _forward_flow(self, x):
        """Forward flow through each layer.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        ldj_flow = torch.zeros(x.shape[0], device=x.device)
        if self.perturbation_type == 'none':
            z = torch.zeros_like(x).detach()
        else:
            z = torch.rand_like(x).detach()
            if self.perturbation_type == 'var':
                z, ldj_flow = self._apply_flow(z, x)
        x, ldj_deq = self._dequantize(x, z)
        y, ldj_sig = self._sigmoid(x, reverse=True)

        ldj = ldj_deq + ldj_sig + ldj_flow
        return y, ldj

    def _reverse_flow(self, y):
        """Forward flow through each layer.

        Parameters
        ----------
        y : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        y, ldj_sig = self._sigmoid(y, reverse=False)
        x, ldj_deq = self._quantize(y)
        ldj = ldj_sig + ldj_deq
        return x, ldj

    def _apply_flow(self, z, x):
        """Add noise generated from complex distribution to input.
        Scale result to [0, 1].

        Parameters
        ----------
        z : torch.Tensor
        x : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        """
        u = (x / (self.quantums - 1)) * 2 - 1  # scale input to [-1, 1]
        u = self.downsample(u)

        z, ldj_log = self._sigmoid(z, reverse=True)  # transform to [-infinity,+infinity]
        z, ldj_flow = self.flow(z, u)  # estimate posterior
        z, ldj_sig = self._sigmoid(z, reverse=False)  # transform back to [0, 1]

        ldj = ldj_log + ldj_flow + ldj_sig
        return z, ldj

    def _dequantize(self, x, z):
        """Add noise generated from uniform distribution to input.
        Scale result to [0, 1].

        Parameters
        ----------
        x : torch.Tensor
        z : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        """
        y = (x + z) / self.quantums
        ldj = -np.log(self.quantums) * np.prod(y.shape[1:])
        return y, ldj

    def _quantize(self, y):
        """Discretize [0, 1] continuous input into [0, 256).

        Parameters
        ----------
        y : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        """
        x = quantize(y, self.quantums)
        ldj = np.log(self.quantums) * np.prod(x.shape[1:])
        return x, ldj

    def _sigmoid(self, x, reverse=False):
        """Apply sigmoid function.
        y = 1/(1+exp(-x))
        Inverse of sigmoid is the logit function.
        y = log(x/(1-x))

        Parameters
        ----------
        x : torch.Tensor
        reverse : bool

        Returns
        -------
        y : torch.Tensor
        """
        if not reverse:
            y = torch.sigmoid(x)
            ldj = (-x - 2 * f.softplus(-x)).sum(dim=[1, 2, 3])
        else:
            x = x * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            y = torch.log(x) - torch.log(1 - x)
            ldj = np.log(1 - self.alpha) * np.prod(x.shape[1:])
            ldj += (-torch.log(x) - torch.log(1 - x)).sum(dim=[1, 2, 3])
        return y, ldj
