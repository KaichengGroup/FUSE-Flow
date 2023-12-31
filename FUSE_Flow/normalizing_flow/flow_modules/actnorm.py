import torch
from torch import nn


class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).
    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    Adapted from:
        > https://github.com/openai/glow
    """

    def __init__(self, num_channels, height, width):
        super().__init__()

        # Input gets concatenated along channel axis
        # num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            mean, inv_std = self._get_moments(x)
            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, reverse=False):
        # import pdb;pdb.set_trace()
        # x = torch.cat(x, dim=1)
        # import pdb;pdb.set_trace()
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, reverse)

        return x, ldj


class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow
    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels):
        super(ActNorm, self).__init__(num_channels, 1, 1)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = -self.inv_std.log().sum() * x.size(2) * x.size(3)
        else:
            x = x * self.inv_std
            sldj = self.inv_std.log().sum() * x.size(2) * x.size(3)
        return x, sldj


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.
    Args:
        tensor (torch.Tensor):
        dim (list):
        keepdims (bool):
    Returns:


    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of values to average.
    dim : list
        List of dimensions along which to take the mean.
    keepdims : bool
        Keep dimensions rather than squeezing.

    Returns
    -------
    tensor : torch.Tensor
        New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor
