from torch import nn


class Squeeze(nn.Module):
    """Squeeze layer in Scale Block.
    Invertible manipulation of resolution; trades channel for spatial resolution.

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Parameters
    ----------
    factor : int
        Factor at which resolution grows or shrinks.
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, reverse):
        if not reverse:
            return self._forward_flow(x)
        else:
            return self._reverse_flow(x)

    def _forward_flow(self, x):
        """Factor x factor neighbourhoods are stacked channel-wise.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        """
        n, c, h, w = x.shape
        x = x.view(n, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        y = x.view(n, c * self.factor * self.factor, h // self.factor, w // self.factor)
        return y

    def _reverse_flow(self, y):
        """Collect multiple channels into factor x factor neighbourhoods.

        Parameters
        ----------
        y : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        """
        n, c, h, w = y.shape
        y = y.view(n, c // self.factor // self.factor, self.factor, self.factor, h, w)
        y = y.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = y.view(n, c // self.factor // self.factor, h * self.factor, w * self.factor)
        return x
