import torch
from torch import nn


class Split(nn.Module):
    """Split layer in Scale Block.
    Evaluation or sampling of prior.

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, prior, reverse):
        if not reverse:
            return self._forward_flow(x, prior)
        else:
            return self._reverse_flow(x, prior)

    @staticmethod
    def _forward_flow(x, prior):
        """Split off half channels to evaluate log-probability under the prior.

        Parameters
        ----------
        x : torch.Tensor
        prior : torch.distributions.distribution.Distribution

        Returns
        -------
        y : torch.Tensor
        lp : torch.Tensor
            Log-probability of sample under prior.
        """
        y, z = x.chunk(2, dim=1)
        lp = prior.log_prob(z).sum(dim=[1, 2, 3])
        return y, lp

    @staticmethod
    def _reverse_flow(y, prior):
        """Sample prior to concatenate along channels.

        Parameters
        ----------
        y : torch.Tensor
        prior : torch.distributions.distribution.Distribution

        Returns
        -------
        x : torch.Tensor
        lp : torch.Tensor
            Log-probability of sample under prior.
        """
        z = prior.sample(sample_shape=y.shape).squeeze(dim=-1).to(y.device)
        x = torch.cat([y, z], dim=1)
        lp = -prior.log_prob(z).sum(dim=[1, 2, 3])
        return x, lp
