import pytorch_lightning as pl
import torch
from torch import nn

from FUSE_Flow.other_modules.utils import SBPosition
from .flow_modules.scale_block import ScaleBlock


class GenerativeFlow(pl.LightningModule):
    """Conditional Normalizing Flow in FUSE-Flow.

    Based on papers:
    "SRFlow: Learning the Super-Resolution Space with Normalizing Flow"
    by Andreas Lugmayr, Martin Danelljan, Luc Van Gool, and Radu Timofte
    (https://arxiv.org/abs/2006.14200).
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Parameters
    ----------
    est_arch : type
        Architecture of neural network as estimator for parameters log(s) and t.
    output_shape : tuple
        Shape of output image.
    n_scale : int
        Number of scale blocks.
    factor : int
        Factor at which features are shrunk at each scale block.
    n_flow : int
        Number of Flow Steps per Scale Block.
    c_u : int
        Number of channels of conditional input u.
        This should be 0 if no conditional input is used.
    ablation : dict
        Configurations for ablation tests.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, output_shape, n_scale, factor,
                 n_flow, c_u, ablation, hyper):
        super().__init__()
        c, h, w = output_shape
        self.z_shape = (
            c * factor**n_scale,
            h // factor**(n_scale-1),
            w // factor**(n_scale-1)
        )

        self.scale_blocks = nn.ModuleList()
        for i in range(n_scale):
            if i == 0:
                block_position = SBPosition.first
                c_in = c
            elif i == n_scale - 1:
                block_position = SBPosition.last
                c_in = c * (factor ** (i+1))
            else:
                block_position = SBPosition.middle
                c_in = c * (factor ** (i+1))
            self.scale_blocks.append(ScaleBlock(
                est_arch=est_arch,
                factor=factor,
                n_flow=n_flow,
                c_x=c_in,
                c_u=c_u // (factor ** (n_scale-i-1)),
                block_position=block_position,
                ablation=ablation,
                hyper=hyper
            ))

    def forward(self, x, u_dict, prior, reverse):
        if not reverse:
            return self._forward_flow(x, u_dict, prior)
        else:
            return self._reverse_flow(x, u_dict, prior)

    def _forward_flow(self, x, u_dict, prior):
        """Compute log-likelihood of input data.

        Parameters
        ----------
        x : torch.Tensor
        u_dict : dict or None
            Stores the outputs of the autoencoder at various scale levels.
        prior : torch.distributions.distribution.Distribution
            Distribution for log probability evaluation and sampling.

        Returns
        -------
        ll : torch.Tensor
            Log-likelihood of sample under the posterior.
        """
        # forward flow
        ll = torch.zeros(x.shape[0], device=self.device)
        for i, scale_block in enumerate(self.scale_blocks):
            if u_dict is not None:
                u = u_dict[len(self.scale_blocks) - 1 - i]
            else:
                u = u_dict
            x, ll_block = scale_block(x, u, prior, reverse=False)
            ll += ll_block
        lp = prior.log_prob(x).sum(dim=[1, 2, 3])
        ll += lp
        return ll

    def _reverse_flow(self, n, u_dict, prior):
        """Transform prior into complex posterior distribution.

        Parameters
        ----------
        n : int
            Number of random samples.
        u_dict : dict or None
            Stores the outputs of the autoencoder at various scale levels.
        prior : torch.distributions.distribution.Distribution
            Distribution for log probability evaluation and sampling.

        Returns
        -------
        x : torch.Tensor
            Mean bits per dimension or mean log-likelihood
        ll : torch.Tensor
            Log-likelihood of sample under the prior.
        """
        # Sample latent representation from prior
        x = prior.sample(sample_shape=(n, *self.z_shape)).squeeze(dim=-1).to(self.device)

        # Transform z to x by inverting the flows
        ll = torch.zeros(n, device=self.device)
        for i, scale_block in enumerate(reversed(self.scale_blocks)):
            if u_dict is not None:
                u = u_dict[i]
            else:
                u = u_dict
            x, ll_block = scale_block(x, u, prior, reverse=True)
            ll += ll_block
        return x, ll
