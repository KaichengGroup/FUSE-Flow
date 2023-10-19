import torch
from torch import nn

from FUSE_Flow.other_modules.utils import SBPosition
from .flow_step import FlowStep
from .split import Split
from .squeeze import Squeeze


class ScaleBlock(nn.Module):
    """Scale Block in Normalizing Flow.
    A block that represents a single scale level in multiscale architecture.

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
    factor : int
        Factor at which features are shrunk at each scale block.
    n_flow : int
        Number of flow steps per Scale Block.
    c_x : int
        Number of channels of input x.
    c_u : int
        Number of channels of conditional input u.
        This should be 0 if no conditional input is used.
    block_position : FUSE_Flow.other_modules.utils.SBPosition
        If this is the last (smallest) scale block, no parameterize layer.
    ablation : dict
        Configurations for ablation tests.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, factor, n_flow, c_x, c_u, block_position, ablation, hyper):
        super().__init__()
        if block_position != SBPosition.first:
            self.squeeze_layer = Squeeze(factor)
        else:
            self.squeeze_layer = None
        if block_position == SBPosition.first:
            ablation = ablation.copy()
            ablation['no_coupling'] = True
        self.flow_layers = nn.ModuleList()
        for i in range(n_flow):
            self.flow_layers.append(
                FlowStep(
                    est_arch=est_arch,
                    c_x=c_x,
                    c_u=c_u,
                    is_transition=not i,
                    ablation=ablation,
                    hyper=hyper
                ))
        if block_position == SBPosition.middle:
            self.split_layer = Split()
        else:
            self.split_layer = None

    def forward(self, x, u, prior, reverse):
        if not reverse:
            return self._forward_flow(x, u, prior)
        else:
            return self._reverse_flow(x, u, prior)

    def _forward_flow(self, x, u, prior):
        """Forward flow through squeeze, multiple flow step, and split layers.

        Parameters
        ----------
        x : torch.Tensor
        u : torch.Tensor or None
            Conditional input.
        prior : torch.distributions.distribution.Distribution

        Returns
        -------
        x : torch.Tensor
        ll : torch.Tensor
            Log-likelihood contribution of block.
        """
        ll = torch.zeros(x.shape[0], device=x.device)
        if self.squeeze_layer is not None:
            x = self.squeeze_layer(x, reverse=False)
        for flow_layer in self.flow_layers:
            x, ldj = flow_layer(x, u, reverse=False)
            ll += ldj
        if self.split_layer is not None:
            x, lp = self.split_layer(x, prior, reverse=False)
            ll += lp
        return x, ll

    def _reverse_flow(self, y, u, prior):
        """Reverse flow through squeeze, multiple flow step, and split layers.

        Parameters
        ----------
        y : torch.Tensor
        u : torch.Tensor or None
            Conditional input.
        prior : torch.distributions.distribution.Distribution

        Returns
        -------
        y : torch.Tensor
        ll : torch.Tensor
            Log-likelihood contribution of block.
        """
        ll = torch.zeros(y.shape[0], device=y.device)
        if self.split_layer is not None:
            y, lp = self.split_layer(y, prior, reverse=True)
            ll += lp
        for flow_layer in reversed(self.flow_layers):
            y, ldj = flow_layer(y, u, reverse=True)
            ll += ldj
        if self.squeeze_layer is not None:
            y = self.squeeze_layer(y, reverse=True)
        return y, ll
