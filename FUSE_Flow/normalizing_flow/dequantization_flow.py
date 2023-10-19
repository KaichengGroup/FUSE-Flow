import pytorch_lightning as pl
import torch
from torch import nn

from .flow_modules.flow_step import FlowStep
from .flow_modules.squeeze import Squeeze


class DequantizationFlow(pl.LightningModule):
    """Conditional Normalizing Flow in SRFlow++.

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
        Number of Flow Steps per Scale Block.
    c_x : int
        Number of channels of input x.
    c_u : int
        Number of channels of conditional input u.
        This should be 0 if no conditional input is used.
    ablation : dict
        Configurations for ablation tests.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, factor, n_flow, c_x, c_u, ablation, hyper):
        super().__init__()
        self.squeeze_layer = Squeeze(factor)
        self.flow_layers = nn.ModuleList()
        for i in range(n_flow):
            self.flow_layers.append(FlowStep(
                est_arch=est_arch,
                c_x=c_x,
                c_u=c_u,
                is_transition=False,
                ablation=ablation,
                hyper=hyper
            ))

    def forward(self, x, u):
        """Compute log-likelihood of input data.

        Parameters
        ----------
        x : torch.Tensor
        u : torch.Tensor or None
            Conditional input.

        Returns
        -------
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        ldj = torch.zeros(x.shape[0], device=x.device)
        x = self.squeeze_layer(x, reverse=False)
        for flow_layer in self.flow_layers:
            x, ldj_flow = flow_layer(x, u, reverse=False)
            ldj += ldj_flow
        y = self.squeeze_layer(x, reverse=True)
        return y, ldj
