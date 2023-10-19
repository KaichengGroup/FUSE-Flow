import torch
from torch import nn

from .actnorm import ActNorm
from .affine_coupling import AffineCoupling
from .affine_injection import AffineInjection
from .conv_1x1 import InvertibleConv1x1
from .logistic_coupling import LogisticCoupling


class FlowStep(nn.Module):
    """Flow Step in Scale Block.
    A single atomic step that represents a normalized, conditional, tractable, flexible,
    and learnable bijective transformation applied on all channels.

    Based on the paper:
    "SRFlow: Learning the Super-Resolution Space with Normalizing Flow"
    by Andreas Lugmayr, Martin Danelljan, Luc Van Gool, and Radu Timofte
    (https://arxiv.org/abs/2006.14200).

    Parameters
    ----------
    est_arch : type
        Architecture of neural network as estimator for parameters log(s) and t.
    c_x : int
        Number of channels of input x.
    c_u : int
        Number of channels of conditional input u.
        This should be 0 if no conditional input is used.
    is_transition : bool, optional
        Determines if flow step is a transition step (without conditional components).
    ablation : dict
        Configurations for ablation tests.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, c_x, c_u, is_transition, ablation, hyper):
        super().__init__()
        self.ablation = ablation
        self.is_transition = is_transition
        if not ablation['no_actnorm']:
            self.actnorm = ActNorm(c_x)
        if not ablation['no_1x1_conv']:
            self.conv1x1 = InvertibleConv1x1(c_x)
        if ablation['no_transition'] or not is_transition:
            if not ablation['no_injection']:
                self.injection = AffineInjection(est_arch, c_x, c_u, ablation, hyper)
            if not ablation['no_coupling']:
                if ablation['logistic_coupling']:
                    self.coupling = LogisticCoupling(est_arch, c_x, c_u, ablation, hyper)
                else:
                    self.coupling = AffineCoupling(est_arch, c_x, c_u, ablation, hyper)

    def forward(self, x, u, reverse):
        if not reverse:
            return self._forward_flow(x, u)
        else:
            return self._reverse_flow(x, u)

    def _forward_flow(self, x, u):
        """Forward flow through each layer.

        Parameters
        ----------
        x : torch.Tensor
        u : torch.Tensor or None
            Conditional input.

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        ldj = torch.zeros(x.size(0), device=x.device)
        if not self.ablation['no_actnorm']:
            x, ldj_actnorm = self.actnorm(x, reverse=False)
            ldj += ldj_actnorm.expand(x.size(0))
        if not self.ablation['no_1x1_conv']:
            x, ldj_conv = self.conv1x1(x, reverse=False)
            ldj += ldj_conv.expand(x.size(0))
        if self.ablation['no_transition'] or not self.is_transition:
            if not self.ablation['no_injection']:
                x, ldj_injection = self.injection(x, u, reverse=False)
                ldj += ldj_injection
            if not self.ablation['no_coupling']:
                x, ldj_coupling = self.coupling(x, u, reverse=False)
                ldj += ldj_coupling
        return x, ldj

    def _reverse_flow(self, y, u):
        """Reverse flow through each layer.

        Parameters
        ----------
        y : torch.Tensor
        u : torch.Tensor or None
            Conditional input.

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        ldj = torch.zeros(y.size(0), device=y.device)
        if self.ablation['no_transition'] or not self.is_transition:
            if not self.ablation['no_coupling']:
                y, ldj_coupling = self.coupling(y, u, reverse=True)
                ldj += ldj_coupling
            if not self.ablation['no_injection']:
                y, ldj_injection = self.injection(y, u, reverse=True)
                ldj += ldj_injection
        if not self.ablation['no_1x1_conv']:
            y, ldj_conv = self.conv1x1(y, reverse=True)
            ldj += ldj_conv.expand(y.size(0))
        if not self.ablation['no_actnorm']:
            y, ldj_actnorm = self.actnorm(y, reverse=True)
            ldj += ldj_actnorm.expand(y.size(0))
        return y, ldj
