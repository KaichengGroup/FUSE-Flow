import torch
from torch import nn

from FUSE_Flow.other_modules.utils import AEInit


class AffineInjection(nn.Module):
    """Affine Injection layer in Flow Step.
    A tractable, flexible, and learnable bijective transformation using external information.

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
    ablation : dict
        Configurations for ablation tests.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, c_x, c_u, ablation, hyper):
        super().__init__()
        if c_u > 0:
            self.estimator = est_arch(
                c_in=c_u,
                c_out=c_x * 2,
                c_hid=c_x * hyper['c_u_mult'],
                n_layers=hyper['n_conv'],
                init=AEInit.zero,
                attention_type=ablation['attention_type'],
                attn_red_ratio=hyper['attn_red_ratio']
            )
            self.scaling_factor = nn.Parameter(torch.zeros(c_x))

    def forward(self, x, u, reverse):
        """Compute parameters log(s) and t and perform affine transformation.

        Parameters
        ----------
        x : torch.Tensor
        u : torch.Tensor or None
            Conditional input.
        reverse : bool

        Returns
        -------
        y : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        # skip if not conditional setting
        if u is None:
            return x, torch.zeros(x.shape[0], device=x.device)

        # estimate parameters
        nn_out = self.estimator(u)
        log_s, t = nn_out.chunk(2, dim=1)

        # stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        log_s = torch.tanh(log_s / s_fac) * s_fac

        # affine transformation
        if not reverse:
            return self._forward_flow(x, log_s, t)
        else:
            return self._reverse_flow(x, log_s, t)

    @staticmethod
    def _forward_flow(x, log_s, t):
        """Apply affine transformation.
        y = s * x + t

        Parameters
        ----------
        x : torch.Tensor
        log_s : torch.Tensor
        t : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        s = log_s.exp()
        y = x.mul(s).add(t)
        ldj = log_s.sum(dim=[1, 2, 3])
        return y, ldj

    @staticmethod
    def _reverse_flow(y, log_s, t):
        """Apply affine transformation.
        x = (y - t) / s

        Parameters
        ----------
        y : torch.Tensor
        log_s : torch.Tensor
        t : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        s = log_s.neg().exp()
        x = y.sub(t).mul(s)
        ldj = log_s.sum(dim=[1, 2, 3])
        return x, ldj
