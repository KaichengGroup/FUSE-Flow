import torch
from torch import nn

from FUSE_Flow.other_modules.utils import AEInit
from . import log_dist as logistic


class LogisticCoupling(nn.Module):
    """Conditional Affine Coupling layer in Flow Step.
    A tractable, flexible, and learnable bijective transformation.

    Based on papers:
    "SRFlow: Learning the Super-Resolution Space with Normalizing Flow"
    by Andreas Lugmayr, Martin Danelljan, Luc Van Gool, and Radu Timofte
    (https://arxiv.org/abs/2006.14200).
    "NICE: Non-linear Independent Components Estimation"
    by Laurent Dinh David Krueger Yoshua Bengio
    (https://arxiv.org/abs/1410.8516).

    Parameters
    ----------
    est_arch : type
        Architecture of neural network as estimator for parameters log(s) and t.
    c_x : int
        Number of channels of input x.
    c_u : int
        Number of channels of conditional input u.
        This should be 0 if no conditional input is used.
    n_comp : int
        Number of components in the mixture.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, c_x, c_u, ablation, hyper, n_comp=8):
        super().__init__()
        self.k = n_comp
        self.estimator = est_arch(
            c_in=c_x // 2 + c_u,
            c_out=c_x // 2 * (2 + 3 * self.k),
            c_hid=c_x * hyper['c_u_mult'],
            n_layers=hyper['n_conv'],
            init=AEInit.zero,
            attention_type=ablation['attention_type'],
            attn_red_ratio=hyper['attn_red_ratio']
        )
        self.scaling_factor = nn.Parameter(torch.zeros(c_x // 2))

    def forward(self, x, u, reverse):
        """Compute parameters log(s) and t and perform affine coupling.

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
        # estimate parameters
        b, c, h, w = x.size()
        x_a, x_b = x.chunk(2, dim=1)
        if u is None:
            x_nn = x_b
        else:
            x_nn = torch.cat((x_b, u), dim=1)
        nn_out = self.estimator(x_nn)

        nn_out = nn_out.view(b, -1, c//2, h, w)
        log_s, t, pi, mu, scales = nn_out.split((1, 1, self.k, self.k, self.k), dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        log_s = torch.tanh(log_s / s_fac) * s_fac
        s = log_s.exp().squeeze(1)
        t = t.squeeze(1)
        scales = scales.clamp(min=-7)  # From the code in original Flow++ paper

        # affine coupling
        if not reverse:
            return self._forward_flow(x_a, x_b, s, t, pi, mu, scales)
        else:
            return self._reverse_flow(x_a, x_b, s, t, pi, mu, scales)

    @staticmethod
    def _forward_flow(x_a, x_b, a, b, pi, mu, s):
        """Apply affine coupling.
        y_a = s * x_a + t
        y_b = x_b

        Parameters
        ----------
        x_a : torch.Tensor
        x_b : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        y_a = logistic.mixture_log_cdf(x_a, pi, mu, s).exp()
        y_a, ldj_scale = logistic.inverse(y_a)
        y_a = (y_a + b) * a
        y_b = x_b
        ldj_log = logistic.mixture_log_pdf(x_a, pi, mu, s)
        y = torch.cat((y_a, y_b), dim=1)
        ldj = (ldj_scale + ldj_log + a).flatten(1).sum(-1)
        return y, ldj

    @staticmethod
    def _reverse_flow(y_a, y_b, a, b, pi, mu, s):
        """Apply affine coupling.
        x_a = (y_a - t) / s
        x_b = y_b

        Parameters
        ----------
        y_a : torch.Tensor
        y_b : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        x_a = y_a * a.mul(-1).exp() - b
        x_a, ldj_scale = logistic.inverse(x_a, reverse=True)
        x_a = x_a.clamp(1e-5, 1. - 1e-5)
        x_a = logistic.mixture_inv_cdf(x_a, pi, mu, s)
        x_b = y_b
        ldj_log = logistic.mixture_log_pdf(x_a, pi, mu, s)
        x = torch.cat((x_a, x_b), dim=1)
        ldj = -(ldj_scale + ldj_log + a).flatten(1).sum(-1)
        return x, ldj
