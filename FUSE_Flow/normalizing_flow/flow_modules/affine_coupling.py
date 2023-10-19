import torch
from torch import nn

from FUSE_Flow.other_modules.utils import AEInit


class AffineCoupling(nn.Module):
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
    ablation : dict
        Configurations for ablation tests.
    hyper : dict
        Hyper-parameters.
    """

    def __init__(self, est_arch, c_x, c_u, ablation, hyper):
        super().__init__()
        self.estimator = est_arch(
            c_in=c_x // 2 + c_u,
            c_out=c_x // 2 * 2,
            c_hid=c_x * hyper['c_u_mult'],
            n_layers=hyper['n_conv'],
            init=AEInit.zero,
            attention_type=ablation['attention_type'],
            attn_red_ratio=hyper['attn_red_ratio'],
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
        x_a, x_b = x.chunk(2, dim=1)
        if u is None:
            x_nn = x_b
        else:
            x_nn = torch.cat((x_b, u), dim=1)
        nn_out = self.estimator(x_nn)
        log_s, t = nn_out.chunk(2, dim=1)

        # stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        log_s = torch.tanh(log_s / s_fac) * s_fac

        # affine coupling
        if not reverse:
            return self._forward_flow(x_a, x_b, log_s, t)
        else:
            return self._reverse_flow(x_a, x_b, log_s, t)

    @staticmethod
    def _forward_flow(x_a, x_b, log_s, t):
        """Apply affine coupling.
        y_a = s * x_a + t
        y_b = x_b

        Parameters
        ----------
        x_a : torch.Tensor
        x_b : torch.Tensor
        log_s : torch.Tensor
        t : torch.Tensor

        Returns
        -------
        y : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        s = log_s.exp()
        y_a = x_a.mul(s).add(t)
        y_b = x_b
        y = torch.cat((y_a, y_b), dim=1)
        ldj = log_s.sum(dim=[1, 2, 3])
        return y, ldj

    @staticmethod
    def _reverse_flow(y_a, y_b, log_s, t):
        """Apply affine coupling.
        x_a = (y_a - t) / s
        x_b = y_b

        Parameters
        ----------
        y_a : torch.Tensor
        y_b : torch.Tensor
        log_s : torch.Tensor
        t : torch.Tensor

        Returns
        -------
        x : torch.Tensor
        ldj : torch.Tensor
            Log-determinant of Jacobian Matrix.
        """
        s = log_s.neg().exp()
        x_a = y_a.sub(t).mul(s)
        x_b = y_b
        x = torch.cat((x_a, x_b), dim=1)
        ldj = log_s.sum(dim=[1, 2, 3])
        return x, ldj
