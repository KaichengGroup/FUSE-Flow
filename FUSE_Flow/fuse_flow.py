import math

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from data_modules.augmentation import DataAugmentation
from FUSE_Flow.other_modules.utils import quantize, ae_losses, PRETRAIN_PATH, DequantizationType, AEInit
from FUSE_Flow.normalizing_flow.dequantization_flow import DequantizationFlow
from FUSE_Flow.normalizing_flow.generative_flow import GenerativeFlow
from FUSE_Flow.other_modules.adaptive_unet import AdaptiveUNet, DownsampleBlock
from FUSE_Flow.other_modules.conv_modules.conv_block import ConvBlock
from FUSE_Flow.other_modules.dequantize import Dequantization
from FUSE_Flow.other_modules.gated_resnet import GatedResidualNet


class FUSEFlow(pl.LightningModule):
    """Implementation of FUSE-Flow.

    Based on the paper:
    "Quantitative mapping of unsectioned histology with fibre optic ultraviolet excitation and generative modelling"
    by Joel Lang Yi Ang, Ko Hui Tan, Alexander Si Kai Yong, Chiyo Wan Xuan Tan, Jessica Sze Jia Kng,
    Cyrus Jia Jun Tan, Rachael Hui Kie Soh, Julian Yi Hong Tan, Kaicheng Liang

    Parameters
    ----------
    output_shape : tuple
        Shape of output image.
    ablation : dict
        Ablation configurations.
    hyper : dict
        Hyper-parameter configurations.
    sample_size : int
        Number of samples to draw from learned posterior distribution.
    quantums : int
        Number of possible discrete values (usually 256 for 8-bit image).
    """

    def __init__(self, input_shape, output_shape, ablation, hyper, temperature, augmentations,
                 sample_size=None, quantums=256):
        super().__init__()
        self.prior = None
        self.ablation = ablation
        self.hyper = hyper
        self.temperature = temperature
        self.sample_size = sample_size
        self.aug = DataAugmentation(augmentations)

        # factor at which data expands or shrinks
        factor = hyper['factor']

        # height is arbitrarily chosen instead of width for comparison
        c_x, h_x, _ = input_shape
        c_y, h_y, _ = output_shape

        # initialize dequantization
        if not ablation['no_flow']:
            if ablation['dequantization'] == DequantizationType.var:
                deq_flow = DequantizationFlow(
                    est_arch=GatedResidualNet,
                    factor=factor,
                    n_flow=hyper['dequantization']['n_step'],
                    c_x=c_x * factor ** 2,
                    c_u=c_x,
                    ablation=ablation,
                    hyper=hyper['estimators']
                )
                downsample = DownsampleBlock(c_x, c_x, c_x, hyper['dequantization']['n_conv'],
                                             AEInit.xavier, ablation["attention_type"],
                                             hyper['estimators']['attn_red_ratio'])
            else:
                deq_flow = None
                downsample = None
            self.dequantizer = Dequantization(
                flow=deq_flow,
                downsample=downsample,
                perturbation_type=ablation['dequantization'],
                quantums=quantums
            )

        # initialize autoencoder
        if not ablation['no_autoencoder']:
            self.adaptive_unet = AdaptiveUNet(
                d_x=h_x,
                d_y=h_y,
                factor=factor,
                add_depth=hyper['flow']['n_scale_add'],
                c_in=c_x,
                c_hid=hyper['autoencoder']['c_u'],
                n_conv=hyper['autoencoder']['n_conv'],
                no_skip=ablation['no_skip'],
                attention_type=ablation['attention_type'],
                attn_red_ratio=hyper['autoencoder']['attn_red_ratio'],
            )
            if not ablation['no_pretrain']:
                state_dict = torch.load(PRETRAIN_PATH)['state_dict']
                for key, value in state_dict.copy().items():
                    module_levels = key.split('.')
                    if module_levels[0] != 'adaptive_unet':
                        del state_dict[key]
                    else:
                        state_dict['.'.join(module_levels[1:])] = state_dict.pop(key)
                self.adaptive_unet.load_state_dict(state_dict)
                if not ablation['no_freeze']:
                    self.adaptive_unet.freeze()

        # initialize main generative normalizing flow
        if not ablation['no_flow']:
            # scale difference between input and output
            scale = int(max(h_x, h_y)/min(h_x, h_y))

            # number of scale blocks in normalizing flow
            # log_factor(pixel_scale) + 1 is the minimum
            # log_factor(pixel_scale) + 1 + n is the maximum
            # where n is the largest value where input_shape[1]/factor**n is odd
            n_scale = int(math.log(scale, factor) + 1 + hyper['flow']['n_scale_add'])

            self.normalizing_flow = GenerativeFlow(
                est_arch=GatedResidualNet,
                output_shape=output_shape,
                n_scale=n_scale,
                factor=factor,
                n_flow=hyper['flow']['n_step'],
                c_u=hyper['autoencoder']['c_u'] if not ablation['no_autoencoder'] else 0,
                ablation=ablation,
                hyper=hyper['estimators']
            )
        else:
            scale = int(max(h_x, h_y)/min(h_x, h_y))
            max_depth = int(math.log(scale, factor) + 1 + hyper['flow']['n_scale_add'])

            self.output_block = ConvBlock(
                nn.Conv2d,
                hyper['autoencoder']['c_u'] // (factor ** (max_depth-1)),
                c_x, 3, 1, 1,
                AEInit.xavier,
                ablation['attention_type'],
                hyper['estimators']['attn_red_ratio']
            )
            self.sigmoid = nn.Sigmoid()
            self.ae_loss = ae_losses[ablation['autoencoder_loss']]

    def forward(self, lr):
        """Training.

        Parameters
        ----------
        lr : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
        """
        x = lr.repeat(self.sample_size, 1, 1, 1)

        if not self.ablation['no_autoencoder']:
            u_dict = self.adaptive_unet(x)
        else:
            u_dict = None

        if not self.ablation['no_flow']:
            x, _ = self.normalizing_flow(
                x=x.shape[0],
                u_dict=u_dict,
                prior=self.prior,
                reverse=True
            )
            sr, _ = self.dequantizer(x, reverse=True)
        else:
            y = self.sigmoid(self.output_block(u_dict[max(u_dict.keys())]))
            sr = quantize(y, 256)
        return lr, sr

    def on_train_start(self):
        self.prior = torch.distributions.normal.Normal(
            loc=0.0,
            scale=self.temperature
        )

    def training_step(self, batch, batch_idx):
        """Training.

        Parameters
        ----------
        batch : tuple
        batch_idx : int

        Returns
        -------
        loss : torch.Tensor
        """
        lr, hr = batch
        if not self.ablation['no_augmentation']:
            lr, hr = self.aug(lr, hr)
            lr = lr.squeeze(0)
            hr = hr.squeeze(0)

        if not self.ablation['no_autoencoder']:
            u_dict = self.adaptive_unet(lr)
        else:
            u_dict = None

        if not self.ablation['no_flow']:
            x, ll_deq = self.dequantizer(hr * 255, reverse=False)
            ll_flow = self.normalizing_flow(
                x=x,
                u_dict=u_dict,
                prior=self.prior,
                reverse=False
            )
            ll = ll_deq + ll_flow  # cumulative log-likelihood
            nll = -ll  # negative log-likelihood
            bpd = nll * np.log2(np.exp(1)) / np.prod(x.shape[1:])  # bits per dimension
            loss = bpd.mean()
        else:
            y = self.sigmoid(self.output_block(u_dict[max(u_dict.keys())]))
            loss = self.ae_loss(y, hr)

        self.log('loss', loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def on_predict_start(self):
        self.prior = torch.distributions.normal.Normal(
            loc=0.0,
            scale=self.temperature
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        lr, _ = batch
        _, sr = self(lr)
        return sr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyper['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=self.hyper['gamma'])
        return [optimizer], [scheduler]
