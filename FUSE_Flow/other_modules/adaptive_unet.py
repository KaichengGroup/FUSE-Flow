import math

import pytorch_lightning as pl
import torch
from torch import nn

from FUSE_Flow.other_modules.utils import AEInit
from .conv_modules.conv_block import ConvBlock
from .gated_resnet import UpsampleBlock, DownsampleBlock


class AdaptiveUNet(pl.LightningModule):
    """SR network architecture that uses Residual-in-Residual Dense Blocks.
    Implement Figure (3) in ESRGAN paper.

    Parameters
    ----------
    d_x : int
        Priority dimension (height or width) of input chosen for downstream comparisons.
    d_y : int
        Priority dimension (height or width) of output chosen for downstream comparisons.
    add_depth : int
        Additional depth on top of that required based on difference in scale of input and output.
        Largest value this value can take is the largest n where input_shape[1]/factor**n is whole and odd.
    factor: int
        Factor at which data expands or shrinks. Currently only works for factor = 2.
    c_in : int
        Number of channels of input tensor.
    c_hid : int
        Number of channels of inner convolutional layers.
    n_conv : int
        Number of conv layers.
    no_skip : bool
        To include skip connection between mirrored layers.
    attention_type: AttentionType
        type of attention implemented in gated conv blocks
    attn_red_ratio : float  # default 16
        Minimum value = 1, Maximum value = c_in, set reduction from 1 to c_in using attn_red_ratio
        Smaller attn_red_ratio --> Less Parameters
        Hyperparameter to vary capacity and computational cost of SE blocks in the network.
    """

    def __init__(self, d_x, d_y, add_depth, factor, c_in, c_hid, n_conv, no_skip,
                 attention_type, attn_red_ratio):
        super().__init__()
        self.save_hyperparameters()
        self.no_skip = no_skip
        # double the number of channels needed if no skip connection
        if no_skip:
            c_inter = c_hid
        else:
            c_inter = c_hid//2
        # larger of the input and output priority dimension

        d_l = max(d_x, d_y)
        # larger of the input and output priority dimension
        d_s = min(d_x, d_y)
        # scale difference between input and output
        scale = int(d_l / d_s)
        # max depth of U-Net
        max_depth = int(math.log(scale, factor) + 1 + add_depth)
        # represents dimension size of unwanted depths
        denominator = d_l // (factor ** (max_depth - 1))
        # number of down-sampling blocks
        n_down = math.floor(math.log(d_x / denominator, factor))
        # number of up-sampling layers in encoder
        n_enc_up = max_depth - 1 - n_down - math.ceil(math.log(scale, factor) % 1)
        # number of up-sampling layers in decoder
        n_dec_up = math.floor(math.log(d_y / denominator, factor))
        # discrepancy between size of input priority dimension and nearest larger multiple of 2
        k_up = d_l // (factor ** math.floor(math.log(scale, factor))) - d_s
        # discrepancy between size of input priority dimension and nearest smaller multiple of 2
        k_down = d_s - d_l // (factor ** math.ceil(math.log(scale, factor)))
        # need resizing if data is not multiple of 2
        self.need_resizing = k_up or k_down

        # encoder
        if not no_skip:
            c_up = c_inter // (factor ** (n_down+self.need_resizing))
            self.up_resizer = nn.Sequential(
                *[ConvBlock(nn.ConvTranspose2d, c_in, c_up,
                            3, 1, 1, AEInit.xavier, attention_type, attn_red_ratio)] +
                 [ConvBlock(nn.ConvTranspose2d, c_up, c_up,
                            3, 1, 0, AEInit.xavier, attention_type, attn_red_ratio)] * (k_up // 2)
            )
            left_up_blocks = []
            for i in range(n_enc_up):
                left_up_blocks.append(
                    UpsampleBlock(
                        c_up // (factor ** i),
                        c_up // (factor ** (i+1)),
                        c_up // (factor ** i),
                        n_conv,
                        AEInit.xavier,
                        attention_type,
                        attn_red_ratio
                    )
                )
            self.left_up_blocks = nn.ModuleList(left_up_blocks)

        c_down = c_inter // (factor ** n_down)
        if n_down == 0:
            c_down = c_inter * 2
        if k_down > 0:
            self.down_resizer = nn.Sequential(
                *[ConvBlock(nn.Conv2d, c_in, c_down,
                            3, 1, 1, AEInit.xavier, attention_type, attn_red_ratio)] +
                 [ConvBlock(nn.Conv2d, c_down, c_down,
                            3, 1, 0, AEInit.xavier, attention_type, attn_red_ratio)] * (k_down // 2)
            )
        else:
            self.down_resizer = ConvBlock(nn.Conv2d, c_in, c_down,
                                          3, 1, 1, AEInit.xavier, attention_type, attn_red_ratio)
        down_blocks = []
        for i in range(n_down):
            if i == n_down-1:  # encoding
                down_blocks.append(
                    DownsampleBlock(
                        c_down * (factor ** i),
                        c_hid,
                        c_hid,
                        n_conv,
                        AEInit.xavier,
                        attention_type,
                        attn_red_ratio
                    )
                )
            else:
                down_blocks.append(
                    DownsampleBlock(
                        c_down * (factor ** i),
                        c_down * (factor ** (i + 1)),
                        c_down * (factor ** i),
                        n_conv,
                        AEInit.xavier,
                        attention_type,
                        attn_red_ratio
                    )
                )
        self.down_blocks = nn.ModuleList(down_blocks)

        # decoder
        right_up_blocks = []
        for i in range(n_dec_up):
            right_up_blocks.append(
                UpsampleBlock(
                    c_hid // (factor ** i),
                    c_hid // (factor ** (i+2)),
                    c_hid // (factor ** i),
                    n_conv,
                    AEInit.xavier,
                    attention_type,
                    attn_red_ratio
                )
            )
        self.right_up_blocks = nn.ModuleList(right_up_blocks)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return intermediate predictions of most blocks.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        features : dict
        """
        residuals = {}
        starting_idx = len(self.down_blocks)
        if not self.no_skip:
            x_up = self.up_resizer(x)
            if self.need_resizing:
                starting_idx += 1
                residuals[starting_idx] = x_up
            for i, block in enumerate(self.left_up_blocks):
                x_up = block(x_up)
                residuals[starting_idx+i+1] = x_up

        x_down = self.down_resizer(x)
        for i, block in enumerate(self.down_blocks):
            if not self.no_skip:
                residuals[len(self.down_blocks)-i] = x_down
            x_down = block(x_down)

        features = {0: x_down}  # encoding

        for i, block in enumerate(self.right_up_blocks):
            x_down = block(x_down)
            if not self.no_skip:
                x_down = torch.cat((x_down, residuals[i+1]), dim=1)
            features[i+1] = x_down

        return features
