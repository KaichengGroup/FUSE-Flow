import math

from torch import nn

from FUSE_Flow.other_modules.utils import AEInit, AttentionType
from .conv_modules.concat_elu import ConcatELU

from .conv_modules.gated_conv import GatedConv
from .conv_modules.se_gated_conv import SEGatedConv
from .conv_modules.cbam_gated_conv import CBAMGatedConv


from .conv_modules.layer_norm_channels import LayerNormChannels


class GatedResidualNetBase(nn.Module):
    """Convolutional neural network that includes an attention mechanism
    similar to transformers.

    Based on papers:
    "Flow++: Improving Flow-Based Generative Models with
    Variational Dequantization and Architecture Design"
    by Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel
    (https://arxiv.org/abs/1902.00275).
    "Free-Form Image Inpainting with Gated Convolution"
    by Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, Thomas Huang
    (https://arxiv.org/abs/1806.03589).
    "Language Modeling with Gated Convolutional Networks"
    by Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier
    (https://arxiv.org/abs/1612.08083).

    Parameters
    ----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    c_hid : int
        Number of hidden dimensions to use within the network.
    n_layers : int
        Number of gated ResNet blocks to apply.
    conv : type
        nn.Conv2d or nn.ConvTranspose2d
    k_size : int
        Kernel size.
    stride : int | tuple
        Controls the stride for the cross-correlation, a single number or a tuple.
    padding : int | tuple | str
        Controls the amount of padding applied to the input.
        It can be either a string {‘valid’, ‘same’}
        or an int / a tuple of ints giving the amount of implicit padding applied on both sides.
    init : AEInit
        Weight initialization method.
    attention_type: AttentionType
        type of attention implemented in gated conv blocks
    attn_red_ratio : float  # default 16
        Minimum value = 1, Maximum value = c_in, set reduction from 1 to c_in using attn_red_ratio
        Smaller attn_red_ratio --> Less Parameters
        Hyperparameter to vary capacity and computational cost of SE blocks in the network.
    """

    def __init__(self, c_in, c_out, c_hid, n_layers,
                 conv, k_size, stride, padding, init, attention_type, attn_red_ratio):
        super().__init__()
        # dimension manipulating layers
        layers = [
            conv(c_in, c_hid, kernel_size=k_size, stride=stride, padding=padding)
        ]
        # main dimension-consistent computational layers
        for layer_index in range(n_layers):
            if attention_type == AttentionType.none:
                layers += [GatedConv(c_hid, c_hid), LayerNormChannels(c_hid)]
            elif attention_type == AttentionType.se:
                layers += [SEGatedConv(c_hid, c_hid, attn_red_ratio), LayerNormChannels(c_hid)]
            elif attention_type == AttentionType.cbam:
                layers += [CBAMGatedConv(c_hid, c_hid, attn_red_ratio), LayerNormChannels(c_hid)]
        layers += [ConcatELU(), nn.Conv2d(2 * c_hid, c_out, kernel_size=3, padding=1)]

        self.nn = nn.Sequential(*layers)

        # initialize weights and biases
        # xavier uniform initialization as proposed in
        # Understanding the difficulty of training deep feedforward neural networks
        for name, param in self.nn.named_parameters():
            if name.endswith('.bias'):
                param.data.fill_(0)
            elif name.endswith('.weight'):
                if len(param.shape) >= 2:
                    bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                    param.data.uniform_(-bound, bound)
        if init == AEInit.zero:
            # zero initialization as proposed in
            # Chapter 3.3 of Glow: Generative Flow with Invertible 1×1 Convolutions
            self.nn[-1].weight.data.zero_()
            self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)


class GatedResidualNet(GatedResidualNetBase):
    def __init__(self, c_in, c_out, c_hid, n_layers, init, attention_type, attn_red_ratio):
        super().__init__(c_in, c_out, c_hid, n_layers, nn.Conv2d,
                         3, 1, 1, init, attention_type, attn_red_ratio)

    def forward(self, x):
        return self.nn(x)


class UpsampleBlock(GatedResidualNetBase):
    def __init__(self, c_in, c_out, c_hid, n_layers, init, attention_type, attn_red_ratio):
        super().__init__(c_in, c_out, c_hid, n_layers, nn.ConvTranspose2d,
                         4, 2, 1, init, attention_type, attn_red_ratio)

    def forward(self, x):
        return self.nn(x)


class DownsampleBlock(GatedResidualNetBase):
    def __init__(self, c_in, c_out, c_hid, n_layers, init, attention_type, attn_red_ratio):
        super().__init__(c_in, c_out, c_hid, n_layers, nn.Conv2d,
                         4, 2, 1, init, attention_type, attn_red_ratio)

    def forward(self, x):
        return self.nn(x)
