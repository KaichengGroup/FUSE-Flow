import math

from torch import nn

from FUSE_Flow.other_modules.utils import AEInit


class ConvBlock(nn.Module):
    def __init__(self, conv, c_in, c_out, kernel_size, stride, padding, init, attention_type, attn_red_ratio):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(negative_slope=0.2),
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
        )

        # initialize weights and biases
        if init == AEInit.zero:
            self.block[-1].weight.data.zero_()
            self.block[-1].bias.data.zero_()
        elif init == AEInit.xavier:
            for name, param in self.block.named_parameters():
                if name.endswith('.bias'):
                    param.data.fill_(0)
                elif name.endswith('.weight'):
                    if len(param.shape) >= 2:
                        bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                        param.data.uniform_(-bound, bound)

    def forward(self, x):
        return self.block(x)
