from torch import nn


class LayerNormChannels(nn.Module):
    """This module applies layer norm across channels in an image.
    Has been shown to work well with ResNet connections.

    Parameters
    ----------
    c_in : int
        Number of channels of the input.
    """

    def __init__(self, c_in):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
