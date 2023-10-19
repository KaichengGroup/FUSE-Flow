import kornia.augmentation as aug
import torch
from torch import nn


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, config) -> None:
        super().__init__()
        aug_list = []
        if config['hor_flip']:
            aug_list.append(aug.RandomHorizontalFlip(p=0.5))
        if config['ver_flip']:
            aug_list.append(aug.RandomVerticalFlip(p=0.5))
        if config['col_jig']:
            aug_list.append(aug.ColorJiggle(0.1, 0.1, 0.25, 0.5, p=0.5))
        self.aug_list = aug.AugmentationSequential(
            *aug_list,
            data_keys=['input', 'input']
        )

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x, y):
        x_out, y_out = self.aug_list(x, y)
        return x_out, y_out
