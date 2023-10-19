import os
from typing import Tuple, Optional, Callable, Any

import numpy as np
from torchvision.datasets import VisionDataset


class NPZDataset(VisionDataset):
    """Load datasets from NPZ files.
    NPZ files are assumed to have 2 files named "x" and "y"
    that represent the input and target, respectively.

    Parameters
    ----------
    root : str
        Root directory of dataset where directory.
        ``celebA.npz`` exists or will be saved to if download is set to True.
    transform : callable, optional
        A function/transform that takes in a Numpy array.
    """

    def __init__(
            self,
            root: str,
            filename: str,
            transform: Optional[Callable] = None
    ):
        super().__init__(root, transform=transform)
        self.data, self.target = self._load_npz(os.path.join(root, f'{filename}.npz'))

    @staticmethod
    def _load_npz(npz_path):
        data = np.load(npz_path)
        input_array = data['x']
        output_array = data['y']
        return input_array, output_array

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_image = self.data[index]
        output_image = self.target[index]

        if self.transform is not None:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

    def __len__(self) -> int:
        return len(self.data)
