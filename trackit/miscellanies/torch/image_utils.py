import numpy as np
import torch


def convert_ndarray_image_to_torch_tensor(image: np.ndarray, to_float32: bool = True) -> torch.Tensor:
    image = torch.from_numpy(image).permute(2, 0, 1)
    if to_float32:
        image = image.float()
    return image


def convert_torch_tensor_image_to_ndarray(image: torch.Tensor, to_uint8: bool = True) -> np.ndarray:
    image = image.permute(1, 2, 0)
    if to_uint8:
        image = image.byte()
    return image.numpy()
