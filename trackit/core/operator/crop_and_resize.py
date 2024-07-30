from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


def crop_and_resize(image: torch.Tensor, bbox: np.ndarray, target_size: Union[Tuple[int, int], np.ndarray], interpolation_mode: str, interpolation_align_corners: bool):
    assert image.ndim == 3
    assert bbox.ndim == 1
    assert bbox.shape[0] == 4
    if np.issubdtype(bbox.dtype, np.floating):
        bbox = bbox_rasterize(bbox)
    cropped_region = image[:, bbox[1]: bbox[3], bbox[0]: bbox[2]]
    return F.interpolate(cropped_region.unsqueeze(0), (target_size[0], target_size[1]), mode=interpolation_mode, align_corners=interpolation_align_corners).squeeze(0)
