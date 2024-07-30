import numpy as np


def bbox_scale(bbox: np.ndarray, scale: np.ndarray):
    return bbox_scale_(bbox.copy(), scale)


def bbox_scale_(bbox: np.ndarray, scale: np.ndarray):
    bbox[..., ::2] *= scale[..., (0, )]
    bbox[..., 1::2] *= scale[..., (1, )]
    return bbox
