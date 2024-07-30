import numpy as np


def bbox_compute_area(bbox: np.ndarray):
    return np.prod(bbox[..., 2:] - bbox[..., :2], axis=-1)
