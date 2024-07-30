import numpy as np


def bbox_is_valid(bbox: np.ndarray):
    validity = np.all(np.isfinite(bbox), axis=-1)
    validity = np.logical_and(np.all(bbox[..., :2] < bbox[..., 2:], axis=-1), validity)
    return validity
