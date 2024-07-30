import numpy as np


def bbox_rasterize(bbox: np.ndarray, eps: float = 1e-4, dtype=np.int32):
    assert np.issubdtype(bbox.dtype, np.floating)
    bbox = bbox_rasterize_(bbox.copy(), eps)
    return bbox.astype(dtype)


def bbox_rasterize_(bbox: np.ndarray, eps: float = 1e-4):
    assert np.issubdtype(bbox.dtype, np.floating)
    bbox[..., 2] += (1 - eps)
    bbox[..., 3] += (1 - eps)
    return np.floor(bbox, out=bbox)

