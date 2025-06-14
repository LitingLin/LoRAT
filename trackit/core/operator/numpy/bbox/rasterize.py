import numpy as np


def bbox_rasterize(bbox: np.ndarray, eps: float = 1e-4, dtype=np.int32, end_exclusive: bool = True):
    assert np.issubdtype(bbox.dtype, np.floating)
    bbox = bbox_rasterize_(bbox.copy(), eps, end_exclusive)
    return bbox.astype(dtype)


def bbox_rasterize_(bbox: np.ndarray, eps: float = 1e-4, end_exclusive: bool = True):
    assert np.issubdtype(bbox.dtype, np.floating)
    bbox[..., 2] += (1 - eps)
    bbox[..., 3] += (1 - eps)
    np.floor(bbox, out=bbox)
    if not end_exclusive:
        bbox[..., 2] -= 1.
        bbox[..., 3] -= 1.
    return bbox
