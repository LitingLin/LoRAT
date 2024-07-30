import numpy as np


def bbox_scale_and_translate(bbox: np.ndarray, scale: np.ndarray, translation: np.ndarray):
    return bbox_scale_and_translate_(bbox.copy(), scale, translation)


def bbox_scale_and_translate_(bbox: np.ndarray, scale: np.ndarray, translation: np.ndarray):
    bbox[..., ::2] *= scale[..., (0, )]
    bbox[..., ::2] += translation[..., (0, )]

    bbox[..., 1::2] *= scale[..., (1, )]
    bbox[..., 1::2] += translation[..., (1, )]
    return bbox
