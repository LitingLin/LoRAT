import numpy as np


def point_scale_and_translate(point: np.ndarray, scale: np.ndarray, translation: np.ndarray):
    return point_scale_and_translate_(point.copy(), scale, translation)


def point_scale_and_translate_(point: np.ndarray, scale: np.ndarray, translation: np.ndarray):
    point[..., (0, )] *= scale[..., (0, )]
    point[..., (0, )] += translation[..., (0, )]

    point[..., (1, )] *= scale[..., (1, )]
    point[..., (1, )] += translation[..., (1, )]
    return point
