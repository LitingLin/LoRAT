import numpy as np


def point_horizontal_flip(point: np.ndarray, width: int):
    point = point.copy()
    point[..., 0] = width - point[..., 0]
    return point


def point_vertical_flip(point: np.ndarray, height: int):
    point = point.copy()
    point[..., 1] = height - point[..., 1]
    return point


def point_diagonal_flip(point: np.ndarray, width: int, height: int):
    point = point.copy()
    point[..., 0] = width - point[..., 0]
    point[..., 1] = height - point[..., 1]
    return point


def point_flip(point: np.ndarray, width: int, height: int, flip_horizontal: bool, flip_vertical: bool):
    if flip_horizontal:
        point = point_horizontal_flip(point, width)
    if flip_vertical:
        point = point_vertical_flip(point, height)
    return point
