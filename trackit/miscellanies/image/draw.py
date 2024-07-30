import numpy as np
from typing import Tuple, Union
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


def draw_box_on_image_(image: np.ndarray, box: np.ndarray, color: Union[np.ndarray, Tuple[int, int, int]], thickness: int = 1) -> np.ndarray:
    """
    Draw a box on the image.

    Parameters:
        image: numpy array, the input image (shape: (H, W, C))
        box: numpy array, the box to be drawn (shape: (4,)), in the format (x1, y1, x2, y2)
        color: numpy array or tuple, the color of the box lines
        thickness: int, thickness of the box lines (default: 1)

    Returns:
        numpy array, the image with the box drawn on it
    """
    assert np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert box.ndim == 1
    assert box.size == 4
    assert isinstance(color, np.ndarray) or isinstance(color, tuple)
    assert isinstance(thickness, int)
    assert thickness > 0

    box = box.clip(min=0)
    if np.issubdtype(box.dtype, np.floating):
        assert np.isfinite(box).all()
        box = bbox_rasterize(box)
    x1, y1, x2, y2 = box.tolist()

    image[y1: y1 + thickness, x1: x2] = color  # Top line
    image[y2: y2 + thickness, x1: x2] = color  # Bottom line
    image[y1: y2, x1: x1 + thickness] = color  # Left line
    image[y1: y2, x2: x2 + thickness] = color  # Right line
    return image
