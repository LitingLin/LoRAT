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
        box = bbox_rasterize(box, end_exclusive=False)
    x1, y1, x2, y2 = box.tolist()

    H, W, _ = image.shape
    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(W, x2)
    y2_c = min(H, y2)
    # Top line
    # Starts at y1_c, ends at y1_c + thickness, but not exceeding y2_c
    y_end = min(y1_c + thickness, y2_c)
    image[y1_c: y_end, x1_c: x2_c] = color

    # Bottom line
    # Starts at y2_c - thickness (but not below y1_c), ends at y2_c
    y_start = max(y1_c, y2_c - thickness)
    image[y_start: y2_c, x1_c: x2_c] = color

    # Left line
    # Starts at x1_c, ends at x1_c + thickness, but not exceeding x2_c
    x_end = min(x1_c + thickness, x2_c)
    image[y1_c: y2_c, x1_c: x_end] = color

    # Right line
    # Starts at x2_c - thickness (but not below x1_c), ends at x2_c
    x_start = max(x1_c, x2_c - thickness)
    image[y1_c: y2_c, x_start: x2_c] = color
    return image
