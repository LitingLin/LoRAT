import numpy as np
from PIL import Image


def from_pil_image(pil_image: Image) -> np.ndarray:
    return np.array(pil_image)


def to_pil_image(image: np.ndarray, palette=None) -> Image:
    pil_image = Image.fromarray(image)
    if palette is not None:
        pil_image.putpalette(palette)
    return pil_image
