# modified from https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py

from typing import Sequence, Union, Tuple
import torch
import numpy as np
import torchvision.transforms.functional as F
import numbers

from .pipeline import ImageOnlyAugmentation


def _check_input(value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(f"If {name} is a single number, it must be non negative.")
        value = [center - float(value), center + float(value)]
        if clip_first_on_zero:
            value[0] = max(value[0], 0.0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        value = [float(value[0]), float(value[1])]
    else:
        raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

    if not bound[0] <= value[0] <= value[1] <= bound[1]:
        raise ValueError(f"{name} values should be between {bound}, but got {value}.")

    # if value is 0 or (1., 1.) for brightness/contrast/saturation
    # or (0., 0.) for hue, do nothing
    if value[0] == value[1] == center:
        return None
    else:
        return tuple(value)


class ColorJitter(ImageOnlyAugmentation):
    def __init__(self,
                 brightness: Union[float, Tuple[float, float]] = 0,
                 contrast: Union[float, Tuple[float, float]] = 0,
                 saturation: Union[float, Tuple[float, float]] = 0,
                 hue: Union[float, Tuple[float, float]] = 0):
        self.brightness = _check_input(brightness, "brightness")
        self.contrast = _check_input(contrast, "contrast")
        self.saturation = _check_input(saturation, "saturation")
        self.hue = _check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def __call__(self, images: Sequence[torch.Tensor], rng_engine: np.random.Generator) -> Sequence[torch.Tensor]:
        fn_idx = rng_engine.permutation(4)

        brightness_factor = None if self.brightness is None else float(rng_engine.uniform(self.brightness[0], self.brightness[1]))
        contrast_factor = None if self.contrast is None else float(rng_engine.uniform(self.contrast[0], self.contrast[1]))
        saturation_factor = None if self.saturation is None else float(rng_engine.uniform(self.saturation[0], self.saturation[1]))
        hue_factor = None if self.hue is None else float(rng_engine.uniform(self.hue[0], self.hue[1]))

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                images = tuple(F.adjust_brightness(img, brightness_factor) for img in images)
            elif fn_id == 1 and contrast_factor is not None:
                images = tuple(F.adjust_contrast(img, contrast_factor) for img in images)
            elif fn_id == 2 and saturation_factor is not None:
                images = tuple(F.adjust_saturation(img, saturation_factor) for img in images)
            elif fn_id == 3 and hue_factor is not None:
                images = tuple(F.adjust_hue(img, hue_factor) for img in images)
        return images
