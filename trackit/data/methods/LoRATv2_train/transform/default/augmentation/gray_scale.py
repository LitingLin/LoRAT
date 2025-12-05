from typing import Sequence
import numpy as np
import torch
import torchvision.transforms.functional as F

from .pipeline import ImageOnlyAugmentation


class GrayScaleAugmentation(ImageOnlyAugmentation):
    def __init__(self, gray_scale_probability: float):
        self.gray_scale_probability = gray_scale_probability

    def __call__(self, images: Sequence[torch.Tensor], rng_engine: np.random.Generator) -> Sequence[torch.Tensor]:
        if self.gray_scale_probability > 0 and rng_engine.random() < self.gray_scale_probability:
            images = tuple(F.rgb_to_grayscale(img, num_output_channels=3).contiguous() for img in images)
        return images
