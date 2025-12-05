# modified from https://github.com/facebookresearch/deit/blob/main/augment.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
from typing import Sequence

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import ImageFilter, ImageOps, Image

from .pipeline import ImageOnlyAugmentation


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, images: Sequence[Image.Image], rng_engine: np.random.Generator):
        if self.prob == 1 or (self.prob > 0 and rng_engine.random() <= self.prob):
            filter_ = ImageFilter.GaussianBlur(
                radius=rng_engine.uniform(self.radius_min, self.radius_max)
            )
            images = tuple(img.filter(filter_) for img in images)

        return images


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, images: Sequence[Image.Image], rng_engine: np.random.Generator):
        if self.p == 1 or (self.p > 0 and rng_engine.random() <= self.p):
            images = tuple(ImageOps.solarize(img) for img in images)
        return images


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, images: Sequence[Image.Image], rng_engine: np.random.Generator):
        if self.p == 1 or (self.p > 0 and rng_engine.random() <= self.p):
            images = tuple(self.transf(img) for img in images)
        return images


class DeiT3Augmentation(ImageOnlyAugmentation):
    def __init__(self):
        self.augmentations = ([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)])

    def __call__(self, images: Sequence[torch.Tensor], rng_engine: np.random.Generator) -> Sequence[torch.Tensor]:
        images = tuple(F.to_pil_image(img, mode='RGB') for img in images)

        aug_index = rng_engine.choice(3)
        aug = self.augmentations[aug_index]

        images = aug(images, rng_engine)

        images = tuple(F.to_tensor(img) for img in images)
        return images
