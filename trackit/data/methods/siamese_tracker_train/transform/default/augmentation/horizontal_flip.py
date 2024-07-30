from typing import Sequence, Tuple
import torch
import numpy as np
import torchvision.transforms.functional as F
from .pipeline import Augmentation
from trackit.core.operator.numpy.bbox.flip import bbox_horizontal_flip


class HorizontalFlipAugmentation(Augmentation):
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray], rng_engine: np.random.Generator) -> Tuple[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        if self.probability > 0 and rng_engine.random() < self.probability:
            assert all(len(img.shape) == 3 for img in images)  # CHW
            images = tuple(F.hflip(img) for img in images)
            all_image_w = tuple(img.shape[-1] for img in images)
            bboxes = tuple(bbox_horizontal_flip(bbox, w) for bbox, w in zip(bboxes, all_image_w))
        return images, bboxes
