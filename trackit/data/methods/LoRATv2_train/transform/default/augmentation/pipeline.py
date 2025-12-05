from typing import Sequence, Union, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AnnotatedImage:
    image: torch.Tensor
    bbox: np.ndarray


class ImageOnlyAugmentation:
    def __call__(self, images: Sequence[torch.Tensor], rng_engine: np.random.Generator) -> Sequence[torch.Tensor]:
        raise NotImplementedError


class Augmentation:
    def __call__(self, images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray], rng_engine: np.random.Generator) -> Tuple[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        raise NotImplementedError


@dataclass
class AugmentationConfig:
    targets: Sequence[str]
    target_selector: Optional[slice]
    augmentation_fn: Union[Augmentation, ImageOnlyAugmentation]
    joint: bool


class AugmentationPipeline:
    def __init__(self, augmentations: Sequence[AugmentationConfig]):
        self.augmentations = augmentations

    def __call__(self, context: Dict[str, Sequence[AnnotatedImage]], rng_engine: np.random.Generator):
        for augmentation in self.augmentations:
            targets = augmentation.targets
            target_selector = augmentation.target_selector
            images, target_lengths = self._collect_images(context, targets)
            augmentation_fn = augmentation.augmentation_fn
            augmentation_with_box = isinstance(augmentation_fn, Augmentation)

            if augmentation_with_box:
                bboxes = self._collect_bboxes(context, targets)
                if target_selector is not None:
                    images[target_selector], bboxes[target_selector] = self._apply_augmentation_with_boxes(images[target_selector], bboxes[target_selector], augmentation, rng_engine)
                else:
                    images, bboxes = self._apply_augmentation_with_boxes(images, bboxes, augmentation, rng_engine)
            else:
                if target_selector is not None:
                    images[target_selector] = self._apply_image_only_augmentation(images[target_selector], augmentation, rng_engine)
                else:
                    images = self._apply_image_only_augmentation(images, augmentation, rng_engine)

            self._update_context(context, targets, images, bboxes if augmentation_with_box else None)

    def _collect_images(self, context: Dict[str, Sequence[AnnotatedImage]], targets: Sequence[str]
                        ) -> Tuple[Sequence[torch.Tensor], Sequence[int]]:
        images, target_lengths = [], []
        for target in targets:
            for annotated_image in context[target]:
                images.append(annotated_image.image)
            target_lengths.append(len(context[target]))
        return images, target_lengths

    def _collect_bboxes(self, context: Dict[str, Sequence[AnnotatedImage]], targets: Sequence[str]
                        ) -> Sequence[np.ndarray]:
        return [annotated_image.bbox for target in targets for annotated_image in context[target]]

    def _apply_augmentation_with_boxes(self, images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray],
                                       augmentation: AugmentationConfig, rng_engine: np.random.Generator
                                       ) -> Tuple[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        augmentation_fn = augmentation.augmentation_fn
        if augmentation.joint:
            return augmentation_fn(images, bboxes, rng_engine)
        else:
            new_images, new_bboxes = [], []
            for image, bbox in zip(images, bboxes):
                augmented_images, augmented_bboxes = augmentation_fn([image], [bbox], rng_engine)
                new_images.append(augmented_images[0])
                new_bboxes.append(augmented_bboxes[0])
            return new_images, new_bboxes

    def _apply_image_only_augmentation(self, images: Sequence[torch.Tensor], augmentation: AugmentationConfig,
                                       rng_engine: np.random.Generator) -> Sequence[torch.Tensor]:
        augmentation_fn = augmentation.augmentation_fn
        if augmentation.joint:
            return augmentation_fn(images, rng_engine)
        else:
            return [augmentation_fn([image], rng_engine)[0] for image in images]

    def _update_context(self, context: Dict[str, Sequence[AnnotatedImage]], targets: Sequence[str],
                        images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray] = None):
        index = 0
        for target in targets:
            for annotated_image in context[target]:
                annotated_image.image = images[index]
                if bboxes is not None:
                    annotated_image.bbox = bboxes[index]
                index += 1
