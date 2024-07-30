import torch
import numpy as np

from trackit.core.operator.numpy.bbox.format import bbox_get_width_and_height, bbox_get_center_point
from trackit.core.operator.numpy.bbox.utility.image import get_image_center_point
from trackit.core.operator.numpy.bbox.scale_and_translate import bbox_scale_and_translate
from trackit.core.operator.numpy.bbox.utility.image import is_bbox_intersecting_image, \
    bbox_clip_to_image_boundary_
from trackit.core.operator.scale_and_translate import scale_and_translate, scale_and_translate_subpixel, \
    reverse_scale_and_translation_parameters
from typing import Optional, Tuple


def _get_jittered_scale(scale: np.ndarray, scaling_jitter_factor: float, rng_engine: np.random.Generator):
    return scale / np.exp(rng_engine.standard_normal(2, dtype=np.float64) * scaling_jitter_factor)


def _get_jittered_translation(scale: np.ndarray, translation: np.ndarray, translation_jitter_factor: float,
                              bbox: np.ndarray, rng_engine: np.random.Generator):
    assert bbox.ndim == 1
    wh = bbox_get_width_and_height(bbox)
    max_translate = (wh * scale).sum() * 0.5 * translation_jitter_factor
    return rng_engine.uniform(low=-1, high=1, size=2) * max_translate + translation


def _get_scale_from_area_factor(bbox: np.ndarray, area_factor: float, output_size: np.ndarray):
    wh = bbox_get_width_and_height(bbox)
    area = wh + (area_factor - 1) * (wh.sum() * 0.5)
    scale = np.sqrt(output_size.prod() / area.prod())
    return scale.repeat(2)


def get_scale_and_translation_factors(bbox: np.ndarray, area_factor: float, output_size: np.ndarray):
    scale = _get_scale_from_area_factor(bbox, area_factor, output_size)

    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    return scale, target_center - source_center * scale


def get_jittered_scale_and_translation_factors(bbox: np.ndarray, area_factor: float, output_size: np.ndarray,
                                               scale_jitter_factor: float, translation_jitter_factor: float,
                                               rng_engine: np.random.Generator):
    scale = _get_scale_from_area_factor(bbox, area_factor, output_size)
    scale = _get_jittered_scale(scale, scale_jitter_factor, rng_engine)

    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    translation = target_center - source_center * scale

    translation = _get_jittered_translation(scale, translation, translation_jitter_factor, bbox, rng_engine)
    return scale, translation


def get_siamfc_cropping_params(bbox: np.ndarray, area_factor: float, output_size: np.ndarray) -> np.ndarray:
    scale, translation = get_scale_and_translation_factors(bbox, area_factor, output_size)
    return np.stack((scale, translation), axis=0)


def apply_siamfc_cropping(
        image: torch.Tensor, output_size: np.ndarray, curation_parameter: np.ndarray,
        interpolation_mode: str, align_corners: bool = False, image_mean: Optional[torch.Tensor] = None,
        out_image: Optional[torch.Tensor] = None, out_image_mean: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    assert image.dtype in (torch.float32, torch.float64), "image.dtype must be float32 or float64, but got {}".format(
        image.dtype)
    if image_mean is None:
        image_mean = torch.mean(image, dim=(-2, -1), out=out_image_mean)
    else:
        if out_image_mean is not None:
            out_image_mean[:] = image_mean
    scale, translation = curation_parameter

    cropped_image, adjusted_scale, adjusted_translation = scale_and_translate(
        image, output_size, scale, translation, image_mean,
        interpolation_mode, align_corners,
        out_image, return_adjusted_params=True)

    adjusted_cropping_params = np.stack((adjusted_scale, adjusted_translation), axis=-2)
    return cropped_image, image_mean, adjusted_cropping_params


def apply_siamfc_cropping_subpixel(
        image: torch.Tensor, output_size: np.ndarray, curation_parameter: np.ndarray,
        interpolation_mode: str, align_corners: bool = False, padding_mode: str = 'zeros'
) -> torch.Tensor:
    assert image.dtype in (torch.float32, torch.float64), \
        "image.dtype must be float32 or float64, but got {}".format(image.dtype)
    if curation_parameter.ndim == 3:
        scale, translation = (np.squeeze(p, axis=1) for p in np.split(curation_parameter, 2, axis=1))
    else:
        scale, translation = curation_parameter
    cropped_image = scale_and_translate_subpixel(
        image, output_size.tolist(), scale, translation,
        interpolation_mode, align_corners, padding_mode)
    return cropped_image


def prepare_siamfc_cropping_with_augmentation(
        box: np.ndarray, area_factor: float, output_size: np.ndarray,
        scale_jitter_factor: float, translation_jitter_factor: float,
        rng_engine: np.random.Generator = np.random.default_rng(),
        min_object_size: np.ndarray = np.array((1., 1.)),
        max_object_size: np.ndarray = np.array((np.inf, np.inf)),
        min_object_ratio: float = 0.0,
        max_object_ratio: float = 1.0,
        retry_count: int = 10):
    is_positive = True
    count = -1
    while True:
        count += 1
        if scale_jitter_factor == 0 and translation_jitter_factor == 0:
            scale, translation = get_scale_and_translation_factors(box, area_factor, output_size)
            break

        scale, translation = get_jittered_scale_and_translation_factors(
            box, area_factor, output_size,
            scale_jitter_factor, translation_jitter_factor,
            rng_engine)

        output_bbox = bbox_scale_and_translate(box, scale, translation)

        if not is_bbox_intersecting_image(output_bbox, output_size):
            if count > retry_count:
                is_positive = False
                break
            continue

        bbox_clip_to_image_boundary_(output_bbox, output_size)
        bbox_wh = bbox_get_width_and_height(output_bbox)

        if np.any(bbox_wh < min_object_size) or \
                np.any(bbox_wh > max_object_size) or \
                (np.prod(bbox_wh) > (output_size.prod() * max_object_ratio)) or \
                (np.prod(bbox_wh) < (output_size.prod() * min_object_ratio)):
            if count > retry_count:
                is_positive = False
                break
            continue
        break

    cropping_params = np.stack((scale, translation))

    return cropping_params, is_positive


def apply_siamfc_cropping_to_boxes(boxes: np.ndarray, cropping_params: np.ndarray):
    if boxes.ndim == 1:
        assert cropping_params.ndim == 2
        scale, translation = cropping_params
    elif boxes.ndim == 2:
        if cropping_params.ndim == 2:
            cropping_params = np.tile(np.expand_dims(cropping_params, axis=0), (boxes.shape[0], 1, 1))
        else:
            assert cropping_params.ndim == 3
        scale, translation = (np.squeeze(p, axis=1) for p in np.split(cropping_params, 2, axis=1))
    else:
        raise ValueError(f"Invalid boxes shape: {boxes.shape}")
    return bbox_scale_and_translate(boxes, scale, translation)


def reverse_siamfc_cropping_params(cropping_params: np.ndarray) -> np.ndarray:
    if cropping_params.ndim == 2:
        scale, translation = cropping_params
    else:
        scale, translation = (np.squeeze(p, axis=1) for p in np.split(cropping_params, 2, axis=-2))
    return np.stack(reverse_scale_and_translation_parameters(scale, translation), axis=-2)


def scale_siamfc_cropping_params(cropping_params: np.ndarray, old_output_size: np.ndarray, new_output_size: np.ndarray):
    scale_factor = new_output_size / old_output_size
    return cropping_params * scale_factor
