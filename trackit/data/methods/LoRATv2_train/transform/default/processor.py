from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.modules.utils import _pair

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import prepare_siamfc_cropping_with_augmentation, apply_siamfc_cropping, \
    apply_siamfc_cropping_to_boxes
from typing import Optional, Sequence, Mapping, Iterable
from trackit.data.protocol.train_input import TrainData
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.data.utils.collation_helper import collate_element_as_torch_tensor

from ..._types import TemporalTrackerTrainingSample, SOTFrameInfo
from .. import TemporalTrackerTrain_DataTransform
from .augmentation import AugmentationPipeline
from .plugin import ExtraTransform, ExtraTransform_DataCollector


@dataclass(frozen=True)
class SiamFCCroppingParameter:
    output_size: np.ndarray
    area_factor: float
    scale_jitter_factor: float = 0.
    translation_jitter_factor: float = 0.
    output_min_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((0., 0.)))  # (width, height)
    output_min_object_size_in_ratio: float = 0.  # (width, height)
    output_max_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((float("inf"), float("inf"))))  # (width, height)
    output_max_object_size_in_ratio: float = 1.  # (width, height)
    interpolation_mode: str = 'bilinear'
    interpolation_align_corners: bool = False

    def __post_init__(self):
        if not isinstance(self.output_size, np.ndarray):
            super().__setattr__('output_size', np.array(self.output_size))
        if not isinstance(self.output_min_object_size_in_pixel, np.ndarray):
            super().__setattr__('output_min_object_size_in_pixel', np.array(_pair(self.output_min_object_size_in_pixel)))
        if not isinstance(self.output_max_object_size_in_pixel, np.ndarray):
            super().__setattr__('output_max_object_size_in_pixel', np.array(_pair(self.output_max_object_size_in_pixel)))


class DefaultTemporalTrackerProcessor(TemporalTrackerTrain_DataTransform):
    def __init__(self,
                 template_siamfc_cropping_parameters: SiamFCCroppingParameter | Iterable[SiamFCCroppingParameter],
                 search_region_siamfc_cropping_parameters: SiamFCCroppingParameter | Iterable[SiamFCCroppingParameter],
                 augmentation_pipeline: AugmentationPipeline,
                 static_image_augmentation_pipeline: AugmentationPipeline,
                 norm_stats_dataset_name: str,
                 additional_processors: Optional[Sequence[ExtraTransform]] = None,
                 visualize: bool = False):
        if isinstance(template_siamfc_cropping_parameters, SiamFCCroppingParameter):
            self.template_siamfc_cropping_operators = SiamFCCropping(template_siamfc_cropping_parameters)
        else:
            self.template_siamfc_cropping_operators = tuple(
                SiamFCCropping(template_siamfc_cropping_parameter) for template_siamfc_cropping_parameter in
                template_siamfc_cropping_parameters)
        if isinstance(search_region_siamfc_cropping_parameters, SiamFCCroppingParameter):
            self.search_region_siamfc_cropping_operators = SiamFCCropping(search_region_siamfc_cropping_parameters)
        else:
            self.search_region_siamfc_cropping_operators = tuple(
                SiamFCCropping(search_region_siamfc_cropping_parameter) for search_region_siamfc_cropping_parameter in
                search_region_siamfc_cropping_parameters)

        self.augmentation_pipeline = augmentation_pipeline
        self.static_image_augmentation_pipeline = static_image_augmentation_pipeline
        self.additional_processors = additional_processors

        self.image_normalize_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.norm_stats_dataset_name = norm_stats_dataset_name
        self.visualize = visualize

    def __call__(self, training_sample: TemporalTrackerTrainingSample, rng_engine: np.random.Generator):
        context = {}
        template_siamfc_cropping_operators = tuple(
            self.template_siamfc_cropping_operators for _ in range(len(training_sample.templates))) if isinstance(
            self.template_siamfc_cropping_operators, SiamFCCropping) else self.template_siamfc_cropping_operators
        for i, (template, cropping_operator) in enumerate(
                zip(training_sample.templates, template_siamfc_cropping_operators)):
            context[f'z_{i}_bbox'] = template.object_bbox
            assert cropping_operator.prepare(f'z_{i}', rng_engine, context)

        search_region_siamfc_cropping_operators = tuple(self.search_region_siamfc_cropping_operators for _ in
                                                        range(len(training_sample.search_regions))) if isinstance(
            self.search_region_siamfc_cropping_operators,
            SiamFCCropping) else self.search_region_siamfc_cropping_operators
        for i, (search_region, cropping_operator) in enumerate(
                zip(training_sample.search_regions, search_region_siamfc_cropping_operators)):
            context[f'x_{i}_bbox'] = search_region.object_bbox
            if not cropping_operator.prepare(f'x_{i}', rng_engine, context):
                return None

        num_templates = len(training_sample.templates)
        num_search_regions = len(training_sample.search_regions)
        image_decoding_cache = {}
        for i in range(num_templates):
            _decode_image_with_cache(f'z_{i}', training_sample.templates[i], image_decoding_cache, context)
        for i in range(num_search_regions):
            _decode_image_with_cache(f'x_{i}', training_sample.search_regions[i], image_decoding_cache, context)

        is_same_image = len(image_decoding_cache) == 1
        del image_decoding_cache

        for i, (template, cropping_operator) in enumerate(zip(training_sample.templates, self.template_siamfc_cropping_operators)):
            cropping_operator.do(f'z_{i}', context)
        for i, (search_region, cropping_operator) in enumerate(zip(training_sample.search_regions, self.search_region_siamfc_cropping_operators)):
            cropping_operator.do(f'x_{i}', context)

        self._do_augmentation(context, rng_engine, is_same_image, num_templates, num_search_regions)

        for i in range(num_templates):
            _bbox_clip_to_image_boundary_(context[f'z_{i}_cropped_bbox'], context[f'z_{i}_cropped_image'])
            self.image_normalize_transform_(context[f'z_{i}_cropped_image'])
        for i in range(num_search_regions):
            _bbox_clip_to_image_boundary_(context[f'x_{i}_cropped_bbox'], context[f'x_{i}_cropped_image'])
            self.image_normalize_transform_(context[f'x_{i}_cropped_image'])

        data = {}

        if self.additional_processors is not None:
            for processor in self.additional_processors:
                processor(training_sample, context, data, rng_engine)

        if self.visualize:
            from trackit.data.context.worker import get_current_worker_info
            from .visualization import visualize_temporal_processor
            output_path = get_current_worker_info().get_output_path()
            if output_path is not None:
                visualize_temporal_processor(output_path, training_sample, context, self.norm_stats_dataset_name)


        for i in range(num_templates):
            data[f'z_{i}'] = context[f'z_{i}_cropped_image']
        for i in range(num_search_regions):
            data[f'x_{i}'] = context[f'x_{i}_cropped_image']

        return data

    def _do_augmentation(self, context: dict, rng_engine: np.random.Generator, is_same_image: bool,
                         num_templates: int, num_search_regions: int):
        from .augmentation import AnnotatedImage
        augmentation_context = {}
        augmentation_context['template'] = [AnnotatedImage(context[f'z_{i}_cropped_image'], context[f'z_{i}_cropped_bbox']) for i in range(num_templates)]
        augmentation_context['search_region'] = [AnnotatedImage(context[f'x_{i}_cropped_image'], context[f'x_{i}_cropped_bbox']) for i in range(num_search_regions)]

        if is_same_image:
            self.static_image_augmentation_pipeline(augmentation_context, rng_engine)
        else:
            self.augmentation_pipeline(augmentation_context, rng_engine)

        for i in range(num_templates):
            context[f'z_{i}_cropped_image'] = augmentation_context['template'][i].image
            context[f'z_{i}_cropped_bbox'] = augmentation_context['template'][i].bbox

        for i in range(num_search_regions):
            context[f'x_{i}_cropped_image'] = augmentation_context['search_region'][i].image
            context[f'x_{i}_cropped_bbox'] = augmentation_context['search_region'][i].bbox


def _bbox_clip_to_image_boundary_(bbox: np.ndarray, image: torch.Tensor):
    h, w = image.shape[-2:]
    bbox_clip_to_image_boundary_(bbox, np.array((w, h)))
    assert bbox_is_valid(bbox), f'bbox:\n{bbox}\nimage_size:\n{image.shape}'


class SiamFCCropping:
    def __init__(self, siamfc_cropping_parameter: SiamFCCroppingParameter):
        self.siamfc_cropping_parameter = siamfc_cropping_parameter

    def prepare(self, name: str, rng_engine: np.random.Generator, context: dict):
        cropping_parameter, is_success = \
            prepare_siamfc_cropping_with_augmentation(context[f'{name}_bbox'], self.siamfc_cropping_parameter.area_factor,
                                                      self.siamfc_cropping_parameter.output_size,
                                                      self.siamfc_cropping_parameter.scale_jitter_factor,
                                                      self.siamfc_cropping_parameter.translation_jitter_factor,
                                                      rng_engine,
                                                      self.siamfc_cropping_parameter.output_min_object_size_in_pixel,
                                                      self.siamfc_cropping_parameter.output_max_object_size_in_pixel,
                                                      self.siamfc_cropping_parameter.output_min_object_size_in_ratio,
                                                      self.siamfc_cropping_parameter.output_max_object_size_in_ratio)
        
        context[f'{name}_cropping_parameter'] = cropping_parameter
        context[f'{name}_cropping_is_success'] = is_success
        return is_success

    def do(self, name: str, context: dict, normalized: bool = True):
        cropping_parameter = context[f'{name}_cropping_parameter']
        image = context[f'{name}_image']
        image_cropped, context[f'{name}_image_mean'], cropping_parameter = \
            apply_siamfc_cropping(image, self.siamfc_cropping_parameter.output_size, cropping_parameter,
                                  interpolation_mode=self.siamfc_cropping_parameter.interpolation_mode,
                                  align_corners=self.siamfc_cropping_parameter.interpolation_align_corners)
        if normalized:
            image_cropped.div_(255.)
        context[f'{name}_cropping_parameter'] = cropping_parameter
        context[f'{name}_cropped_image'] = image_cropped
        if context[f'{name}_cropping_is_success']:
            context[f'{name}_cropped_bbox'] = apply_siamfc_cropping_to_boxes(context[f'{name}_bbox'], cropping_parameter)


def _decode_image_with_cache(name: str, frame: SOTFrameInfo, cache: dict, context: dict):
    if frame.image in cache:
        context[f'{name}_image'] = cache[frame.image]
        return
    image = frame.image()

    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1)).contiguous()
    image = image.to(torch.float32)

    cache[frame.image] = image
    context[f'{name}_image'] = image


class TemporalTrackerDefaultProcessorBatchCollator:
    def __init__(self,
                 dtype: torch.dtype,
                 additional_collators: Optional[Sequence[ExtraTransform_DataCollector]] = None):
        self.dtype = dtype
        self.additional_collators = additional_collators

    def __call__(self, batch: Sequence[Mapping], collated: TrainData):
        index = 0
        z_list = []
        while f'z_{index}' in batch[0]:
            z_list.append(collate_element_as_torch_tensor(batch, f'z_{index}').to(self.dtype))
            index += 1
        collated.input['z_list'] = z_list

        index = 0
        x_list = []
        while f'x_{index}' in batch[0]:
            x_list.append(collate_element_as_torch_tensor(batch, f'x_{index}').to(self.dtype))
            index += 1
        collated.input['x_list'] = x_list

        if self.additional_collators is not None:
            for additional_collator in self.additional_collators:
                additional_collator(batch, collated)
