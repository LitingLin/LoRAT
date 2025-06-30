import numpy as np
import torch
from dataclasses import dataclass, field

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import prepare_siamfc_cropping_with_augmentation, apply_siamfc_cropping, \
    apply_siamfc_cropping_to_boxes
from typing import Optional, Sequence, Mapping
from trackit.data.utils.collation_helper import collate_element_as_torch_tensor, collate_element_as_np_array
from trackit.data.protocol.train_input import TrainData
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.data import MainProcessDataPipeline
from trackit.core.runtime.metric_logger import get_current_metric_logger
from trackit.core.runtime.context.epoch import get_current_epoch_context

from ..._types import SiameseTrainingMultiPair, SOTFrameInfo
from .. import SPMTrackTrain_DataTransform
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


class SPMTrackDataProcessor(SPMTrackTrain_DataTransform):
    def __init__(self,
                 template_siamfc_cropping_parameter: SiamFCCroppingParameter,
                 search_region_siamfc_cropping_parameter: SiamFCCroppingParameter,
                 augmentation_pipeline: AugmentationPipeline,
                 norm_stats_dataset_name: str,
                 additional_processors: Optional[Sequence[ExtraTransform]] = None):
        self.template_siamfc_cropping = SiamFCCropping(template_siamfc_cropping_parameter)
        self.search_region_siamfc_cropping = SiamFCCropping(search_region_siamfc_cropping_parameter)

        self.augmentation_pipeline = augmentation_pipeline
        self.additional_processors = additional_processors

        self.image_normalize_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.norm_stats_dataset_name = norm_stats_dataset_name

    def __call__(self, training_pair: SiameseTrainingMultiPair, rng_engine: np.random.Generator):
        context = {'is_positive': training_pair.is_positive}
        for i, template in enumerate(training_pair.template):
            context[f'z_{i}_bbox'] = template.object_bbox
        for i, search in enumerate(training_pair.search):
            context[f'x_{i}_bbox'] = search.object_bbox
        for i in range(len(training_pair.template)):
            assert self.template_siamfc_cropping.prepare(f'z_{i}', rng_engine, context)
        for i in range(len(training_pair.search)):
            if not self.search_region_siamfc_cropping.prepare(f'x_{i}', rng_engine, context):
                return None

        is_positive = training_pair.is_positive

        image_decoding_cache = {}
        for i in range(len(training_pair.template)):
            _decode_image_with_cache(f'z_{i}', training_pair.template[i], image_decoding_cache, context)
        for i in range(len(training_pair.search)):
            _decode_image_with_cache(f'x_{i}', training_pair.search[i], image_decoding_cache, context)
        del image_decoding_cache
        for i in range(len(training_pair.template)):
            self.template_siamfc_cropping.do(f'z_{i}', context)
        for i in range(len(training_pair.search)):
            self.search_region_siamfc_cropping.do(f'x_{i}', context)
        self._do_augmentation(context, rng_engine, len(training_pair.template), len(training_pair.search))

        for i in range(len(training_pair.template)):
            _bbox_clip_to_image_boundary_(context[f'z_{i}_cropped_bbox'], context[f'z_{i}_cropped_image'])
            self.image_normalize_transform_(context[f'z_{i}_cropped_image'])
        for i in range(len(training_pair.search)):
            _bbox_clip_to_image_boundary_(context[f'x_{i}_cropped_bbox'], context[f'x_{i}_cropped_image'])
            self.image_normalize_transform_(context[f'x_{i}_cropped_image'])

        data = {}

        if self.additional_processors is not None:
            for processor in self.additional_processors:
                processor(training_pair, context, data, rng_engine, len(training_pair.template), len(training_pair.search))

        for i in range(len(training_pair.template)):
            data[f'z_{i}_cropped_image'] = context[f'z_{i}_cropped_image']
        for i in range(len(training_pair.search)):
            data[f'x_{i}_cropped_image'] = context[f'x_{i}_cropped_image']

        data['is_positive'] = is_positive

        return data

    def _do_augmentation(self, context: dict, rng_engine: np.random.Generator, template_count: int, search_count: int):
        from .augmentation import AnnotatedImage
        augmentation_context = {}

        for i in range(template_count):
            augmentation_context[f'template_{i}'] = [AnnotatedImage(context[f'z_{i}_cropped_image'], context[f'z_{i}_cropped_bbox'])]
        for i in range(search_count):
            augmentation_context[f'search_region_{i}'] = [AnnotatedImage(context[f'x_{i}_cropped_image'], context[f'x_{i}_cropped_bbox'])]
        self.augmentation_pipeline(augmentation_context, rng_engine)

        for i in range(template_count):
            context[f'z_{i}_cropped_image'] = augmentation_context[f'template_{i}'][0].image
            context[f'z_{i}_cropped_bbox'] = augmentation_context[f'template_{i}'][0].bbox
        #context['z_cropped_image'] = augmentation_context['template'][0].image
        #context['z_cropped_bbox'] = augmentation_context['template'][0].bbox

        for i in range(search_count):
            context[f'x_{i}_cropped_image'] = augmentation_context[f'search_region_{i}'][0].image
            context[f'x_{i}_cropped_bbox'] = augmentation_context[f'search_region_{i}'][0].bbox
        #context['x_cropped_image'] = augmentation_context['search_region'][0].image
        #context['x_cropped_bbox'] = augmentation_context['search_region'][0].bbox


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
        if is_success:
            context[f'{name}_cropping_parameter'] = cropping_parameter
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


class SPMTrackDataProcessorBatchCollator:
    def __init__(self, dtype: torch.dtype, additional_collators: Optional[Sequence[ExtraTransform_DataCollector]] = None):
        self.dtype = dtype
        self.additional_collators = additional_collators

    def __call__(self, batch: Sequence[Mapping], collated: TrainData):
        max_x = 0
        max_z = 0
        for k in batch[0].keys():
            parts = k.split('_')
            if parts[0] == 'z':
                max_z = max(max_z, int(parts[1]))
            if parts[0] == 'x':
                max_x = max(max_x, int(parts[1]))
        data_dict = {}
        for i in range(max_x + 1):
            data_dict[f'x_{i}'] = collate_element_as_torch_tensor(batch, f'x_{i}_cropped_image').to(self.dtype)
        for i in range(max_z + 1):
            data_dict[f'z_{i}'] = collate_element_as_torch_tensor(batch, f'z_{i}_cropped_image').to(self.dtype)
        collated.input.update(data_dict)
        collated.miscellanies.update({'is_positive': collate_element_as_np_array(batch, 'is_positive')})

        if self.additional_collators is not None:
            for additional_collator in self.additional_collators:
                additional_collator(batch, collated, max_z, max_x)


class SPMTrackDataProcessorLoggingHook(MainProcessDataPipeline):
    def pre_process(self, input_data: TrainData) -> TrainData:
        input_data.target['epoch'] = get_current_epoch_context().epoch
        is_positive = input_data.miscellanies['is_positive']
        positive_sample_ratio = (np.sum(is_positive) / len(is_positive)).item()
        get_current_metric_logger().log({'positive_pair': positive_sample_ratio})
        return input_data
