from typing import Dict, Any, Tuple, Callable, Sequence, Optional
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn

from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.runners.evaluation.distributed import EvaluatorContext
from ....components.post_process import TrackerOutputPostProcess
from ....components.tensor_cache import CacheService, TensorCache
from ..interface import TrackingPipeline
from ...types import TrackingPipeline_ResultHolder, TrackerEvaluationPipeline_Context
from ..utils.siamfc_post_process import siamfc_post_process
from .search_region_cropping import SiamFCCroppingParameterProvider
from .template_foreground_indicating_mask_generation import TemplateFeatForegroundMaskGeneration
from .kv_cache import KVCacheMaintainer


@dataclass
class _LocalContext:
    fps: Optional[float] = None
    curation_parameter_provider: SiamFCCroppingParameterProvider = None


class LoRATv2_Phase_1_TrackingPipeline(TrackingPipeline):
    def __init__(self, device: torch.device,
                 curated_template_image_size: Tuple[int, int],
                 curated_search_image_size: Tuple[int, int],
                 search_curation_parameter_provider_factory: Callable[[], SiamFCCroppingParameterProvider],
                 template_feature_map_size: Tuple[int, int],
                 network_output_post_process: TrackerOutputPostProcess,
                 interpolation_mode: str, interpolation_align_corners: bool,
                 normalization_dataset_stats_type: str):
        self.curated_template_image_size = curated_template_image_size
        self.curated_search_image_size = curated_search_image_size
        self.search_curation_parameter_provider_factory = search_curation_parameter_provider_factory
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.kv_cache = KVCacheMaintainer(device)
        self.template_feat_foreground_mask_generation = TemplateFeatForegroundMaskGeneration(
            curated_template_image_size, template_feature_map_size, device)

        self.network_output_post_process = network_output_post_process
        self.device = device

        self.normalization_transform_ = get_dataset_norm_stats_transform(normalization_dataset_stats_type, inplace=True)

    def start(self, evaluator_context: EvaluatorContext, *_):
        max_batch_size = evaluator_context.max_batch_size
        max_concurrent_tasks = evaluator_context.max_batch_size * evaluator_context.num_input_data_streams

        curated_template_image_cache_shape = (max_batch_size, 3, self.curated_template_image_size[1], self.curated_template_image_size[0])
        curated_search_image_cache_shape = (max_batch_size, 3, self.curated_search_image_size[1], self.curated_search_image_size[0])

        self.curation_parameter_cache = np.full((max_batch_size, 2, 2), float('nan'), dtype=np.float64)

        self.all_tracking_task_contexts: Dict[Any, _LocalContext] = {}
        self.all_tracking_template_image_mean_cache = CacheService(TensorCache(max_concurrent_tasks, (3,), self.device))
        self.curated_template_image_cache = torch.empty(curated_template_image_cache_shape, dtype=evaluator_context.dtype,
                                                        device=self.device).fill_(float('nan'))
        self.curated_search_image_cache = torch.empty(curated_search_image_cache_shape, dtype=torch.float,
                                                      device=self.device).fill_(float('nan'))

        self.kv_cache.start(evaluator_context)
        self.network_output_post_process.start()
        self.dtype = evaluator_context.dtype
        self.template_feat_foreground_mask_generation.start(evaluator_context)

    def stop(self, evaluator_context: EvaluatorContext, *_):
        self.network_output_post_process.stop()
        self.kv_cache.stop()
        assert len(self.all_tracking_task_contexts) == 0
        self.template_feat_foreground_mask_generation.stop()
        del self.curation_parameter_cache
        del self.all_tracking_template_image_mean_cache
        del self.all_tracking_task_contexts
        del self.curated_template_image_cache
        del self.curated_search_image_cache
        del self.dtype

    def initialize(self, data: TrackerEvalData, model, context: TrackerEvaluationPipeline_Context, raw_model: nn.Module):
        num_initializing_sequence = 0
        do_initialization_task_ids = []
        for task in data.tasks:
            if task.task_creation_context is not None:
                self.kv_cache.allocate(task.id)
                self.all_tracking_task_contexts[task.id] = _LocalContext(task.task_creation_context.fps)
            if task.tracker_do_init_context is not None:
                init_context = task.tracker_do_init_context
                self.curated_template_image_cache[num_initializing_sequence, ...] = init_context.input_data['curated_image']
                self.all_tracking_template_image_mean_cache.put(task.id, init_context.input_data['image_mean'])
                num_initializing_sequence += 1
                search_image_curation_parameter_provider = self.search_curation_parameter_provider_factory()
                search_image_curation_parameter_provider.initialize(init_context.gt_bbox)
                self.all_tracking_task_contexts[task.id].curation_parameter_provider = search_image_curation_parameter_provider
                do_initialization_task_ids.append(task.id)
                self.template_feat_foreground_mask_generation.initialize(task.id, init_context.gt_bbox, init_context.input_data['curation_parameter'])

        if num_initializing_sequence > 0:
            template_curated_image = self.curated_template_image_cache[: num_initializing_sequence, ...]
            z_feat_mask = self.template_feat_foreground_mask_generation.get_batch(do_initialization_task_ids)
            kv_cache_params = self.kv_cache.get_input_params(do_initialization_task_ids)

            model({'action': 'init',
                   'z': template_curated_image,
                   'z_feat_mask': z_feat_mask,
                   **kv_cache_params})

    def track(self, data: TrackerEvalData, model, context: TrackerEvaluationPipeline_Context,
              result: TrackingPipeline_ResultHolder, raw_model: nn.Module):
        search_images = []
        cropping_parameter_providers = []
        task_ids = []
        task_ids_to_be_finalized = []
        image_size_list = []

        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                task_ids.append(task.id)
                search_image = task.tracker_do_tracking_context.input_data['image']
                H, W = search_image.shape[-2:]
                image_size_list.append(np.array((W, H), dtype=int))
                search_images.append(task.tracker_do_tracking_context.input_data['image'])
                cropping_parameter_providers.append(self.all_tracking_task_contexts[task.id].curation_parameter_provider)
                if task.do_task_finalization:
                    task_ids_to_be_finalized.append(task.id)

        num_tracking_sequence = 0

        for task_id, search_image, cropping_parameter_provider in zip(task_ids, search_images,
                                                                      cropping_parameter_providers):
            template_image_mean = self.all_tracking_template_image_mean_cache.get(task_id)
            curation_parameter = cropping_parameter_provider.get(np.array(self.curated_search_image_size))

            _, _, curation_parameter = \
                apply_siamfc_cropping(search_image.to(torch.float),
                                      np.array(self.curated_search_image_size), curation_parameter,
                                      self.interpolation_mode, self.interpolation_align_corners,
                                      template_image_mean,
                                      out_image=self.curated_search_image_cache[num_tracking_sequence, ...])
            self.curation_parameter_cache[num_tracking_sequence, ...] = curation_parameter
            num_tracking_sequence += 1

        if num_tracking_sequence > 0:
            context.temporary_objects['x_curation_parameter'] = self.curation_parameter_cache[:num_tracking_sequence]
            x = self.curated_search_image_cache[: num_tracking_sequence, ...]
            x.div_(255.)
            self.normalization_transform_(x)
            x = x.to(self.dtype)

            model_output = model({'action': 'track',
                                  'index': 0,
                                  'x': x,
                                  **self.kv_cache.get_input_params(task_ids)})

            post_process_output = self.network_output_post_process(model_output)
            predicted_scores, predicted_bounding_boxes = siamfc_post_process(
                post_process_output, self.curation_parameter_cache[: num_tracking_sequence, ...])

            assert len(task_ids) == len(predicted_scores) == len(predicted_bounding_boxes)

            for index, (task_id, image_size) in enumerate(zip(task_ids, image_size_list)):
                predicted_score = predicted_scores[index].item()
                predicted_bounding_box = predicted_bounding_boxes[index]
                self.all_tracking_task_contexts[task_id].curation_parameter_provider.update(
                    predicted_score, predicted_bounding_box, image_size)
                result.submit(task_id, predicted_bounding_box, predicted_score)

            assert result.is_all_submitted()

        for task in data.tasks:
            if task.do_task_finalization:
                self.all_tracking_template_image_mean_cache.delete(task.id)
                self.all_tracking_task_contexts.pop(task.id)
                self.kv_cache.free(task.id)
                self.template_feat_foreground_mask_generation.remove(task.id)
