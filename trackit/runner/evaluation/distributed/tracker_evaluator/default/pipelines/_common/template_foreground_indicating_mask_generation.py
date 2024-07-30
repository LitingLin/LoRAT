from typing import Tuple

import torch

from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid

from ....default import TrackerEvaluationPipeline, TrackerEvaluationPipeline_Context
from ....components.tensor_cache import CacheService, TensorCache


class TemplateFeatForegroundMaskGeneration(TrackerEvaluationPipeline):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int], device: torch.device,
                 provide_during_tracking: bool = True):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
        self.device = device
        self.background_value = 0
        self.foreground_value = 1
        self.provide_during_tracking = provide_during_tracking

    def start(self, max_batch_size: int, global_objects: dict):
        self.template_mask_cache = CacheService(max_batch_size, TensorCache(max_batch_size, (self.template_feat_size[1], self.template_feat_size[0]), self.device, torch.long))

    def stop(self, global_objects: dict):
        del self.template_mask_cache

    def prepare_initialization(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        do_init_task_ids = []
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                current_init_context = task.tracker_do_init_context
                template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value, dtype=torch.long)
                template_cropped_bbox = get_foreground_bounding_box(current_init_context.gt_bbox, current_init_context.input_data['curation_parameter'], self.stride)
                assert bbox_is_valid(template_cropped_bbox)
                template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
                template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = self.foreground_value
                self.template_mask_cache.put(task.id, template_mask.to(self.device))
                do_init_task_ids.append(task.id)
        if not self.provide_during_tracking:
            if len(do_init_task_ids) > 0:
                model_input_params['z_feat_mask'] = self.template_mask_cache.get_batch(do_init_task_ids)

    def prepare_tracking(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        if self.provide_during_tracking:
            do_track_task_ids = []
            for task in context.input_data.tasks:
                if task.tracker_do_tracking_context is not None:
                    do_track_task_ids.append(task.id)

            if len(do_track_task_ids) > 0:
                model_input_params['z_feat_mask'] = self.template_mask_cache.get_batch(do_track_task_ids)

    def end(self, context: TrackerEvaluationPipeline_Context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.template_mask_cache.delete(task.id)
