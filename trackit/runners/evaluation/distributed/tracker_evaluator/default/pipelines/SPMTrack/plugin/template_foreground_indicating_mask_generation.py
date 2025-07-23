from typing import Tuple, Any
import numpy as np
import torch

from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.runners.evaluation.distributed import EvaluatorContext
from trackit.runners.evaluation.distributed.tracker_evaluator.components.tensor_cache import CacheService, TensorCache
from trackit.runners.evaluation.distributed.tracker_evaluator.default.types import TrackerEvaluationPipeline_Context
from ...utils.bbox_mask_gen import get_foreground_bounding_box
from . import TrackingPipelinePlugin
import functools

class TemplateFeatForegroundMaskGeneration(TrackingPipelinePlugin):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int], device: torch.device,
                 provide_during_tracking: bool = True):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
        self.device = device
        self.background_value = 0
        self.foreground_value = 1
        self.provide_during_tracking = provide_during_tracking

    def start(self, context: EvaluatorContext, global_objects: dict):
        max_concurrent = context.max_batch_size * context.num_input_data_streams
        self.template_mask_cache = CacheService(TensorCache(max_concurrent, (self.template_feat_size[1], self.template_feat_size[0]), self.device, torch.long))
        self.memory_masks = {}

    def stop(self, context: EvaluatorContext, global_objects: dict):
        assert self.template_mask_cache.empty()
        del self.template_mask_cache
        del self.memory_masks

    def prepare_initialization(self, data: TrackerEvalData, model_input_params: dict, *_):
        do_init_task_ids = []
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                current_init_context = task.tracker_do_init_context
                template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value, dtype=torch.long)
                template_cropped_bbox = get_foreground_bounding_box(current_init_context.gt_bbox, current_init_context.input_data['curation_parameter'], self.stride)
                bbox_clip_to_image_boundary_(template_cropped_bbox, np.array(self.template_feat_size))
                assert bbox_is_valid(template_cropped_bbox)
                template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
                template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = self.foreground_value
                self.template_mask_cache.put(task.id, template_mask.to(self.device))
                if task.id in self.memory_masks.keys():
                    self.memory_masks[task.id].append(template_mask.cpu())
                else:
                    self.memory_masks[task.id] = [template_mask.cpu()]
                do_init_task_ids.append(task.id)
        if not self.provide_during_tracking:
            if len(do_init_task_ids) > 0:
                model_input_params['z_feat_mask'] = self.template_mask_cache.get_batch(do_init_task_ids)

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict, *_):
        if self.provide_during_tracking:
            do_track_task_ids = []
            for task in data.tasks:
                if task.tracker_do_tracking_context is not None:
                    do_track_task_ids.append(task.id)

            if len(do_track_task_ids) > 0:
                model_input_params.update({'z_0_feat_mask': [], 'z_1_feat_mask': [], 'z_2_feat_mask': []})
                for i, task_id in enumerate(do_track_task_ids):
                    if len(self.memory_masks[task_id]) == 1:
                        model_input_params['z_0_feat_mask'].append(self.memory_masks[task_id][0].to(self.device))
                        model_input_params['z_1_feat_mask'].append(self.memory_masks[task_id][0].to(self.device))
                        model_input_params['z_2_feat_mask'].append(self.memory_masks[task_id][0].to(self.device))
                    elif len(self.memory_masks[task_id]) == 2:
                        model_input_params['z_0_feat_mask'].append(self.memory_masks[task_id][0].to(self.device))
                        model_input_params['z_1_feat_mask'].append(self.memory_masks[task_id][1].to(self.device))
                        model_input_params['z_2_feat_mask'].append(self.memory_masks[task_id][1].to(self.device))
                    elif len(self.memory_masks[task_id]) == 3:
                        model_input_params['z_0_feat_mask'].append(self.memory_masks[task_id][0].to(self.device))
                        model_input_params['z_1_feat_mask'].append(self.memory_masks[task_id][1].to(self.device))
                        model_input_params['z_2_feat_mask'].append(self.memory_masks[task_id][2].to(self.device))
                    else:
                        track_context = data.tasks[i].tracker_do_tracking_context
                        seleted_indexes = self.select_memory_frames(track_context.frame_index)
                        for i_selected, idx_selected in enumerate(seleted_indexes):
                            z_i_feat_mask = self.memory_masks[task_id][idx_selected].to(self.device)
                            model_input_params['z_{}_feat_mask'.format(i_selected)].append(z_i_feat_mask)

                model_input_params['z_0_feat_mask'] = torch.stack(model_input_params['z_0_feat_mask'], dim=0)
                model_input_params['z_1_feat_mask'] = torch.stack(model_input_params['z_1_feat_mask'], dim=0)
                model_input_params['z_2_feat_mask'] = torch.stack(model_input_params['z_2_feat_mask'], dim=0)

    @functools.lru_cache()
    def select_memory_frames(self, cur_frame_idx):
        num_segments = 2
        assert cur_frame_idx > num_segments
        dur = cur_frame_idx // num_segments
        indexes = np.concatenate([
            np.array([0]),
            np.array(list(range(num_segments))) * dur + dur // 2
        ])
        indexes = np.unique(indexes)
        return list(indexes)

    def on_tracked(self, data: TrackerEvalData, model_outputs: Any, context: TrackerEvaluationPipeline_Context, *_):
        if 'memory_new_z_curated' in context.temporary_objects:
            for task_id in context.temporary_objects['memory_new_z_curated'].keys():
                bbox = context.temporary_objects['memory_new_z_curated_bbox'][task_id]
                curation_parameter = context.temporary_objects['memory_new_z_curation_parameter'][task_id]
                template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value, dtype=torch.long)
                template_cropped_bbox = get_foreground_bounding_box(bbox, curation_parameter, self.stride)
                bbox_clip_to_image_boundary_(template_cropped_bbox, np.array(self.template_feat_size))
                assert bbox_is_valid(template_cropped_bbox)
                template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
                template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = self.foreground_value
                self.memory_masks[task_id].append(template_mask.cpu())

        for task in data.tasks:
            if task.do_task_finalization:
                self.template_mask_cache.delete(task.id)
                del self.memory_masks[task.id]
