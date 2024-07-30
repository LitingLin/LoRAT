from typing import Dict, Tuple, Callable, Any, Optional, List
import numpy as np
import torch
from dataclasses import dataclass, field

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, \
    reverse_siamfc_cropping_params, apply_siamfc_cropping_subpixel, scale_siamfc_cropping_params
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider import CroppingParameterProvider
from ....components.post_process import TrackerOutputPostProcess
from ....components.segmentation import Segmentify_PostProcessor
from ....components.tensor_cache import CacheService, TensorCache

from ... import TrackerEvaluationPipeline


@dataclass
class _LocalContext:
    reset_frame_indices: List[int] = field(default_factory=list)
    siamfc_cropping_params_provider: Optional[CroppingParameterProvider] = None


class OneStreamTracker_Evaluation_MainPipeline(TrackerEvaluationPipeline):
    def __init__(self, device: torch.device,
                 template_image_size: Tuple[int, int],
                 search_region_image_size: Tuple[int, int],  # W, H
                 search_curation_parameter_provider_factory: Callable[[], CroppingParameterProvider],
                 model_output_post_process: TrackerOutputPostProcess,
                 segmentify_post_process: Optional[Segmentify_PostProcessor],
                 interpolation_mode: str, interpolation_align_corners: bool,
                 norm_stats_dataset_name: str, visualization: bool):
        self.template_image_size = template_image_size
        self.search_region_image_size = search_region_image_size

        self.search_image_cropping_params_provider_factory = search_curation_parameter_provider_factory
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.model_output_post_process = model_output_post_process
        self.segmentify_post_process = segmentify_post_process
        self.device = device

        self.image_normalization_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.visualization = visualization

    def start(self, max_batch_size: int, global_shared_objects):
        template_shape = (3, self.template_image_size[1], self.template_image_size[0])
        search_region_shape = (3, self.search_region_image_size[1], self.search_region_image_size[0])

        self.all_tracking_task_local_contexts: Dict[Any, _LocalContext] = {}
        self.all_tracking_template_cache = CacheService(max_batch_size,
                                                        TensorCache(max_batch_size, template_shape, self.device))
        self.all_tracking_template_image_mean_cache = CacheService(max_batch_size,
                                                                   TensorCache(max_batch_size, (3, ), self.device))
        global_shared_objects['template_cache'] = self.all_tracking_template_cache
        global_shared_objects['template_image_mean_cache'] = self.all_tracking_template_image_mean_cache

        self.cropping_parameter_cache = np.full((max_batch_size, 2, 2), float('nan'), dtype=np.float64)
        self.search_region_cache = torch.full((max_batch_size, *search_region_shape), float('nan'),
                                              dtype=torch.float, device=self.device)
        self.model_output_post_process.start()
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.start(max_batch_size)

    def stop(self, global_shared_objects):
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.stop()
        self.model_output_post_process.stop()
        assert len(self.all_tracking_task_local_contexts) == 0, "bug check: some tracking sequences are not finished"
        del self.cropping_parameter_cache
        del self.search_region_cache
        del self.all_tracking_template_cache
        del self.all_tracking_template_image_mean_cache
        del self.all_tracking_task_local_contexts

    def begin(self, context):
        for task in context.input_data.tasks:
            if task.task_creation_context is not None:
                assert task.id not in self.all_tracking_task_local_contexts
                self.all_tracking_task_local_contexts[task.id] = _LocalContext()

    def prepare_initialization(self, context, model_input_params):
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                init_context = task.tracker_do_init_context
                self.all_tracking_template_cache.put(task.id, init_context.input_data['curated_image'])
                self.all_tracking_template_image_mean_cache.put(task.id, init_context.input_data['image_mean'])
                cropping_params_provider = self.search_image_cropping_params_provider_factory()
                cropping_params_provider.initialize(init_context.gt_bbox)
                task_context = self.all_tracking_task_local_contexts[task.id]
                task_context.siamfc_cropping_params_provider = cropping_params_provider
                task_context.reset_frame_indices.append(init_context.frame_index)

    def prepare_tracking(self, context, model_input_params):
        num_tracking_sequence = 0
        task_ids = []
        image_size_list = []
        frame_indices = []
        for task in context.input_data.tasks:
            if task.tracker_do_tracking_context is not None:
                track_context = task.tracker_do_tracking_context
                template_image_mean = self.all_tracking_template_image_mean_cache.get(task.id)
                cropping_params_provider = self.all_tracking_task_local_contexts[task.id].siamfc_cropping_params_provider
                cropping_params = cropping_params_provider.get(np.array(self.search_region_image_size))
                x = track_context.input_data['image'].to(torch.float32)
                H, W = x.shape[-2:]
                image_size_list.append(np.array((W, H), dtype=np.int32))
                _, _, cropping_params = \
                    apply_siamfc_cropping(x, np.array(self.search_region_image_size), cropping_params,
                                          self.interpolation_mode, self.interpolation_align_corners,
                                          template_image_mean,
                                          out_image=self.search_region_cache[num_tracking_sequence, ...])
                self.cropping_parameter_cache[num_tracking_sequence, ...] = cropping_params
                num_tracking_sequence += 1
                task_ids.append(task.id)
                frame_indices.append(track_context.frame_index)

        if num_tracking_sequence == 0:
            return

        context.temporary_objects['task_ids'] = task_ids
        context.temporary_objects['x_frame_sizes'] = image_size_list
        context.temporary_objects['x_frame_indices'] = frame_indices
        context.temporary_objects['x_cropping_params'] = self.cropping_parameter_cache[: num_tracking_sequence, ...]

        z = self.all_tracking_template_cache.get_batch(task_ids)
        x = self.search_region_cache[: num_tracking_sequence, ...]
        x = x / 255.
        self.image_normalization_transform_(x)

        model_input_params.update({'z': z, 'x': x})

    def on_tracked(self, model_outputs, context):
        if model_outputs is None:
            return
        task_ids = context.temporary_objects['task_ids']
        x_frame_sizes = context.temporary_objects['x_frame_sizes']
        x_frame_indices = context.temporary_objects['x_frame_indices']
        x_cropping_params = context.temporary_objects['x_cropping_params']

        outputs = self.model_output_post_process(model_outputs)
        # shape: (num_tracking_sequence), dtype: torch.float
        all_predicted_score = outputs['confidence']
        # shape: (num_tracking_sequence, 4), dtype: torch.float
        all_predicted_bounding_box = outputs['box']
        # shape: (num_tracking_sequence, H, W), dtype: torch.bool, allow None
        all_predicted_mask = outputs.get('mask', None)

        assert all_predicted_score.ndim == 1
        assert all_predicted_bounding_box.ndim == 2
        assert all_predicted_bounding_box.shape[1] == 4
        assert len(task_ids) == len(all_predicted_score) == len(all_predicted_bounding_box)
        if all_predicted_mask is not None:
            assert all_predicted_mask.ndim == 3
            assert all_predicted_mask.shape[0] == len(task_ids)

        all_predicted_score = all_predicted_score.cpu()
        assert torch.all(torch.isfinite(all_predicted_score))
        all_predicted_bounding_box = all_predicted_bounding_box.cpu()
        assert torch.all(torch.isfinite(all_predicted_bounding_box))

        all_predicted_bounding_box = all_predicted_bounding_box.to(torch.float64)

        all_predicted_score = all_predicted_score.numpy()
        all_predicted_bounding_box = all_predicted_bounding_box.numpy()

        all_predicted_bounding_box_on_full_search_image = apply_siamfc_cropping_to_boxes(
            all_predicted_bounding_box, reverse_siamfc_cropping_params(x_cropping_params))
        for predicted_bounding_box_on_full_search_image, image_size in zip(
                all_predicted_bounding_box_on_full_search_image, x_frame_sizes):
            bbox_clip_to_image_boundary_(predicted_bounding_box_on_full_search_image, image_size)

        all_predicted_mask_on_full_search_image = None
        if all_predicted_mask is not None:
            all_predicted_mask_on_full_search_image = []
            for curr_mask, curr_image_size, curr_cropping_parameter in zip(
                    all_predicted_mask, x_frame_sizes, x_cropping_params):
                mask_h, mask_w = curr_mask.shape
                curr_cropping_parameter = scale_siamfc_cropping_params(curr_cropping_parameter,
                                                                       np.array(self.search_region_image_size),
                                                                       np.array((mask_w, mask_h)))
                predicted_mask_on_full_search_image = apply_siamfc_cropping_subpixel(
                    curr_mask.to(torch.float32).unsqueeze(0),
                    np.array(curr_image_size), reverse_siamfc_cropping_params(curr_cropping_parameter),
                    self.interpolation_mode, self.interpolation_align_corners)
                all_predicted_mask_on_full_search_image.append(
                    predicted_mask_on_full_search_image.squeeze(0).to(torch.bool).cpu().numpy())
        else:
            if self.segmentify_post_process is not None:
                full_search_region_images = []

                for task in context.input_data.tasks:
                    if task.tracker_do_tracking_context is not None:
                        full_search_region_images.append(task.tracker_do_tracking_context.input_data['image'])
                all_predicted_mask_on_full_search_image = (
                    self.segmentify_post_process(full_search_region_images,
                                                 all_predicted_bounding_box_on_full_search_image))

        for index, (task_id, image_size, frame_index) in enumerate(zip(task_ids, x_frame_sizes, x_frame_indices)):
            predicted_score = all_predicted_score[index].item()
            predicted_bounding_box_on_full_search_image = all_predicted_bounding_box_on_full_search_image[index]
            local_task_context = self.all_tracking_task_local_contexts[task_id]
            local_task_context.siamfc_cropping_params_provider.update(predicted_score,
                                                                      predicted_bounding_box_on_full_search_image,
                                                                      image_size)
            predicted_mask_on_full_search_image = all_predicted_mask_on_full_search_image[index] \
                if all_predicted_mask_on_full_search_image is not None else None
            context.result.submit(task_id,
                                  predicted_bounding_box_on_full_search_image,
                                  predicted_score,
                                  predicted_mask_on_full_search_image)
            if self.visualization:
                from .visualization import visualize_tracking_result
                sequence_info = context.all_tracks[task_id].sequence_info
                x = self.search_region_cache[index, ...]
                predicted_bounding_box = all_predicted_bounding_box[index]
                predicted_mask = all_predicted_mask[index] if all_predicted_mask is not None else None
                visualize_tracking_result(sequence_info.dataset_name, sequence_info.sequence_name, frame_index,
                                          x, predicted_bounding_box,
                                          predicted_mask, predicted_mask_on_full_search_image)

        assert context.result.is_all_submitted()

    def end(self, context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.all_tracking_template_cache.delete(task.id)
                self.all_tracking_template_image_mean_cache.delete(task.id)
                self.all_tracking_task_local_contexts.pop(task.id)
