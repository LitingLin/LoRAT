import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT, SequenceInfo, FrameEvaluationResult_SOT
from trackit.miscellanies.numpy_array_builder import NumpyArrayBuilder

from .types import TrackerEvaluationPipeline_Context, TrackingPipeline_ResultHolder
from .pipelines.interface import TrackingPipeline

from .. import TrackerEvaluator, EvaluatorContext


class DefaultTrackerEvaluator(TrackerEvaluator):
    def __init__(self, pipeline: TrackingPipeline,
                 time_cost_reduce_with_batch_size: bool = True):
        self.pipeline = pipeline
        self.time_cost_reduce_with_batch_size = time_cost_reduce_with_batch_size

    def start(self, context: EvaluatorContext):
        self.all_tracking_sequence_context: Dict[Any, _Context] = {}
        self.global_shared_objects = {}
        self.current_tracking_context: Optional[TrackerEvaluationPipeline_Context] = None
        self.pipeline.start(context, self.global_shared_objects)

    def stop(self, context: EvaluatorContext):
        assert self.current_tracking_context is None
        self.pipeline.stop(context, self.global_shared_objects)
        del self.global_shared_objects
        assert len(self.all_tracking_sequence_context) == 0
        del self.all_tracking_sequence_context

    def run(self, data: Optional[TrackerEvalData],
                 optimized_model: Any, raw_model: Optional[nn.Module]) -> Optional[Mapping[str, Any]]:
        if data is None:
            return None

        # begin
        for tracking_task in data.tasks:
            if tracking_task.task_creation_context is not None:
                self.all_tracking_sequence_context[tracking_task.id] = \
                    _Context(tracking_task.id, tracking_task.task_creation_context)

        all_related_tracks = {tracking_task.id: self.all_tracking_sequence_context[tracking_task.id].finalize() for tracking_task in data.tasks}
        current_tracking_context = TrackerEvaluationPipeline_Context(all_related_tracks, global_objects=self.global_shared_objects)

        # init
        init_begin_time = time.perf_counter()
        num_init_tracks = 0
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                do_init_context = task.tracker_do_init_context
                track_context = self.all_tracking_sequence_context[task.id]
                track_context.frame_indices.append(do_init_context.frame_index)
                if do_init_context.gt_bbox is not None:
                    track_context.groundtruth_target_existence.append(True)
                    track_context.groundtruth_target_bounding_boxes.append(do_init_context.gt_bbox)
                if do_init_context.gt_mask is not None:
                    track_context.groundtruth_target_foreground_masks.append(do_init_context.gt_mask)
                track_context.prediction_begin_time.append(init_begin_time)
                num_init_tracks += 1

        self.pipeline.initialize(data, optimized_model, current_tracking_context)
        # on initialized
        init_end_time = time.perf_counter()
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                track_context = self.all_tracking_sequence_context[task.id]
                track_context.prediction_end_time.append(init_end_time)
                track_context.batch_size.append(num_init_tracks)

        if num_init_tracks > 0:
            # reinitialize because of tracking context updated
            all_related_tracks = {tracking_task.id: self.all_tracking_sequence_context[tracking_task.id].finalize() for
                                  tracking_task in data.tasks}
            current_tracking_context = TrackerEvaluationPipeline_Context(all_related_tracks, self.global_shared_objects)

        tracking_begin_time = time.perf_counter()
        # prepare tracking
        tracking_id_list = []
        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                do_track_context = task.tracker_do_tracking_context
                this_track_context = self.all_tracking_sequence_context[task.id]
                this_track_context.frame_indices.append(do_track_context.frame_index)
                if do_track_context.gt_bbox is not None:
                    this_track_context.groundtruth_target_existence.append(True)
                    this_track_context.groundtruth_target_bounding_boxes.append(do_track_context.gt_bbox)
                else:
                    if len(this_track_context.groundtruth_target_existence) > 0:
                        this_track_context.groundtruth_target_existence.append(False)
                        this_track_context.groundtruth_target_bounding_boxes.append(np.full(4, float('nan'), dtype=np.float64))
                if len(this_track_context.groundtruth_target_foreground_masks) > 0:
                    this_track_context.groundtruth_target_foreground_masks.append(do_track_context.gt_mask)
                this_track_context.prediction_begin_time.append(tracking_begin_time)
                tracking_id_list.append(task.id)

        tracking_results = TrackingPipeline_ResultHolder(tracking_id_list)

        self.pipeline.track(data, optimized_model, current_tracking_context, tracking_results)
        tracking_end_time = time.perf_counter()

        evaluated_frames = []

        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                track_context = self.all_tracking_sequence_context[task.id]
                tracking_result = tracking_results.get(task.id)
                tracker_has_box_prediction = tracking_result.box is not None
                tracker_has_confidence_prediction = tracking_result.confidence is not None
                tracker_has_mask_prediction = tracking_result.mask is not None

                # assume that initialization frame is not tracked
                if tracker_has_box_prediction and task.tracker_do_init_context is not None:
                    track_context.predicted_bounding_boxes.append(task.tracker_do_init_context.gt_bbox)
                if tracker_has_confidence_prediction and task.tracker_do_init_context is not None:
                    track_context.predicted_confidences.append(1.)
                if tracker_has_mask_prediction and task.tracker_do_init_context is not None:
                    track_context.predicted_masks.append(task.tracker_do_init_context.gt_mask)
                if tracker_has_box_prediction:
                    track_context.predicted_bounding_boxes.append(tracking_result.box)
                if tracker_has_confidence_prediction:
                    track_context.predicted_confidences.append(tracking_result.confidence)
                if tracker_has_mask_prediction:
                    track_context.predicted_masks.append(tracking_result.mask)
                track_context.prediction_end_time.append(tracking_end_time)
                track_context.batch_size.append(len(tracking_id_list))

                evaluated_frames.append(track_context.get_last(self.time_cost_reduce_with_batch_size))

        # finalization
        evaluated_sequences = []
        for task in data.tasks:
            if task.do_task_finalization:
                track_context = self.all_tracking_sequence_context[task.id]
                evaluated_sequences.append(track_context.finalize(self.time_cost_reduce_with_batch_size))
                self.all_tracking_sequence_context.pop(task.id)

        return {'evaluated_sequences': evaluated_sequences, 'evaluated_frames': evaluated_frames}


@dataclass(frozen=True)
class _Context:
    task_id: Any
    sequence_info: SequenceInfo
    frame_indices: NumpyArrayBuilder = field(init=False)
    groundtruth_target_existence: NumpyArrayBuilder = field(init=False)
    groundtruth_target_bounding_boxes: NumpyArrayBuilder = field(init=False)
    groundtruth_target_foreground_masks: List[Image.Image] = field(default_factory=list)
    predicted_confidences: Optional[NumpyArrayBuilder] = field(init=False)
    predicted_bounding_boxes: NumpyArrayBuilder = field(init=False)
    predicted_masks: List[Image.Image] = field(default_factory=list)
    prediction_begin_time: NumpyArrayBuilder = field(init=False)
    prediction_end_time: NumpyArrayBuilder = field(init=False)
    batch_size: NumpyArrayBuilder = field(init=False)

    def __post_init__(self):
        initial_size = self.sequence_info.length if self.sequence_info.length is not None else 32
        object.__setattr__(self, "frame_indices", NumpyArrayBuilder(np.uint32, initial_capacity=initial_size))
        object.__setattr__(self, "groundtruth_target_existence", NumpyArrayBuilder(np.bool_, initial_capacity=initial_size))
        object.__setattr__(self, "groundtruth_target_bounding_boxes", NumpyArrayBuilder(np.float64, initial_capacity=initial_size, extra_dims=(4,)))
        object.__setattr__(self, "predicted_confidences", NumpyArrayBuilder(np.float64, initial_capacity=initial_size))
        object.__setattr__(self, "predicted_bounding_boxes", NumpyArrayBuilder(np.float64, initial_capacity=initial_size, extra_dims=(4,)))
        object.__setattr__(self, "prediction_begin_time", NumpyArrayBuilder(np.float64, initial_capacity=initial_size))
        object.__setattr__(self, "prediction_end_time", NumpyArrayBuilder(np.float64, initial_capacity=initial_size))
        object.__setattr__(self, "batch_size", NumpyArrayBuilder(np.float64, initial_capacity=initial_size))

    def get_last(self, time_cost_reduce_with_batch_size: bool = True) -> FrameEvaluationResult_SOT:
        time_cost = self.prediction_end_time[-1] - self.prediction_begin_time[-1]
        batch_size = self.batch_size[-1]
        if time_cost_reduce_with_batch_size:
            time_cost /= batch_size
        return FrameEvaluationResult_SOT(
            id=self.task_id,
            frame_index=self.frame_indices[-1].item(),
            sequence_info=self.sequence_info,
            groundtruth_box=self.groundtruth_target_bounding_boxes[-1] if len(self.groundtruth_target_bounding_boxes) > 0 else None,
            groundtruth_object_existence_flag=self.groundtruth_target_existence[-1].item() if len(self.groundtruth_target_existence) > 0 else None,
            groundtruth_mask=self.groundtruth_target_foreground_masks[-1] if len(self.groundtruth_target_foreground_masks) > 0 else None,
            output_box=self.predicted_bounding_boxes[-1] if len(self.predicted_bounding_boxes) > 0 else None,
            output_confidence=self.predicted_confidences[-1].item() if len(self.predicted_confidences) > 0 else None,
            output_mask=self.predicted_masks[-1] if len(self.predicted_masks) > 0 else None,
            time_cost=time_cost.item()
        )

    def finalize(self, time_cost_reduce_with_batch_size: bool = True) -> SequenceEvaluationResult_SOT:
        time_cost = self.prediction_end_time.build() - self.prediction_begin_time.build()
        batch_size = self.batch_size.build()
        if time_cost_reduce_with_batch_size:
            time_cost /= batch_size
        groundtruth_box = self.groundtruth_target_bounding_boxes.build() if len(self.groundtruth_target_bounding_boxes) > 0 else None
        groundtruth_existence_flag = self.groundtruth_target_existence.build() if len(self.groundtruth_target_existence) > 0 else None
        groundtruth_mask = tuple(self.groundtruth_target_foreground_masks) if len(self.groundtruth_target_foreground_masks) > 0 else None
        output_box = self.predicted_bounding_boxes.build() if len(self.predicted_bounding_boxes) > 0 else None
        output_confidence = self.predicted_confidences.build() if len(self.predicted_confidences) > 0 else None
        mask_output = tuple(self.predicted_masks) if len(self.predicted_masks) > 0 else None
        return SequenceEvaluationResult_SOT(
            id=self.task_id,
            sequence_info=self.sequence_info,
            evaluated_frame_indices=self.frame_indices.build(),
            groundtruth_box=groundtruth_box,
            groundtruth_object_existence_flag=groundtruth_existence_flag,
            groundtruth_mask=groundtruth_mask,
            output_box=output_box,
            output_confidence=output_confidence,
            output_mask=mask_output,
            time_cost=time_cost,
            batch_size=batch_size
        )

