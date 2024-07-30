from __future__ import annotations
from typing import Optional, Iterable, Sequence
from dataclasses import dataclass

from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.core.runtime.metric_logger import get_current_local_metric_logger
from trackit.miscellanies.torch.metric_logger import ProgressTrackerInterface
from trackit.miscellanies.ema import EMA
from trackit.core.runtime.context.task import get_current_task_context
from trackit.core.runtime.services.batch_collective_communication import BatchCollectiveCommunicationFrequency


class ProgressTracker(ProgressTrackerInterface):
    def __init__(self, total_number_of_tracking_tasks: Optional[int], total_number_of_frames: Optional[int]):
        self._total_number_of_tracking_tasks = total_number_of_tracking_tasks
        self._total_number_of_frames = total_number_of_frames
        self.reset()

    def reset(self) -> None:
        self._evaluated_tracks = 0
        self._last_evaluated_frames = 0
        self._current_evaluated_frames = 0
        self._elapsed_time = 0
        self._all_done = False
        self._fps = float('nan')
        if self._total_number_of_frames is not None:
            self._fps_ema = EMA()

    def set_progress(self, evaluated_tracks: int, evaluated_frames: int, all_done: bool) -> None:
        self._evaluated_tracks = evaluated_tracks
        self._current_evaluated_frames = evaluated_frames
        self._all_done = all_done

    def update(self, elapsed_time: float) -> None:
        self._elapsed_time += elapsed_time

    def is_last(self) -> bool:
        return self._all_done

    def pos(self) -> int:
        return self._evaluated_tracks

    def total(self) -> Optional[int]:
        return self._total_number_of_tracking_tasks

    def eta(self) -> Optional[float]:
        if self._total_number_of_frames is None:
            return None
        self._update_estimated_fps()
        fps_ema = self._fps_ema()
        if fps_ema == 0:
            return None
        return (self._total_number_of_frames - self._current_evaluated_frames) / fps_ema

    def _update_estimated_fps(self):
        if self._current_evaluated_frames - self._last_evaluated_frames > 0 and self._elapsed_time > 0:
            self._fps = (self._current_evaluated_frames - self._last_evaluated_frames) / self._elapsed_time
            if self._total_number_of_frames is not None:
                self._fps_ema(self._fps)
            self._last_evaluated_frames = self._current_evaluated_frames
            self._elapsed_time = 0

    def rate(self):
        self._update_estimated_fps()
        return self._fps


class DistributedTrackerEvaluationProgressOrchestrationAndMonitoring:
    @dataclass()
    class State:
        finished: bool
        scheduled_tracks: int
        evaluated_frames: int
        evaluated_tracks: int

    def __init__(self, data_loader: Iterable[TrackerEvalData], num_workers: int,
                 total_number_of_tracking_tasks: Optional[int], total_number_of_frames: Optional[int]):
        self._data_loader = data_loader
        if num_workers == 0:
            num_workers = 1
        self._num_workers = num_workers
        self._total_number_of_tracking_tasks = total_number_of_tracking_tasks
        self._total_number_of_frames = total_number_of_frames

    def on_epoch_begin(self):
        self._progress_tracker = ProgressTracker(self._total_number_of_tracking_tasks, self._total_number_of_frames)
        local_metric_logger = get_current_local_metric_logger()
        if local_metric_logger is not None:
            local_metric_logger.set_metric_format('batch_size', window_size=1, format='{value}', no_prefix=True)
            local_metric_logger.set_metric_format('scheduled', window_size=1, format='{value}', no_prefix=True)
            local_metric_logger.set_custom_progress_tracker(self._progress_tracker)

    def on_epoch_end(self):
        self._progress_tracker = None
        local_metric_logger = get_current_local_metric_logger()
        if local_metric_logger is not None:
            local_metric_logger.set_custom_progress_tracker(None)

    def __iter__(self):
        self._iter = iter(self._data_loader)
        self._worker_exhausted_flags = [False] * self._num_workers
        self._worker_evaluating_tracks = [0] * self._num_workers
        self._worker_evaluated_frames = [0] * self._num_workers
        self._worker_evaluated_tracks = [0] * self._num_workers
        self._world_done = False
        self._local_done = False

        return self

    def __next__(self):
        local_metric_logger = get_current_local_metric_logger()
        if self._world_done:
            raise StopIteration
        if self._local_done:
            if local_metric_logger is not None:
                local_metric_logger.log({'batch_size': 0})
            return None
        data = next(self._iter)
        local_worker_index = data.miscellanies['local_worker_index']
        self._worker_exhausted_flags[local_worker_index] = len(data.tasks) == 0

        if local_metric_logger is not None:
            local_metric_logger.log({'batch_size': len(data.tasks)})

        if all(self._worker_exhausted_flags):
            self._local_done = True
            get_current_task_context().collective_communication.set_mode(BatchCollectiveCommunicationFrequency.high)
            del self._iter
            return None

        for task in data.tasks:
            if task.task_creation_context:
                self._worker_evaluating_tracks[local_worker_index] += 1
            if task.do_task_finalization:
                self._worker_evaluated_tracks[local_worker_index] += 1
            if task.tracker_do_init_context is not None:
                self._worker_evaluated_frames[local_worker_index] += 1
            if task.tracker_do_tracking_context is not None:
                self._worker_evaluated_frames[local_worker_index] += 1
        return data

    def all_gather_begin(self):
        current_state = DistributedTrackerEvaluationProgressOrchestrationAndMonitoring.State(
            self._local_done, sum(self._worker_evaluating_tracks),
            sum(self._worker_evaluated_frames), sum(self._worker_evaluated_tracks))
        return current_state

    def all_gather_end(self,
                       all_rank_states: Sequence[DistributedTrackerEvaluationProgressOrchestrationAndMonitoring.State]):
        self._world_done = all(state.finished for state in all_rank_states)

        global_scheduled_tracks = sum(state.scheduled_tracks for state in all_rank_states)
        global_evaluated_frames = sum(state.evaluated_frames for state in all_rank_states)
        global_evaluated_tracks = sum(state.evaluated_tracks for state in all_rank_states)
        self._progress_tracker.set_progress(global_evaluated_tracks, global_evaluated_frames, self._world_done)

        local_metric_logger = get_current_local_metric_logger()
        if local_metric_logger is not None:
            local_metric_logger.log({'scheduled': global_scheduled_tracks})
