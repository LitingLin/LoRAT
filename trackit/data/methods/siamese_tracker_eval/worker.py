import concurrent.futures
from typing import Sequence, Optional
import time
import torch

from trackit.core.runtime.metric_logger import get_current_local_metric_logger
from trackit.data import MainProcessDataPipeline
from trackit.data.sampling.per_frame.distributed_dynamic_eval_task_scheduling.distributed_evaluation_task_scheduler import DistributedTrackerEvaluationTaskDynamicSchedulerClient, EvaluationTask
from trackit.miscellanies.torch.distributed import get_rank
from trackit.data.source import TrackingDataset, TrackingDataset_FrameInTrack
from trackit.data.protocol.eval_input import TrackerEvalData, SequenceInfo
from trackit.data.utils.frame_decode import get_frame_decoder
from trackit.data.context.worker import get_current_worker_info

from . import SiameseTrackerEvalDataWorker_FrameContext, SiameseTrackerEvalDataWorker_Task
from .transform import SiameseTrackerEval_DataTransform


def _prepare_frame_context(frame: TrackingDataset_FrameInTrack):
    gt_bbox = frame.get_bounding_box() if frame.get_existence_flag() else None
    image_getter_fn = get_frame_decoder(frame)
    return SiameseTrackerEvalDataWorker_FrameContext(frame.get_frame_index(), image_getter_fn, gt_bbox, None)


def _prepare(batch_index: int, evaluation_task: EvaluationTask, datasets: Sequence[TrackingDataset]):
    dataset = datasets[evaluation_task.dataset_index]
    sequence = dataset[evaluation_task.sequence_index]
    track = sequence.get_track_by_id(evaluation_task.track_id)
    track_context = SequenceInfo(dataset.get_name(), dataset.get_data_split(), dataset.get_full_name(), track.get_name(), len(track), sequence.get_fps()) \
        if evaluation_task.do_task_creation else None
    init_frame_context = _prepare_frame_context(track[evaluation_task.do_init_frame_index]) \
        if evaluation_task.do_init_frame_index is not None else None
    if init_frame_context is not None:
        assert init_frame_context.gt_bbox is not None, 'bbox must be provided for init frame. ' + f"dataset: {dataset.get_name()}, sequence: {sequence.get_name()}, track: {track.get_name()}, frame: {evaluation_task.do_init_frame_index}"
    track_frame_context = _prepare_frame_context(track[evaluation_task.do_track_frame_index]) \
        if evaluation_task.do_track_frame_index is not None else None
    return batch_index, \
           SiameseTrackerEvalDataWorker_Task(evaluation_task.task_index,
                                             track_context,
                                             init_frame_context, track_frame_context,
                                             evaluation_task.do_task_finalization)


class SiameseTrackEvaluationDataInputWorker(torch.utils.data.dataset.Dataset):
    def __init__(self, datasets: Sequence[TrackingDataset],
                 dynamic_task_scheduler: DistributedTrackerEvaluationTaskDynamicSchedulerClient,
                 processor: SiameseTrackerEval_DataTransform,
                 num_io_threads: int):
        self._datasets = datasets
        self._dynamic_task_scheduler = dynamic_task_scheduler
        self._rank = get_rank()
        self._background_io_threads: Optional[concurrent.futures.ThreadPoolExecutor] = None
        assert num_io_threads >= 0
        self._num_io_threads = num_io_threads
        self._processor = processor

    def worker_init(self):
        if self._num_io_threads > 0:
            self._background_io_threads = concurrent.futures.ThreadPoolExecutor(self._num_io_threads)

    def worker_shutdown(self):
        if self._num_io_threads > 0:
            self._background_io_threads.shutdown()
        self._background_io_threads = None

    def __getitem__(self, iteration: int):
        if self._background_io_threads is None:
            self.worker_init()
        miscellanies = {'local_worker_index': get_current_worker_info().worker_id}
        evaluation_tasks = self._dynamic_task_scheduler.get_next_batch(self._rank, iteration)
        if evaluation_tasks is None:
            return TrackerEvalData((), miscellanies)

        if self._num_io_threads > 0:
            io_wait_time = 0
            begin_time = time.perf_counter()
            jobs = tuple(self._background_io_threads.submit(_prepare, batch_index, evaluation_task, self._datasets)
                         for batch_index, evaluation_task in enumerate(evaluation_tasks))

            batch = {}
            while len(jobs) > 0:
                io_begin_time = time.perf_counter()
                done_jobs, unfinished_jobs = concurrent.futures.wait(jobs, return_when=concurrent.futures.FIRST_COMPLETED)
                io_wait_time += time.perf_counter() - io_begin_time
                for done_job in done_jobs:
                    batch_index, evaluation_task_step = done_job.result()
                    batch[batch_index] = self._processor(evaluation_task_step)
                jobs = unfinished_jobs
            batch = tuple(batch[index] for index in sorted(batch.keys()))
            total_time = time.perf_counter() - begin_time
            miscellanies['io_wait'] = io_wait_time / total_time
        else:
            batch = tuple(self._processor(_prepare(batch_index, evaluation_task, self._datasets)[1])
                          for batch_index, evaluation_task in enumerate(evaluation_tasks))

        return TrackerEvalData(batch, miscellanies)


class SiameseTrackEvaluationMainProcessLoggingHook(MainProcessDataPipeline):
    def __init__(self, num_io_threads: int):
        self._num_io_threads = num_io_threads

    def on_epoch_begin(self):
        if self._num_io_threads > 0:
            get_current_local_metric_logger().set_metric_format('io_wait', no_prefix=True, format='{median:.2f}')

    def pre_process(self, input_data: Optional[TrackerEvalData]) -> Optional[TrackerEvalData]:
        if input_data is None:
            return None
        if 'io_wait' in input_data.miscellanies:
            get_current_local_metric_logger().log({'io_wait': input_data.miscellanies['io_wait']})
        return input_data
