import torch.utils.data
import numpy as np
import time
import concurrent.futures
from typing import Sequence, Optional

from trackit.core.runtime.metric_logger import get_current_local_metric_logger
from trackit.data.source import TrackingDataset
from trackit.data.context.worker import get_current_worker_info
from trackit.core.operator.numpy.bbox.format import bbox_cxcywh_to_xyxy
from trackit.data.utils.frame_decode import get_frame_decoder
from trackit.data.protocol.train_input import TrainData
from ... import MainProcessDataPipeline
from ._types import SiameseTrainingMultiPair, SOTFrameInfo
from .transform import SPMTrackTrain_DataTransform
from .training_tuplet_sampling import SamplingResult_Element, SPMTrack_TrainingTupletSampler


def _decode(to_decode: SamplingResult_Element, datasets: Sequence[TrackingDataset], rng_engine: np.random.Generator, prefetch: bool):
    sequence = datasets[to_decode.dataset_index][to_decode.sequence_index]
    track = sequence.get_track_by_id(to_decode.track_id)
    frame = track[to_decode.frame_index]
    image_getter = get_frame_decoder(frame, prefetch)
    object_exists = frame.get_existence_flag()
    if object_exists:
        object_bbox = frame.get_bounding_box().astype(np.float64)
    else:
        object_bbox = rng_engine.random(4, dtype=np.float64)
        object_bbox = bbox_cxcywh_to_xyxy(object_bbox)
        object_bbox *= np.repeat(frame.get_frame_size(), 2)
    return SOTFrameInfo(image_getter, object_bbox, object_exists, sequence, track, frame)


def _decode_with_cache(name: str, to_decode: Sequence[SamplingResult_Element], datasets: Sequence[TrackingDataset],
                       cache: dict, result: dict, rng_engine: np.random.Generator, prefetch: bool):
    for i, element in enumerate(to_decode):
        if element not in cache:
            cache[element] = _decode(element, datasets, rng_engine, prefetch)
        result[f"{name}-{i}"] = cache[element]
    #if to_decode not in cache:
    #    cache[to_decode] = _decode(to_decode, datasets, rng_engine, prefetch)
    #result[name] = cache[to_decode]


def _prepare_training_tuplet(sampler_index: Optional[int],
                             datasets: Sequence[TrackingDataset],
                             SPMTrack_training_tuplet_sampler: SPMTrack_TrainingTupletSampler,
                             rng_engine: np.random.Generator, prefetch: bool,
                             global_job_index: Optional[int] = None, batch_element_index: Optional[int] = None):
    training_tuplet = SPMTrack_training_tuplet_sampler(sampler_index, rng_engine)

    result = {}
    cache = {}
    _decode_with_cache('z', training_tuplet.z, datasets, cache, result, rng_engine, prefetch)
    _decode_with_cache('x', training_tuplet.x, datasets, cache, result, rng_engine, prefetch)
    decoded_training_tuplet = SiameseTrainingMultiPair(training_tuplet.is_positive, [result[key] for key in result.keys() if 'z' in key], [result[key] for key in result.keys() if 'x' in key])
    # bug check
    if decoded_training_tuplet.is_positive:
        for item in decoded_training_tuplet.template:
            assert item.object_exists
        for item in decoded_training_tuplet.search:
            assert item.object_exists

    if global_job_index is not None:
        return global_job_index, batch_element_index, decoded_training_tuplet
    else:
        return decoded_training_tuplet

class SPMTrackTrackerTrainingDataWorker(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence[TrackingDataset],
                 num_samples_per_epoch: int, batch_size: int,
                 SPMTrack_training_tuplet_sampler: SPMTrack_TrainingTupletSampler,
                 data_transform: SPMTrackTrain_DataTransform,
                 num_io_threads: int):
        self.datasets = datasets
        self.num_samples_per_epoch = num_samples_per_epoch
        self.batch_size = batch_size
        self.SPMTrack_training_tuplet_sampler = SPMTrack_training_tuplet_sampler
        self.num_io_threads = num_io_threads
        self.background_io_threads: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.rng_seed = None
        self.transform = data_transform

    def worker_init(self):
        self.rng_seed = get_current_worker_info().rng_seed
        if self.num_io_threads > 0:
            self.background_io_threads = concurrent.futures.ThreadPoolExecutor(self.num_io_threads)

    def worker_shutdown(self):
        if self.num_io_threads > 0:
            self.background_io_threads.shutdown()
        self.background_io_threads = None
        self.rng_seed = None

    def __getitems__(self, job_indices: Sequence[int]):
        if self.rng_seed is None:
            self.worker_init()

        if self.num_io_threads > 0:
            io_wait_time = 0
            begin_time = time.perf_counter()
            batch_element_rng = tuple(np.random.Generator(np.random.PCG64(self.rng_seed + (job_index,))) for job_index in job_indices)
            jobs = tuple(self.background_io_threads.submit(_prepare_training_tuplet,
                                                           job_index,
                                                           self.datasets, self.SPMTrack_training_tuplet_sampler,
                                                           batch_element_rng[batch_element_index], True,
                                                           job_index, batch_element_index)
                         for batch_element_index, job_index in enumerate(job_indices))
            batch = {}
            while len(jobs) > 0:
                io_begin_time = time.perf_counter()
                done_jobs, unfinished_jobs = concurrent.futures.wait(jobs, return_when=concurrent.futures.FIRST_COMPLETED)
                io_wait_time += time.perf_counter() - io_begin_time
                for job_future in done_jobs:
                    job_index, batch_element_index, training_tuplet = job_future.result()
                    data = self.transform(training_tuplet, batch_element_rng[batch_element_index])
                    if data is None:
                        job = self.background_io_threads.submit(_prepare_training_tuplet,
                                                                None,
                                                                self.datasets, self.SPMTrack_training_tuplet_sampler,
                                                                batch_element_rng[batch_element_index], True,
                                                                job_index, batch_element_index)
                        unfinished_jobs = list(unfinished_jobs)
                        unfinished_jobs.append(job)
                    else:
                        batch[job_index] = data

                jobs = unfinished_jobs
            batch = tuple(batch[index] for index in sorted(batch.keys()))
            total_time = time.perf_counter() - begin_time
            io_wait = io_wait_time / total_time
            return batch, io_wait
        else:
            batch = []
            for job_index in job_indices:
                rng = np.random.Generator(np.random.PCG64(self.rng_seed + (job_index,)))
                training_tuplet = _prepare_training_tuplet(job_index,
                                                           self.datasets, self.SPMTrack_training_tuplet_sampler,
                                                           rng, False)
                data = self.transform(training_tuplet, rng)
                while data is None:
                    training_tuplet = _prepare_training_tuplet(None,
                                                               self.datasets, self.SPMTrack_training_tuplet_sampler,
                                                               rng, False)
                    data = self.transform(training_tuplet, rng)
                batch.append(data)
            return batch, None

    def __getitem__(self, job_index: int):
        if self.rng_seed is None:
            self.worker_init()
        rng = np.random.Generator(np.random.PCG64(self.rng_seed + (job_index,)))
        training_tuplet = _prepare_training_tuplet(job_index,
                                                   self.datasets, self.SPMTrack_training_tuplet_sampler,
                                                   rng, False)
        data = self.transform(training_tuplet, rng)
        while data is None:
            training_tuplet = _prepare_training_tuplet(None,
                                                       self.datasets, self.SPMTrack_training_tuplet_sampler,
                                                       rng, False)
            data = self.transform(training_tuplet, rng)
        return data

    def __len__(self):
        return self.num_samples_per_epoch

class SPMTrackTrackerTrainingDataCollator:
    def __init__(self, transform_data_collator):
        self.transform_data_collator = transform_data_collator

    def __call__(self, data):
        batch, io_wait = data
        collated = TrainData(len(batch))
        self.transform_data_collator(batch, collated)
        if io_wait is not None:
            collated.miscellanies['io_wait'] = io_wait
        return collated


class SPMTrackTrackerTrainingHostLoggingHook(MainProcessDataPipeline):
    def __init__(self, num_io_threads: int):
        self._num_io_threads = num_io_threads

    def on_epoch_begin(self):
        if self._num_io_threads > 0:
            get_current_local_metric_logger().set_metric_format('io_wait', no_prefix=True)

    def pre_process(self, input_data: TrainData) -> TrainData:
        if 'io_wait' in input_data.miscellanies:
            get_current_local_metric_logger().log({'io_wait': input_data.miscellanies['io_wait']})
        return input_data
