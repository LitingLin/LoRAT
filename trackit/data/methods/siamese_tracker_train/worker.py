import torch.utils.data
import numpy as np
import time
import concurrent.futures
from typing import Sequence, Optional
from trackit.core.runtime.metric_logger import get_current_local_metric_logger
from trackit.miscellanies.torch.metric_logger import DefaultProgressTracker
from trackit.data.source import TrackingDataset
from trackit.data.context.worker import get_current_worker_info
from trackit.core.operator.numpy.bbox.format import bbox_cxcywh_to_xyxy
from trackit.data.utils.frame_decode import get_frame_decoder
from trackit.data.protocol.train_input import TrainData
from .siamese_training_pair_sampling import SamplingResult_Element, SiamFCTrainingPairSampler
from ... import MainProcessDataPipeline
from ._types import SiameseTrainingPair, SOTFrameInfo
from .transform import SiameseTrackerTrain_DataTransform


def _decode(to_decode: SamplingResult_Element, datasets: Sequence[TrackingDataset], rng_engine: np.random.Generator, prefetch: bool):
    dataset = datasets[to_decode.dataset_index]
    sequence = dataset[to_decode.sequence_index]
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
    return SOTFrameInfo(image_getter, object_bbox, object_exists, dataset, sequence, track, frame)


def _decode_with_cache(name: str, to_decode: SamplingResult_Element, datasets: Sequence[TrackingDataset],
                       cache: dict, result: dict, rng_engine: np.random.Generator, prefetch: bool):
    if to_decode not in cache:
        cache[to_decode] = _decode(to_decode, datasets, rng_engine, prefetch)
    result[name] = cache[to_decode]


def _prepare_siamese_training_pair(sampler_index: Optional[int],
                                   datasets: Sequence[TrackingDataset],
                                   siamese_training_pair_sampler: SiamFCTrainingPairSampler,
                                   rng_engine: np.random.Generator, prefetch: bool,
                                   global_job_index: Optional[int] = None, batch_element_index: Optional[int] = None):
    training_pair = siamese_training_pair_sampler(sampler_index, rng_engine)

    result = {}
    cache = {}
    _decode_with_cache('z', training_pair.z, datasets, cache, result, rng_engine, prefetch)
    _decode_with_cache('x', training_pair.x, datasets, cache, result, rng_engine, prefetch)
    decoded_training_pair = SiameseTrainingPair(training_pair.is_positive,
                                                result['z'], result['x'])

    # bug check
    if decoded_training_pair.is_positive:
        assert decoded_training_pair.template.object_exists
        assert decoded_training_pair.search.object_exists

    if global_job_index is not None:
        return global_job_index, batch_element_index, decoded_training_pair
    else:
        return decoded_training_pair


class SiameseTrackerTrainingDataWorker(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence[TrackingDataset],
                 num_samples_per_epoch: int, batch_size: int,
                 siamese_training_pair_generator: SiamFCTrainingPairSampler,
                 data_transform: SiameseTrackerTrain_DataTransform,
                 num_io_threads: int):
        self.datasets = datasets
        self.num_samples_per_epoch = num_samples_per_epoch
        self.batch_size = batch_size
        self.siamese_training_pair_generator = siamese_training_pair_generator
        self.num_io_threads = num_io_threads
        self.background_io_threads: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.rng = None
        self.transform = data_transform

    def worker_init(self):
        rng_seed = get_current_worker_info().rng_seed
        if self.num_io_threads > 0:
            self.background_io_threads = concurrent.futures.ThreadPoolExecutor(self.num_io_threads)
            seeds = rng_seed.spawn(self.batch_size)
            self.batch_element_rng = tuple(np.random.default_rng(seed) for seed in seeds)
        self.rng = np.random.default_rng(rng_seed.spawn(1)[0].generate_state(1).item())

    def worker_shutdown(self):
        if self.num_io_threads > 0:
            self.background_io_threads.shutdown()
            self.batch_element_rng = None
        self.background_io_threads = None
        self.rng = None

    def __getitems__(self, job_indices: Sequence[int]):
        if self.rng is None:
            self.worker_init()

        if self.num_io_threads > 0:
            io_wait_time = 0
            begin_time = time.perf_counter()
            jobs = tuple(self.background_io_threads.submit(_prepare_siamese_training_pair,
                                                        job_index,
                                                           self.datasets, self.siamese_training_pair_generator,
                                                           self.batch_element_rng[batch_element_index], True,
                                                           job_index, batch_element_index)
                         for batch_element_index, job_index in enumerate(job_indices))
            batch = {}
            while len(jobs) > 0:
                io_begin_time = time.perf_counter()
                done_jobs, unfinished_jobs = concurrent.futures.wait(jobs, return_when=concurrent.futures.FIRST_COMPLETED)
                io_wait_time += time.perf_counter() - io_begin_time
                for job_future in done_jobs:
                    job_index, batch_element_index, siamese_training_pair = job_future.result()
                    data = self.transform(siamese_training_pair, self.rng)
                    if data is None:
                        job = self.background_io_threads.submit(_prepare_siamese_training_pair,
                                                                None,
                                                                self.datasets, self.siamese_training_pair_generator,
                                                                self.batch_element_rng[batch_element_index], True,
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
                siamese_training_pair = _prepare_siamese_training_pair(job_index,
                                                                       self.datasets, self.siamese_training_pair_generator,
                                                                       self.rng, False)
                data = self.transform(siamese_training_pair, self.rng)
                while data is None:
                    siamese_training_pair = _prepare_siamese_training_pair(None,
                                                                           self.datasets, self.siamese_training_pair_generator,
                                                                           self.rng, False)
                    data = self.transform(siamese_training_pair, self.rng)
                batch.append(data)
            return batch, None

    def __getitem__(self, job_index: int):
        if self.rng is None:
            self.worker_init()
        siamese_training_pair = _prepare_siamese_training_pair(job_index,
                                                               self.datasets, self.siamese_training_pair_generator,
                                                               self.rng, False)
        data = self.transform(siamese_training_pair, self.rng)
        while data is None:
            siamese_training_pair = _prepare_siamese_training_pair(None,
                                                                   self.datasets, self.siamese_training_pair_generator,
                                                                   self.rng, False)
            data = self.transform(siamese_training_pair, self.rng)
        return data


    def __len__(self):
        return self.num_samples_per_epoch


class SiameseTrackerTrainingDataCollator:
    def __init__(self, transform_data_collator):
        self.transform_data_collator = transform_data_collator

    def __call__(self, data):
        if isinstance(data, tuple):
            batch, io_wait = data
        else:
            batch = data
            io_wait = None
        collated = TrainData()
        self.transform_data_collator(batch, collated)
        if io_wait is not None:
            collated.miscellanies['io_wait'] = io_wait
        return collated


class ProgressTracker(DefaultProgressTracker):
    def __init__(self, total: int, batch_size: int):
        super().__init__(total)
        self.batch_size = batch_size

    def rate(self):
        if self._time_ema is None:
            return float('nan')
        return (1. / self._time_ema) * self.batch_size


class SiameseTrackerTrainingMainProcessLoggingHook(MainProcessDataPipeline):
    def __init__(self, total_iterations: int, batch_size: int, num_io_threads: int):
        self._total_iterations = total_iterations
        self._batch_size = batch_size
        self._num_io_threads = num_io_threads

    def on_epoch_begin(self):
        if self._num_io_threads > 0:
            get_current_local_metric_logger().set_metric_format('io_wait', no_prefix=True, format='{median:.2f}')
        get_current_local_metric_logger().set_custom_progress_tracker(ProgressTracker(self._total_iterations, self._batch_size))

    def pre_process(self, input_data: TrainData) -> TrainData:
        if 'io_wait' in input_data.miscellanies:
            get_current_local_metric_logger().log({'io_wait': input_data.miscellanies['io_wait']})
        return input_data
