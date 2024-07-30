import numpy as np
from typing import Sequence, Optional, Tuple
import enum
from dataclasses import dataclass

from trackit.data.source import TrackingDataset
from trackit.miscellanies.simple_api_gateway import ServerLauncher, Response, Client
from ._dynamic_task_scheduler import DynamicTaskScheduler


class SequenceOrder(enum.Enum):
    sequential = enum.auto()
    random = enum.auto()


@dataclass()
class EvaluationTask:
    task_index: int
    dataset_index: int
    sequence_index: int
    track_id: int
    do_task_creation: bool
    do_init_frame_index: Optional[int]
    do_track_frame_index: Optional[int]
    do_task_finalization: bool


class DistributedTrackerEvaluationTaskDynamicScheduler:
    def __init__(self, datasets: Sequence[TrackingDataset], repeat_times: Sequence[int], world_size: int,
                 sequence_order: SequenceOrder=SequenceOrder.sequential, rng_seed: Optional[int]=None):
        assert len(datasets) == len(repeat_times)

        dataset_indices = []
        sequence_indices = []
        track_id_list = []
        track_length_list = []

        number_of_frames = 0
        number_of_tasks = 0

        for index_of_dataset, (dataset, repeat_time) in enumerate(zip(datasets, repeat_times)):
            for index_of_repeat in range(repeat_time):
                for index_of_sequence, sequence in enumerate(dataset):
                    sub_dataset_indices = np.full(sequence.get_number_of_tracks(), index_of_dataset, dtype=int)
                    sub_sequence_indices = np.full(sequence.get_number_of_tracks(), index_of_sequence, dtype=int)
                    sub_track_id_list = np.empty(sequence.get_number_of_tracks(), dtype=int)
                    sub_track_length_list = np.empty(sequence.get_number_of_tracks(), dtype=int)
                    for i, index_of_track in enumerate(range(sequence.get_number_of_tracks())):
                        track = sequence.get_track_by_index(index_of_track)
                        sub_track_id_list[i] = track.get_object_id()
                        sub_track_length_list[i] = len(track)
                        number_of_tasks += 1
                        number_of_frames += len(track)

                    dataset_indices.append(sub_dataset_indices)
                    sequence_indices.append(sub_sequence_indices)
                    track_id_list.append(sub_track_id_list)
                    track_length_list.append(sub_track_length_list)

        dataset_indices = np.concatenate(dataset_indices, axis=0)
        sequence_indices = np.concatenate(sequence_indices, axis=0)
        track_id_list = np.concatenate(track_id_list, axis=0)
        track_length_list = np.concatenate(track_length_list, axis=0)

        if sequence_order == SequenceOrder.random:
            rng_engine = np.random.default_rng(rng_seed)
            permutation_indices = rng_engine.permutation(len(dataset_indices))

            dataset_indices = dataset_indices[permutation_indices]
            sequence_indices = sequence_indices[permutation_indices]
            track_id_list = track_id_list[permutation_indices]
            track_length_list = track_length_list[permutation_indices]

        self.dataset_indices = dataset_indices
        self.sequence_indices = sequence_indices
        self.track_id_list = track_id_list
        self.track_length_list = track_length_list
        self._number_of_tasks = number_of_tasks
        self._number_of_frames = number_of_frames
        self._scheduler = DynamicTaskScheduler(track_length_list - 1, world_size)

    def get_number_of_frames(self):
        return self._number_of_frames

    def get_number_of_tasks(self):
        return self._number_of_tasks

    def reset(self):
        self._scheduler.reset()

    def get_next_batch(self, rank_id: int, rank_iteration: int, batch_size: int) -> Optional[Sequence[EvaluationTask]]:
        batch_indices = self._scheduler.get_next_batch(rank_id, rank_iteration, batch_size)

        if batch_indices is None:
            return None

        evaluation_tasks = []
        for index, step_index in batch_indices:
            dataset_index = self.dataset_indices[index]
            sequence_index = self.sequence_indices[index]
            track_id = self.track_id_list[index]
            track_length = self.track_length_list[index]

            track_frame_index = step_index + 1

            is_last_frame = track_frame_index + 1 == track_length
            if track_frame_index == 1:
                do_task_init = True
                do_init_frame_index = 0
                do_track_frame_index = 1
            else:
                do_task_init = False
                do_init_frame_index = None
                do_track_frame_index = track_frame_index
            evaluation_tasks.append(EvaluationTask(index,
                                                   dataset_index, sequence_index, track_id,
                                                   do_task_init,
                                                   do_init_frame_index, do_track_frame_index,
                                                   is_last_frame))

        return tuple(evaluation_tasks)


class _DistributedTrackerEvaluationTaskDynamicScheduler_APIGatewayCallback:
    def __init__(self, scheduler: DistributedTrackerEvaluationTaskDynamicScheduler, batch_size: int, initialize: bool):
        self._scheduler = scheduler
        self._batch_size = batch_size
        if initialize:
            self._scheduler.reset()

    def __call__(self, command: Tuple, response: Response):
        if command[0] == 'get_next_batch':
            response.set_body(self._scheduler.get_next_batch(command[1], command[2], self._batch_size))
        elif command[0] == 'reset':
            self._scheduler.reset()
        else:
            raise NotImplementedError(command)


class DistributedTrackerEvaluationTaskDynamicSchedulerServer(ServerLauncher):
    def __init__(self, socket_address: str, key: str,
                 scheduler: DistributedTrackerEvaluationTaskDynamicScheduler, batch_size: int, initialize: bool):
        super(DistributedTrackerEvaluationTaskDynamicSchedulerServer, self).__init__(
            socket_address, key,
            callback=_DistributedTrackerEvaluationTaskDynamicScheduler_APIGatewayCallback(scheduler, batch_size, initialize))


class DistributedTrackerEvaluationTaskDynamicSchedulerClient:
    def __init__(self, socket_address: str, key: str):
        self._client = Client(socket_address, key)

    def get_next_batch(self, rank_id: int, rank_iteration: int) -> Optional[Tuple[EvaluationTask, ...]]:
        return self._client('get_next_batch', rank_id, rank_iteration)

    def reset(self):
        return self._client('reset')

    def close(self):
        self._client.stop()
