import time
from dataclasses import dataclass
from typing import Protocol, Any, Sequence
from enum import Enum, auto

from trackit.miscellanies.torch.distributed.collective_communication import CollectiveCommunication
from trackit.miscellanies.torch.distributed import is_dist_initialized, get_rank
from . import BatchCollectiveCommunicationServiceRegistry, BatchCollectiveCommunicationServiceOperatorAllGatherRegistry, BatchCollectiveCommunicationServiceOperatorGatherRegistry


@dataclass()
class CollectiveCommunicationIterationCounter:
    all_gather = 0
    gather = 0


class CollectiveCommunicationService(Protocol):
    def all_gather(self, object_: Any, index: int) -> Sequence[Any]:
        ...


class DummyDistributedCommunication:
    @staticmethod
    def all_gather(object_, index: int = 0):
        return object_,


def _run_batching_all_gather(collective_communication: CollectiveCommunicationService,
                             batch_service_registry_operator_all_gather_list: Sequence[BatchCollectiveCommunicationServiceOperatorAllGatherRegistry],
                             iteration: int):
    batch = []
    for registry in batch_service_registry_operator_all_gather_list:
        for context in registry.list():
            batch.append(context.data_prepare_fn())

    if len(batch) == 0:
        return
    gathered_all_rank_batches = collective_communication.all_gather(batch, index=iteration)
    assert all(len(rank_batch) == len(batch) for rank_batch in gathered_all_rank_batches)
    index = 0
    for registry in batch_service_registry_operator_all_gather_list:
        for context in registry.list():
            context.on_gathered_fn(tuple(rank_batch[index] for rank_batch in gathered_all_rank_batches))
            index += 1


def _run_batching_gather(collective_communication: CollectiveCommunicationService,
                         batched_service_registry_operator_gather_list: Sequence[BatchCollectiveCommunicationServiceOperatorGatherRegistry],
                         iteration: int):
    batch = []
    for registry in batched_service_registry_operator_gather_list:
        for context in registry.list():
            batch.append(context.data_prepare_fn())

    if len(batch) == 0:
        return
    gathered = collective_communication.all_gather(batch, index=iteration)
    assert all(len(rank_batch) == len(batch) for rank_batch in gathered)
    index = 0
    for registry in batched_service_registry_operator_gather_list:
        for context in registry.list():
            if context.dst_rank == get_rank():
                context.on_gathered_fn(tuple(rank_batch[index] for rank_batch in gathered))
            index += 1


def _run(all_registry: Sequence[BatchCollectiveCommunicationServiceRegistry],
         collective_communication: CollectiveCommunicationService,
         iteration_counter: CollectiveCommunicationIterationCounter):
    registry_list_do_all_gather = tuple(registry.all_gather for registry in all_registry if not registry.all_gather.is_empty())
    if len(registry_list_do_all_gather) > 0:
        _run_batching_all_gather(collective_communication, registry_list_do_all_gather, iteration_counter.all_gather)
        iteration_counter.all_gather += 1
    registry_list_do_gather = tuple(registry.gather for registry in all_registry if not registry.gather.is_empty())
    if len(registry_list_do_gather) > 0:
        _run_batching_gather(collective_communication, registry_list_do_gather, iteration_counter.all_gather)
        iteration_counter.gather += 1


def _has_object_in_registries(batch_collective_communication_service_registry_list: Sequence[BatchCollectiveCommunicationServiceRegistry]):
    for registry in batch_collective_communication_service_registry_list:
        if not registry.is_empty():
            return True
    return False


class BatchCollectiveCommunicationFrequency(Enum):
    normal = auto()
    high = auto()


class BatchCollectiveCommunication:
    def begin(self, batch_collective_communication_service_registry_list: Sequence[BatchCollectiveCommunicationServiceRegistry]):
        self._batch_collective_communication_service_registry_list = batch_collective_communication_service_registry_list
        self._iteration_counter = CollectiveCommunicationIterationCounter()

    def end(self):
        del self._iteration_counter
        del self._batch_collective_communication_service_registry_list

    def set_mode(self, frequency: BatchCollectiveCommunicationFrequency):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class TimedBatchCollectiveCommunication_FixedStepInterval(BatchCollectiveCommunication):
    def __init__(self, step_interval: int = 1):
        self._normal_step_interval = step_interval
        self._low_step_interval = 1

    def begin(self, batch_collective_communication_service_registry_list: Sequence[BatchCollectiveCommunicationServiceRegistry]):
        super(TimedBatchCollectiveCommunication_FixedStepInterval, self).begin(batch_collective_communication_service_registry_list)
        self._step_interval = self._normal_step_interval
        self._step_interval_changed = False
        if is_dist_initialized() and _has_object_in_registries(batch_collective_communication_service_registry_list):
            self._collective_communication = CollectiveCommunication()
        else:
            self._collective_communication = DummyDistributedCommunication
        self._step = 0

    def run(self):
        self._step += 1
        if self._step % self._step_interval == 0:
            _run(self._batch_collective_communication_service_registry_list, self._collective_communication, self._iteration_counter)

    def set_mode(self, frequency: BatchCollectiveCommunicationFrequency):
        if frequency == BatchCollectiveCommunicationFrequency.normal:
            self._step_interval = self._normal_step_interval
        elif frequency == BatchCollectiveCommunicationFrequency.high:
            self._step_interval = self._low_step_interval
        else:
            raise ValueError("Unknown frequency: {}".format(frequency))
        self._step_interval_changed = True

    def end(self):
        if self._step % self._step_interval != 0 or self._step_interval_changed:
            _run(self._batch_collective_communication_service_registry_list, self._collective_communication, self._iteration_counter)

        del self._step
        del self._collective_communication
        del self._step_interval
        del self._step_interval_changed
        super(TimedBatchCollectiveCommunication_FixedStepInterval, self).end()


class BatchCollectiveCommunication_FixedTimeInterval(BatchCollectiveCommunication):
    def __init__(self, time_interval: float):
        self._time_interval = time_interval

    def begin(self, batch_collective_communication_service_registry_list: Sequence[BatchCollectiveCommunicationServiceRegistry]):
        super(BatchCollectiveCommunication_FixedTimeInterval, self).begin(batch_collective_communication_service_registry_list)
        self._high_freq = False
        if is_dist_initialized() and _has_object_in_registries(batch_collective_communication_service_registry_list):
            self._collective_communication = CollectiveCommunication()
        else:
            self._collective_communication = DummyDistributedCommunication
        self._last_time = time.perf_counter()

    def set_mode(self, frequency: BatchCollectiveCommunicationFrequency):
        if frequency == BatchCollectiveCommunicationFrequency.normal:
            self._high_freq = False
        elif frequency == BatchCollectiveCommunicationFrequency.high:
            self._high_freq = True
        else:
            raise ValueError("Unknown frequency: {}".format(frequency))

    def run(self):
        if self._high_freq or time.perf_counter() - self._last_time >= self._time_interval:
            _run(self._batch_collective_communication_service_registry_list, self._collective_communication, self._iteration_counter)
            self._last_time = time.perf_counter()

    def end(self):
        _run(self._batch_collective_communication_service_registry_list, self._collective_communication, self._iteration_counter)

        del self._collective_communication
        del self._last_time
        del self._high_freq
        super(BatchCollectiveCommunication_FixedTimeInterval, self).end()
