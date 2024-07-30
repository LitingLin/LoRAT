from dataclasses import dataclass
from typing import Dict, Iterable, Any, Optional
from trackit.runner import Runner
from trackit.core.runtime.metric_logger import MetricLogger, LocalMetricLoggerWrapper
from trackit.core.runtime.services import ServicesRegistry
from trackit.core.runtime.services.batch_collective_communication import BatchCollectiveCommunication
from trackit.core.runtime.utils.epoch_activation_criteria import EpochActivationCriterion
from trackit.core.runtime.services.checkpoint import get_state_from_registries, load_state_to_registries


class EpochIterator:
    def __init__(self, epochs):
        self.epochs = epochs
        self.current_epoch = 0

    def get_current(self):
        return self.current_epoch

    def __len__(self):
        return self.epochs

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_epoch >= self.epochs:
            raise StopIteration
        else:
            current_epoch = self.current_epoch
            self.current_epoch += 1
            return current_epoch

    def get_state(self):
        return self.current_epoch

    def load_state(self, state):
        self.current_epoch = state


class GlobalIterationCounter:
    def __init__(self):
        self._iter = 0
        self._iter = 0

    def update(self, value: int = 1):
        self._iter += value

    def get_iteration(self):
        return self._iter

    def get_state(self):
        return self._iter

    def load_state(self, state):
        self._iter = state


@dataclass(frozen=True)
class ApplicationTaskDescription:
    name: str
    data_name: str
    runner_name: str
    epoch_activation_criteria: Optional[EpochActivationCriterion]
    metric_logger: MetricLogger
    local_metric_logger: LocalMetricLoggerWrapper
    is_train: bool
    services_registry: ServicesRegistry
    collective_communication: BatchCollectiveCommunication


@dataclass(frozen=True)
class ApplicationDataContext:
    name: str
    batch_size: Optional[int]
    services_registry: ServicesRegistry
    data_input_pipeline: Iterable[Any]


@dataclass(frozen=True)
class ApplicationRunnerContext:
    name: str
    services_registry: ServicesRegistry
    runner: Runner


def _get_rng_state():
    import numpy as np
    import random
    import torch
    return {'numpy': np.random.get_state(),
            'python': random.getstate(),
            'torch': torch.get_rng_state()}


def _load_rng_state(state_dict: dict):
    import numpy as np
    import random
    import torch
    np.random.set_state(state_dict['numpy'])
    random.setstate(state_dict['python'])
    torch.set_rng_state(state_dict['torch'])


@dataclass(frozen=True)
class ApplicationContext:
    data_inputs: Dict[str, ApplicationDataContext]
    runners: Dict[str, ApplicationRunnerContext]
    tasks: Dict[str, ApplicationTaskDescription]
    epoch: EpochIterator
    iteration: GlobalIterationCounter

    def state_dict(self):
        state_dict = {}
        state_dict['epoch'] = self.epoch.get_state()
        state_dict['iteration'] = self.iteration.get_state()

        checkpoint_registries = self._get_checkpoint_registries()
        state_dict['objects'] = get_state_from_registries(checkpoint_registries)

        state_dict['rng'] = _get_rng_state()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.epoch.load_state(state_dict['epoch'])
        self.iteration.load_state(state_dict['iteration'])

        checkpoint_registries = self._get_checkpoint_registries()
        load_state_to_registries(state_dict['objects'], checkpoint_registries)

        _load_rng_state(state_dict['rng'])

    def _get_checkpoint_registries(self):
        checkpoint_registries = []
        for data_context in self.data_inputs.values():
            checkpoint_registries.append((f'/data/{data_context.name}/', data_context.services_registry.checkpoint))
        for runner_context in self.runners.values():
            checkpoint_registries.append(
                (f'/runner/{runner_context.name}/', runner_context.services_registry.checkpoint))
        return checkpoint_registries
