from dataclasses import dataclass
from typing import Dict, Iterable, Any, Optional

from trackit.core.runtime.context.task import TaskContext
from trackit.data.context import DataContext
from trackit.runners import Runner
from trackit.core.runtime.metric_logger import MetricLogger, LocalMetricLoggerWrapper
from trackit.core.runtime.services import ServicesRegistry
from trackit.core.runtime.services.garbage_collection import GarbageCollection
from trackit.core.runtime.services.checkpoint import get_state_from_registries, load_state_to_registries
from trackit.runners.context import RunnerContext


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
        self._step = 0
        self._sample_processed = 0

    def update(self, step_batch_size: int = 1):
        self._step += 1
        self._sample_processed += step_batch_size

    def get_iteration(self):
        return self._step

    def get_sample_processed(self):
        return self._sample_processed

    def get_state(self):
        return self._step, self._sample_processed

    def load_state(self, state):
        self._step, self._sample_processed = state


@dataclass(frozen=True)
class ApplicationTaskContext:
    context: TaskContext
    metric_logger: MetricLogger
    local_metric_logger: LocalMetricLoggerWrapper
    services_registry: ServicesRegistry
    garbage_collection: GarbageCollection


@dataclass(frozen=True)
class ApplicationDataContext:
    context: DataContext
    batch_size: Optional[int]
    services_registry: ServicesRegistry
    data_input_pipeline: Iterable[Any]


@dataclass(frozen=True)
class ApplicationRunnerContext:
    context: RunnerContext
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
    tasks: Dict[str, ApplicationTaskContext]
    epoch: EpochIterator
    iteration: GlobalIterationCounter

    def state_dict(self, checkpoint_path: str):
        state_dict = {}
        state_dict['epoch'] = self.epoch.get_state()
        state_dict['iteration'] = self.iteration.get_state()

        runner_states = {}
        for runner_name, runner in self.runners.items():
            runner_states[runner_name] = runner.runner.get_state(checkpoint_path)
        state_dict['runners'] = runner_states

        checkpoint_registries = self._get_checkpoint_registries()
        state_dict['objects'] = get_state_from_registries(checkpoint_registries)

        state_dict['rng'] = _get_rng_state()
        return state_dict

    def load_state_dict(self, state_dict: dict, checkpoint_path: str):
        self.epoch.load_state(state_dict['epoch'])
        self.iteration.load_state(state_dict['iteration'])

        runner_states = state_dict['runners']
        for runner_name, runner_state in runner_states.items():
            self.runners[runner_name].runner.set_state(runner_state, checkpoint_path)

        checkpoint_registries = self._get_checkpoint_registries()
        load_state_to_registries(state_dict['objects'], checkpoint_registries)

        _load_rng_state(state_dict['rng'])

    def _get_checkpoint_registries(self):
        checkpoint_registries = []
        for data_context in self.data_inputs.values():
            checkpoint_registries.append((f'/data/{data_context.context.name}/', data_context.services_registry.checkpoint))
        for runner_context in self.runners.values():
            checkpoint_registries.append(
                (f'/runner/{runner_context.context.name}/', runner_context.services_registry.checkpoint))
        return checkpoint_registries
