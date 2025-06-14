from typing import Dict, Iterable, Tuple
from trackit.core.runtime.context.epoch import EpochContext, set_current_epoch_context
from trackit.core.runtime.context.task import TaskContext, set_current_task_context
from trackit.data.context import DataContext, set_current_data_context
from trackit.runners.context import set_current_runner_context, RunnerContext


class GlobalContextManager:
    def __init__(self):
        self._data_contexts: Dict[str, DataContext] = {}
        self._runner_contexts: Dict[str, RunnerContext] = {}
        self._task_contexts: Dict[str, TaskContext] = {}

    def set_data_context(self, data_name: str, context: DataContext):
        self._data_contexts[data_name] = context

    def set_runner_context(self, runner_name: str, context: RunnerContext):
        self._runner_contexts[runner_name] = context

    def set_task_context(self, task_name: str, context: TaskContext):
        self._task_contexts[task_name] = context

    def get_data_context(self, data_name: str) -> DataContext:
        return self._data_contexts[data_name]

    def get_runner_context(self, runner_name: str) -> RunnerContext:
        return self._runner_contexts[runner_name]

    def get_task_context(self, task_name: str) -> TaskContext:
        return self._task_contexts[task_name]

    def get_data_context_iterator(self) -> Iterable[Tuple[str, DataContext]]:
        return self._data_contexts.items()

    def get_runner_context_iterator(self) -> Iterable[Tuple[str, RunnerContext]]:
        return self._runner_contexts.items()

    def get_task_context_iterator(self) -> Iterable[Tuple[str, TaskContext]]:
        return self._task_contexts.items()

    def activate(self, task_name: str, epoch: int):
        task_context = self._task_contexts[task_name]
        data_name = task_context.data_name
        runner_name = task_context.runner_name
        data_context = self._data_contexts[data_name]
        runner_context = self._runner_contexts[runner_name]
        set_current_data_context(data_context)
        set_current_runner_context(runner_context)
        set_current_task_context(task_context)
        set_current_epoch_context(EpochContext(epoch))

    @staticmethod
    def finalize():
        set_current_data_context(None)
        set_current_runner_context(None)
        set_current_task_context(None)
        set_current_epoch_context(None)
