import os
from dataclasses import dataclass, field
from typing import Optional, Mapping, Any

from trackit.core.runtime.utils.execution_trigger import ExecutionCriterion
from trackit.core.runtime.services.batch_collective_communication import BatchCollectiveCommunication
from .epoch import get_current_epoch_context


@dataclass()
class _MutableVariables:
    output_path_exists: bool = False


@dataclass(frozen=True)
class TaskContext:
    name: str
    is_train: bool
    _output_path: Optional[str]
    data_name: str
    runner_name: str
    epoch_selector: ExecutionCriterion
    variables: Mapping[str, Any]
    collective_communication: BatchCollectiveCommunication

    __mutable_vars: _MutableVariables = field(default_factory=_MutableVariables)

    def get_output_path(self, create_if_not_exists: bool = True) -> Optional[str]:
        if create_if_not_exists:
            if not self.__mutable_vars.output_path_exists and self._output_path is not None:
                os.makedirs(self._output_path, exist_ok=True)
                self.__mutable_vars.output_path_exists = True
        return self._output_path

    def get_current_epoch_output_path(self, create_if_not_exists: bool = True) -> Optional[str]:
        if self._output_path is None:
            return None
        epoch_context = get_current_epoch_context()
        assert epoch_context is not None, 'Epoch context is not set'
        epoch_output_path = os.path.join(self._output_path, f'epoch_{epoch_context.epoch}')
        if create_if_not_exists:
            os.makedirs(epoch_output_path, exist_ok=True)
        return epoch_output_path


__task_context: Optional[TaskContext]


def get_current_task_context() -> Optional[TaskContext]:
    return __task_context


def set_current_task_context(context: Optional[TaskContext]):
    global __task_context
    __task_context = context
