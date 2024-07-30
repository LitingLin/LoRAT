import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class _MutableVariables:
    output_path_exists: bool = False


@dataclass(frozen=True)
class EpochContext:
    epoch: int
    __mutable_variables: _MutableVariables = field(default_factory=_MutableVariables)

    def get_current_epoch_output_path(self, create_if_not_exists: bool = True) -> Optional[str]:
        from .task import get_current_task_context
        output_path = get_current_task_context().get_output_path(create_if_not_exists)
        if output_path is not None:
            output_path = os.path.join(output_path, f'epoch_{self.epoch}')
        if not self.__mutable_variables.output_path_exists and output_path is not None:
            if create_if_not_exists:
                try:
                    os.mkdir(output_path)
                except FileExistsError:
                    pass
                self.__mutable_variables.output_path_exists = True
        return output_path


_context: Optional[EpochContext] = None


def get_current_epoch_context() -> Optional[EpochContext]:
    return _context


def set_current_epoch_context(context: Optional[EpochContext]):
    global _context
    _context = context
