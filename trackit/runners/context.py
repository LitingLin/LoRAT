from dataclasses import dataclass
from typing import Mapping, Optional, Any


@dataclass(frozen=True)
class RunnerContext:
    name: str
    variables: Mapping[str, Any]


__context: Optional[RunnerContext] = None


def get_current_runner_context():
    return __context


def set_current_runner_context(context: Optional[RunnerContext]):
    global __context
    __context = context
