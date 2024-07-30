from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class RunnerContext:
    name: str
    variables: Mapping


__context: Optional[RunnerContext] = None


def get_current_runner_context():
    return __context


def set_current_runner_context(context: Optional[RunnerContext]):
    global __context
    __context = context
