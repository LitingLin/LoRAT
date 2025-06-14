import torch
from dataclasses import dataclass
from typing import Optional, Mapping, Any


@dataclass
class DataContext:
    name: str
    variables: Mapping[str, Any]
    dtype: torch.dtype


__data_context: Optional[DataContext] = None


def get_current_data_context() -> Optional[DataContext]:
    return __data_context


def set_current_data_context(context: Optional[DataContext]):
    global __data_context
    __data_context = context
