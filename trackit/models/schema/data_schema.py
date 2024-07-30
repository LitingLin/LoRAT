import torch
from collections.abc import Iterable, Mapping
from enum import Enum, auto


class ModelInputOutputDataSchema(Enum):
    Singleton = auto()
    List = auto()
    Dict = auto()


def get_model_input_output_data_schema(data):
    if isinstance(data, torch.Tensor):
        return ModelInputOutputDataSchema.Singleton
    elif isinstance(data, Mapping):
        return ModelInputOutputDataSchema.Dict
    elif isinstance(data, Iterable):
        return ModelInputOutputDataSchema.List
    else:
        raise ValueError(data, 'unsupported data schema')
