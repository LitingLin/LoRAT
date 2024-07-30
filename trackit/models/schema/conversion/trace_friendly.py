import torch
from collections.abc import Mapping, Iterable
from typing import Callable, Any, Tuple
from trackit.models.schema.data_schema import get_model_input_output_data_schema, ModelInputOutputDataSchema
from dataclasses import dataclass


def singleton_schema_flatten(data: torch.Tensor):
    return (data, ), None


def singleton_schema_restore(flatten_data: Tuple[torch.Tensor], restore_context):
    assert len(flatten_data) == 1
    return flatten_data[0]


def list_schema_flatten(data: Iterable):
    flattened_restore_index = []
    flattened_elements = []
    no_use_restore_index = []
    no_use_elements = []
    for index, element in enumerate(data):
        if isinstance(element, torch.Tensor):
            flattened_elements.append(element)
            flattened_restore_index.append(index)
        else:
            no_use_elements.append(element)
            no_use_restore_index.append(index)

    return tuple(flattened_elements), (flattened_restore_index, no_use_restore_index, no_use_elements)


def list_schema_restore(flattened_data: Tuple[torch.Tensor], restore_context):
    flattened_restore_index, no_use_restore_index, no_use_elements = restore_context
    restored_data = [None] * (len(flattened_restore_index) + len(no_use_restore_index))
    for index, flatten_element in zip(flattened_restore_index, flattened_data):
        restored_data[index] = flatten_element

    for index, no_use_element in zip(no_use_restore_index, no_use_elements):
        restored_data[index] = no_use_element
    return restored_data


def dict_schema_flatten(data: Mapping):
    flattened_restore_key_names = []
    flattened_elements = []
    no_use_restore_key_names = []
    no_use_elements = []
    for element_key, element_value in sorted(data.items()):
        if isinstance(element_value, torch.Tensor):
            flattened_elements.append(element_value)
            flattened_restore_key_names.append(element_key)
        else:
            no_use_elements.append(element_value)
            no_use_restore_key_names.append(element_key)
    return tuple(flattened_elements), (flattened_restore_key_names, no_use_restore_key_names, no_use_elements)


def dict_schema_restore(flattened_data: Tuple[torch.Tensor], restore_context):
    flattened_restore_key_names, no_use_restore_key_names, no_use_elements = restore_context
    restored_data = {}

    for element_key, element_value in zip(flattened_restore_key_names, flattened_data):
        restored_data[element_key] = element_value

    for element_key, element_value in zip(no_use_restore_key_names, no_use_elements):
        restored_data[element_key] = element_value

    return restored_data


@dataclass
class TraceFriendlyDataAdaptor:
    schema: ModelInputOutputDataSchema

    flatten_fn: Callable[[Any], Tuple[Any, Any]]
    restore_context: Any
    restore_fn: Callable[[Any, Any], Any]

    def flatten(self, data):
        flattened, _ = self.flatten_fn(data)
        return flattened

    def restore(self, data):
        return self.restore_fn(data, self.restore_context)

    def get_data_names(self):
        if self.schema == ModelInputOutputDataSchema.Dict:
            return self.restore_context[0]
        else:
            return None


def get_trace_friendly_data_adaptor(data):
    data_type = get_model_input_output_data_schema(data)

    if data_type == ModelInputOutputDataSchema.Singleton:
        flatten_fn = singleton_schema_flatten
        restore_fn = singleton_schema_restore
    elif data_type == ModelInputOutputDataSchema.List:
        flatten_fn = list_schema_flatten
        restore_fn = list_schema_restore
    elif data_type == ModelInputOutputDataSchema.Dict:
        flatten_fn = dict_schema_flatten
        restore_fn = dict_schema_restore
    else:
        raise NotImplementedError()

    flattened_data, restore_context = flatten_fn(data)
    return TraceFriendlyDataAdaptor(data_type, flatten_fn, restore_context, restore_fn), flattened_data
