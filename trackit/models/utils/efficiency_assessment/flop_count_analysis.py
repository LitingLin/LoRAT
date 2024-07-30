from typing import Any

import torch
import torch.nn as nn
import inspect
from fvcore.nn import FlopCountAnalysis, flop_count_table

from trackit.models import ModelManager
from trackit.models.schema.data_schema import get_model_input_output_data_schema, ModelInputOutputDataSchema
from ._execution_path_iterator import iterate_model_execution_paths


def _flatten_kwargs(signature: inspect.Signature, kwargs: dict) -> tuple:
    bound_args = signature.bind(**kwargs)
    bound_args.apply_defaults()
    assert len(bound_args.kwargs) == 0
    return bound_args.args


def _create_flop_count_analysis(model: nn.Module, data: Any) -> FlopCountAnalysis:
    if get_model_input_output_data_schema(data) == ModelInputOutputDataSchema.Dict:
        forward_signature = inspect.signature(model.forward)
        data = _flatten_kwargs(forward_signature, data)
    return FlopCountAnalysis(model, data)


class ModelFlopAnalyzer:
    def __init__(self, model: nn.Module, input_data: Any):
        self._flop_analysis = _create_flop_count_analysis(model, input_data)

    @property
    def flop_table(self):
        return flop_count_table(self._flop_analysis, max_depth=10)

    @property
    def flops_by_module_and_operator(self):
        return self._flop_analysis.by_module_and_operator()

    @property
    def unsupported_ops(self):
        return self._flop_analysis.unsupported_ops()

    @property
    def total_flops(self):
        return self._flop_analysis.total()


def analyze_model_flops_for_all_paths(model_manager: ModelManager, device: torch.device):
    for execution_path in iterate_model_execution_paths(model_manager, 1, device, torch_jit_trace_compatible=True):
        yield execution_path.path, ModelFlopAnalyzer(execution_path.model, execution_path.sample_input)
