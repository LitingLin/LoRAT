from typing import Any, Optional

import torch
import torch.nn as nn
import inspect
from fvcore.nn import FlopCountAnalysis, flop_count_table
from trackit.miscellanies.torch.amp_autocast import get_torch_amp_autocast_fn
from trackit.models import ModelManager
from trackit.models.schema.data_schema import get_model_input_output_data_schema, ModelInputOutputDataSchema
from .._execution_path_iterator import iterate_model_execution_paths
from .op_handles.scaled_dot_product_attention import scaled_dot_product_attention_flop_jit


def _flatten_kwargs(signature: inspect.Signature, kwargs: dict) -> tuple:
    bound_args = signature.bind(**kwargs)
    bound_args.apply_defaults()
    assert len(bound_args.args) > 0 and len(bound_args.kwargs) == 0, "failed to flatten kwargs according to function signature"
    return bound_args.args


def _create_flop_count_analysis(model: nn.Module, data: Any,) -> FlopCountAnalysis:
    if get_model_input_output_data_schema(data) == ModelInputOutputDataSchema.Dict:
        forward_signature = inspect.signature(model.forward)
        data = _flatten_kwargs(forward_signature, data)
    flop_count_analysis = FlopCountAnalysis(model, data)
    flop_count_analysis.set_op_handle('aten::scaled_dot_product_attention', scaled_dot_product_attention_flop_jit)
    return flop_count_analysis


class ModelFlopAnalyzer:
    def __init__(self, model: nn.Module, input_data: Any, device: torch.device, amp_dtype: Optional[torch.dtype] = None):
        self._flop_analysis = _create_flop_count_analysis(model, input_data)
        self._amp_autocast_fn = get_torch_amp_autocast_fn(device.type, amp_dtype is not None, amp_dtype)

    @property
    def flop_table(self):
        with self._amp_autocast_fn(), torch.inference_mode():
            return flop_count_table(self._flop_analysis, max_depth=10)

    @property
    def flops_by_module_and_operator(self):
        with self._amp_autocast_fn(), torch.inference_mode():
            return self._flop_analysis.by_module_and_operator()

    @property
    def unsupported_ops(self):
        return self._flop_analysis.unsupported_ops()

    @property
    def total_flops(self):
        with self._amp_autocast_fn(), torch.inference_mode():
            return self._flop_analysis.total()


def analyze_model_flops_for_all_paths(model_manager: ModelManager, device: torch.device,
                                      inference_precision, train_inference_precision):
    for execution_path in iterate_model_execution_paths(model_manager, 1, device, inference_precision, train_inference_precision,
                                                        torch_jit_trace_compatible=True):
        yield execution_path.path, ModelFlopAnalyzer(execution_path.model, execution_path.sample_input, device, execution_path.auto_mixed_precision_dtype)
