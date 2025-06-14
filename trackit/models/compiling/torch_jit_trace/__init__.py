import torch
from trackit.models import ModelManager
from .. import InferenceEngine, OptimizedModel
from ..auto_trace import enable_inference_engine_by_auto_tracing
from ..trace_fn.torch_jit_trace import trace_module_with_torch_jit_trace


class TorchJITTraceInferenceEngine(InferenceEngine):
    def __call__(self, model_manager: ModelManager, device: torch.device, dtype: torch.dtype,
                 max_batch_size: int, max_num_input_data_streams: int):
        model = model_manager.create(device, dtype, torch_jit_trace_compatible=True, optimize_for_inference=True).model
        model.eval()
        return OptimizedModel(
            enable_inference_engine_by_auto_tracing(model,
                                                    trace_module_with_torch_jit_trace,
                                                    max_batch_size, device, dtype),
            model, device, dtype, None)
