import torch
from trackit.models import ModelManager, ModelImplSuggestions
from .. import InferenceEngine
from ..auto_trace import enable_inference_engine_by_auto_tracing
from ..trace_fn.torch_jit_trace import trace_module_with_torch_jit_trace


class TorchJITTraceInferenceEngine(InferenceEngine):
    def __call__(self, model_manager: ModelManager, device: torch.device, max_batch_size: int):
        model = model_manager.create(device, ModelImplSuggestions(torch_jit_trace_compatible=True, optimize_for_inference=True)).model
        model.eval()
        dummy_data_generator = model_manager.sample_input_data_generator
        assert dummy_data_generator is not None, "dummy_data_generator must be provided for torch.jit.trace"
        return (enable_inference_engine_by_auto_tracing(model, dummy_data_generator, trace_module_with_torch_jit_trace, max_batch_size, device),
                model)
