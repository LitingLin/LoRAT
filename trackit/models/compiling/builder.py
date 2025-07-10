import torch
from trackit.miscellanies.printing import pretty_format
from . import InferenceEngine


def build_inference_engine(config: dict, device: torch.device) -> InferenceEngine:
    print(f"Inference engine:\n" + pretty_format(config, indent_level=1))
    if config['type'] == 'plain':
        from .plain.builder import build_plain_inference_engine
        return build_plain_inference_engine(config, device)
    elif config['type'] == 'torch.jit.trace':
        from .torch_jit_trace import TorchJITTraceInferenceEngine
        return TorchJITTraceInferenceEngine()
    else:
        raise ValueError(config['type'])
