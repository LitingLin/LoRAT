from . import InferenceEngine


def build_inference_engine(config: dict) -> InferenceEngine:
    if config['type'] == 'plain':
        from .plain.builder import build_plain_inference_engine
        return build_plain_inference_engine(config)
    elif config['type'] == 'torch.jit.trace':
        from .torch_jit_trace import TorchJITTraceInferenceEngine
        return TorchJITTraceInferenceEngine()
    else:
        raise ValueError(config['type'])
