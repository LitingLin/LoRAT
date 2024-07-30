import torch

from . import PlainInferenceEngine
from .torch_compile_option import TorchCompileOption
from .auto_mixed_precision_option import AutoMixedPrecisionOption


def build_plain_inference_engine(config: dict):
    return PlainInferenceEngine(config.get('enable_data_parallel', False), TorchCompileOption.from_config(config['torch_compile']), AutoMixedPrecisionOption.from_config(config['auto_mixed_precision']))
