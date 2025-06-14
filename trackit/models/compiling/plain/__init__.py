from typing import Callable
import torch
import torch.nn as nn
import torch.amp
from trackit.models.schema.input.auto_unpack import auto_unpack_and_call
from trackit.models import ModelManager
from trackit.miscellanies.torch.amp_autocast import get_torch_amp_autocast_fn
from trackit.miscellanies.torch.data_parallel import should_use_data_parallel
from .torch_compile_option import TorchCompileOption
from .auto_mixed_precision_option import AutoMixedPrecisionOption
from .. import InferenceEngine, OptimizedModel


class PlainWrapper(nn.Module):
    def __init__(self, model: nn.Module, amp_autocast_fn: Callable):
        super().__init__()
        self.model = model
        self.amp_autocast_fn = amp_autocast_fn

    def forward(self, data):
        with torch.inference_mode(), self.amp_autocast_fn():
            return auto_unpack_and_call(data, self.model)


class PlainInferenceEngine(InferenceEngine):
    def __init__(self, enable_data_parallel: bool = False,
                 torch_compile_option: TorchCompileOption = TorchCompileOption(),
                 auto_mixed_precision_option: AutoMixedPrecisionOption = AutoMixedPrecisionOption()):
        self._enable_data_parallel = enable_data_parallel
        self._torch_compile_option = torch_compile_option
        self._auto_mixed_precision_option = auto_mixed_precision_option

    def __call__(self, model_manager: ModelManager, device: torch.device, dtype: torch.dtype,
                 max_batch_size: int, max_num_input_data_streams: int):
        model = model_manager.create(device, dtype=dtype, optimize_for_inference=True).model
        model.eval()
        return OptimizedModel(self._apply(model, device), model, device, dtype,
                              self._auto_mixed_precision_option.dtype if self._auto_mixed_precision_option.enabled else None)

    def _apply(self, model: torch.nn.Module, device: torch.device):
        if self._torch_compile_option.enabled:
            if self._torch_compile_option.parameters is None:
                model = torch.compile(model)
                print('torch.compile is enabled.')
            else:
                if self._torch_compile_option.parameters.get('backend', None) == 'tensorrt':
                    import torch_tensorrt
                model = torch.compile(model, **self._torch_compile_option.parameters)
                print(f'torch.compile is enabled with options {self._torch_compile_option.parameters}.')

        if self._enable_data_parallel and should_use_data_parallel(device):
            model = torch.nn.DataParallel(model, output_device=device)
            print('DataParallel is enabled.')

        amp_autocast_fn = get_torch_amp_autocast_fn(device.type,
                                                    self._auto_mixed_precision_option.enabled,
                                                    self._auto_mixed_precision_option.dtype)
        return PlainWrapper(model, amp_autocast_fn)
