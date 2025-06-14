import torch
from dataclasses import dataclass
from typing import Optional
from trackit.miscellanies.system.operating_system import get_os_running_on, OperatingSystem


@dataclass(frozen=True)
class TorchCompileOption:
    enabled: bool = False
    parameters: Optional[dict] = None

    @staticmethod
    def from_config(config: dict):
        enable = config['enabled']
        if enable and int(torch.__version__.split('.')[0]) < 2:
            print('torch.compile is only supported in PyTorch 2.0 or above.')
            return TorchCompileOption(False, None)
        if enable and get_os_running_on() != OperatingSystem.Linux:
            print('Only Linux is supported for torch.compile, disabled')
            return TorchCompileOption(False, None)
        return TorchCompileOption(enable, config.get('parameters', None))
