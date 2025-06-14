import sys
from dataclasses import dataclass, field
from typing import Optional

from trackit.miscellanies.system.operating_system import get_os_running_on, OperatingSystem
from trackit.miscellanies.torch.check_version import is_torch_version_greater_or_equal

@dataclass(frozen=True)
class CompiledAutogradOptions:
    enabled: bool = False
    dynamic: bool = False
    parameters: Optional[dict] = None
    
    @classmethod
    def from_config(cls, config: Optional[dict]):
        if config is not None:
            enabled = config.get('enabled', False)
            if enabled and get_os_running_on() != OperatingSystem.Linux:  # workaround: to remove when pytorch supports
                enabled = False
            if enabled and not is_torch_version_greater_or_equal((2, 4)):
                print("compiled autograd is only supported on PyTorch >= 2.4", file=sys.stderr)
                enabled = False
            return cls(
                enabled=enabled,
                dynamic=config.get('dynamic', False),
                parameters=config.get('parameters', None)
            )
        else:
            return cls()


@dataclass(frozen=True)
class TorchCompileOptions:
    enabled: bool = False
    parameters: Optional[dict] = None
    compiled_autograd: CompiledAutogradOptions = field(default_factory=CompiledAutogradOptions)
    
    @classmethod
    def from_config(cls, config: Optional[dict]):
        if config is not None:
            enabled = config.get('enabled', False)
            if enabled and get_os_running_on() != OperatingSystem.Linux:  # workaround: to remove when pytorch supports
                print('Only Linux is supported for torch.compile, disabled', file=sys.stderr)
                enabled = False
            
            if enabled and not is_torch_version_greater_or_equal((2,)):
                print("torch.compile is not supported for PyTorch 1.x", file=sys.stderr)
                enabled = False
            return cls(
                enabled=enabled,
                parameters=config.get('parameters', None),
                compiled_autograd=CompiledAutogradOptions.from_config(config.get('compiled_autograd', None))
            )
        else:
            return cls()