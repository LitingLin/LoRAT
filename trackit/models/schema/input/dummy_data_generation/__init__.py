import torch
from typing import Any, Sequence


class SampleInputDataGeneratorInterface:
    def get(self, batch_size: int, device: torch.device) -> Any:
        raise NotImplementedError()


class SampleInputDataGeneratorInterface_MultiPath:
    def get(self, name: str, batch_size: int, device: torch.device) -> Any:
        raise NotImplementedError()

    def get_path_names(self, with_train: bool = True, with_eval: bool = True) -> Sequence[str]:
        raise NotImplementedError()
