from dataclasses import dataclass
from typing import Iterable, Any, Sequence, Optional

import torch.nn as nn


class MainProcessDataPipeline:
    def start(self, epoch: int, model: nn.Module):
        pass

    def stop(self, epoch: int):
        pass

    def pre_process(self, input_data: Any) -> Any:
        return input_data

    def post_process(self, output_data: Any) -> Any:
        return output_data


@dataclass(frozen=True)
class DataPipeline:
    input: Iterable[Any]
    on_main_process: Optional[Sequence[MainProcessDataPipeline]] = None
