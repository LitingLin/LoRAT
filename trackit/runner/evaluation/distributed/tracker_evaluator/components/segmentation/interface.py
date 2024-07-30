import torch
from typing import Optional, Tuple


class MaskGenerator:
    def start(self):
        pass

    def stop(self):
        pass

    def __call__(self, search_region: torch.tensor, predicted_bbox: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_recommended_input_resolution(self) -> Optional[Tuple[int, int]]:
        raise NotImplementedError()
