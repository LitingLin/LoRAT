import numpy as np
import copy
from typing import Optional


class CroppingParameterProvider:
    def initialize(self, bounding_box: np.ndarray) -> None:
        raise NotImplementedError

    def get(self, cropped_image_size: np.ndarray, area_factor: float | None = None) -> np.ndarray:
        raise NotImplementedError

    def update(self, confidence: Optional[float], bounding_box: np.ndarray, image_size: np.ndarray) -> None:
        raise NotImplementedError


class CroppingParameterProviderFactory:
    def __init__(self, provider: CroppingParameterProvider):
        self.provider = provider

    def __call__(self):
        return copy.deepcopy(self.provider)
