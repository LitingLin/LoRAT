import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Optional
from trackit.data.source import TrackingDataset_Sequence, TrackingDataset_Track, TrackingDataset_FrameInTrack


@dataclass(frozen=True)
class SOTFrameInfo:
    image: Callable[[], np.ndarray]
    object_bbox: np.ndarray
    object_exists: bool
    sequence: TrackingDataset_Sequence
    track: TrackingDataset_Track
    frame: TrackingDataset_FrameInTrack


@dataclass(frozen=True)
class SiameseTrainingPair:
    is_positive: bool
    template: SOTFrameInfo
    search: SOTFrameInfo
    aux_frames: Optional[Sequence[SOTFrameInfo]] = None