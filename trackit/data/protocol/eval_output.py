from typing import NamedTuple, Tuple
import numpy as np
from PIL import Image
from typing import Optional
from . import SequenceInfo


class SequenceEvaluationResult_SOT(NamedTuple):
    id: int
    sequence_info: SequenceInfo
    evaluated_frame_indices: np.ndarray
    groundtruth_box: Optional[np.ndarray]
    groundtruth_object_existence_flag: Optional[np.ndarray]
    groundtruth_mask: Optional[Tuple[Image.Image, ...]]
    output_box: Optional[np.ndarray]
    output_confidence: Optional[np.ndarray]
    output_mask: Optional[Tuple[Image.Image, ...]]
    time_cost: np.ndarray
    batch_size: np.ndarray


class FrameEvaluationResult_SOT(NamedTuple):
    id: int
    frame_index: int
    sequence_info: SequenceInfo
    groundtruth_box: Optional[np.ndarray]
    groundtruth_object_existence_flag: Optional[float]
    groundtruth_mask: Optional[Image.Image]
    output_box: Optional[np.ndarray]
    output_confidence: Optional[float]
    output_mask: Optional[Image.Image]
    time_cost: float
