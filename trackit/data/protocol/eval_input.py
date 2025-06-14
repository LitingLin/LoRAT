from typing import NamedTuple, Sequence, Optional, Mapping
import numpy as np
from PIL import Image
from . import SequenceInfo


class TrackerEvalData_FrameData(NamedTuple):
    frame_index: int
    gt_bbox: Optional[np.ndarray]
    gt_mask: Optional[Image.Image]
    input_data: dict


class TrackerEvalData_TaskDesc(NamedTuple):
    id: int  # global unique task index

    task_creation_context: Optional[SequenceInfo]
    tracker_do_init_context: Optional[TrackerEvalData_FrameData]
    tracker_do_tracking_context: Optional[TrackerEvalData_FrameData]
    do_task_finalization: bool


class TrackerEvalData(NamedTuple):
    tasks: Sequence[TrackerEvalData_TaskDesc]
    miscellanies: Mapping
