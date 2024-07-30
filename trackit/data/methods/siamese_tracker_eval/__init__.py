from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional
from trackit.data.protocol.eval_input import SequenceInfo


@dataclass(frozen=True)
class SiameseTrackerEvalDataWorker_FrameContext:
    frame_index: int
    get_image: Callable[[], np.ndarray]
    gt_bbox: Optional[np.ndarray]


@dataclass(frozen=True)
class SiameseTrackerEvalDataWorker_Task:
    task_index: int
    do_task_creation: Optional[SequenceInfo]
    do_tracker_init: Optional[SiameseTrackerEvalDataWorker_FrameContext]
    do_tracker_track: Optional[SiameseTrackerEvalDataWorker_FrameContext]
    do_task_finalization: bool
