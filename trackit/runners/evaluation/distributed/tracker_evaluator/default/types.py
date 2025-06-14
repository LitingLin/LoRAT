from typing import Any
from typing import Dict
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class _Result:
    box: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    mask: Optional[Image.Image] = None


class TrackingPipeline_ResultHolder:
    def __init__(self, tracking_task_id_list: Sequence[int]):
        self._results: Dict[int, Optional[_Result]] = {task_id: None for task_id in tracking_task_id_list}

    def submit(self,
               id_: int,
               box: Optional[np.ndarray] = None,
               confidence: Optional[float] = None,
               mask: Optional[Image.Image] = None):
        assert id_ in self._results, f"Invalid id: {id_}"
        self._results[id_] = _Result(box, confidence, mask)

    def is_all_submitted(self) -> bool:
        return all(result is not None for result in self._results.values())

    def get_all(self) -> Dict[Any, Optional[_Result]]:
        return self._results

    def get(self, id_: int) -> Optional[_Result]:
        return self._results[id_]


@dataclass(frozen=True)
class TrackerEvaluationPipeline_Context:
    history: Dict[int, SequenceEvaluationResult_SOT]
    global_objects: dict = field(default_factory=dict)
    temporary_objects: dict = field(default_factory=dict)


