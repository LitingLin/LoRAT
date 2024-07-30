import torch.nn as nn
import numpy as np
from typing import Optional, Any, Dict, Sequence
from dataclasses import dataclass, field
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT


@dataclass()
class _Result:
    box: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    mask: Optional[np.ndarray] = None


class TrackerEvaluationPipeline_ResultHolder:
    def __init__(self, tracking_task_id_list: Sequence[int]):
        self._results: Dict[int, Optional[_Result]] = {task_id: None for task_id in tracking_task_id_list}

    def submit(self, id_: int, box: Optional[np.ndarray] = None, confidence: Optional[float] = None, mask: Optional[np.ndarray] = None):
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
    input_data: TrackerEvalData
    all_tracks: Dict[int, SequenceEvaluationResult_SOT]
    global_objects: dict = field(default_factory=dict)
    temporary_objects: dict = field(default_factory=dict)
    result: TrackerEvaluationPipeline_ResultHolder = field(init=False)

    def __post_init__(self):
        tracking_task_id_list = []
        for tracking_task in self.input_data.tasks:
            if tracking_task.tracker_do_tracking_context is not None:
                tracking_task_id_list.append(tracking_task.id)
        object.__setattr__(self, 'result', TrackerEvaluationPipeline_ResultHolder(tracking_task_id_list))

    def batch_size(self):
        return len(tuple(tracking_task for tracking_task in self.input_data.tasks if tracking_task.tracker_do_tracking_context is not None))


class TrackerEvaluationPipeline:
    def start(self, max_batch_size: int, global_objects: dict):
        pass

    def stop(self, global_objects: dict):
        pass

    def begin(self, context: TrackerEvaluationPipeline_Context):
        pass

    def prepare_initialization(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        pass

    def on_initialized(self, model_outputs, context: TrackerEvaluationPipeline_Context):
        pass

    def prepare_tracking(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        pass

    def on_tracked(self, model_outputs, context: TrackerEvaluationPipeline_Context):
        pass

    def do_custom_update(self, model, raw_model: nn.Module, context: TrackerEvaluationPipeline_Context):
        pass

    def end(self, context: TrackerEvaluationPipeline_Context):
        pass
