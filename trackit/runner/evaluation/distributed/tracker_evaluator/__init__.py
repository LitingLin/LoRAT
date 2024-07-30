import torch.nn as nn
from typing import Any, Optional
from trackit.data.protocol.eval_input import TrackerEvalData


class TrackerEvaluator:
    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def begin(self, data: TrackerEvalData):
        raise NotImplementedError()

    def prepare_initialization(self) -> Any:
        raise NotImplementedError()

    def on_initialized(self, model_init_output: Any):
        raise NotImplementedError()

    def prepare_tracking(self) -> Any:
        raise NotImplementedError()

    def on_tracked(self, model_track_outputs: Any):
        raise NotImplementedError()

    def do_custom_update(self, compiled_model: Any, raw_model: Optional[nn.Module]):
        raise NotImplementedError()

    def end(self) -> Any:
        raise NotImplementedError()


def run_tracker_evaluator(tracker_evaluator: TrackerEvaluator, data: Optional[TrackerEvalData],
                          optimized_model: Any, raw_model: Optional[nn.Module]):
    if data is None:
        return None
    tracker_evaluator.begin(data)
    tracker_initialization_params = tracker_evaluator.prepare_initialization()
    tracker_initialization_results = optimized_model(tracker_initialization_params) if tracker_initialization_params is not None else None
    tracker_evaluator.on_initialized(tracker_initialization_results)
    tracker_tracking_params = tracker_evaluator.prepare_tracking()
    tracking_outputs = optimized_model(tracker_tracking_params) if tracker_tracking_params is not None else None
    tracker_evaluator.on_tracked(tracking_outputs)
    tracker_evaluator.do_custom_update(optimized_model, raw_model)
    outputs = tracker_evaluator.end()
    return outputs
