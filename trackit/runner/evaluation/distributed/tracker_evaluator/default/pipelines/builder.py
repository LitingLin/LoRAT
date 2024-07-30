import torch
from typing import Sequence

from . import TrackerEvaluationPipeline


def build_tracker_evaluator_data_pipeline(evaluator_config: dict, config: dict, device: torch.device) -> Sequence[TrackerEvaluationPipeline]:
    pipeline_config = evaluator_config['pipeline']
    if pipeline_config['type'] == 'one_stream_tracker':
        from .one_stream.builder import build_one_stream_tracker_pipeline
        return build_one_stream_tracker_pipeline(pipeline_config, config, device)
    else:
        raise NotImplementedError("Unknown tracker evaluator type: {}".format(pipeline_config['type']))
