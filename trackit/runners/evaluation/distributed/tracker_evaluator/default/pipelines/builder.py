import torch


def build_tracker_evaluator_pipeline(pipeline_config: dict, config: dict, device: torch.device, num_epochs: int):
    if pipeline_config['type'] == 'one_stream_tracker':
        from .one_stream.builder import build_one_stream_tracker_pipeline
        return build_one_stream_tracker_pipeline(pipeline_config, config, device)
    else:
        raise NotImplementedError(pipeline_config['type'])
