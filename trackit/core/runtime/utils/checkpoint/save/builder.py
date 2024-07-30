from typing import Union
from . import CheckpointDumper
from ...epoch_activation_criteria.builder import build_epoch_activation_criteria


def _build_checkpoint_sub_dumper(checkpoint_dumper_config: dict, output_path: str, num_epochs: int):
    if checkpoint_dumper_config['type'] == 'regular':
        epoch_activation_criteria = build_epoch_activation_criteria(checkpoint_dumper_config.get('epoch_trigger', None), num_epochs)
        resumable = checkpoint_dumper_config.get('resumable', False)

        from . import EpochEventCheckpointDumper
        return EpochEventCheckpointDumper(output_path, epoch_activation_criteria, num_epochs, resumable, checkpoint_dumper_config.get('max_to_keep', -1))
    elif checkpoint_dumper_config['type'] == 'best':
        metric_name = checkpoint_dumper_config['metric']
        top_k = checkpoint_dumper_config.get('top_k', 1)
        resumable = checkpoint_dumper_config.get('resumable', False)
        epoch_activation_criteria = build_epoch_activation_criteria(checkpoint_dumper_config.get('epoch_trigger', None), num_epochs)

        from . import BestCheckpointDumper
        return BestCheckpointDumper(output_path, metric_name, top_k, epoch_activation_criteria, resumable)
    elif checkpoint_dumper_config['type'] == 'latest':
        resumable = checkpoint_dumper_config.get('resumable', False)

        from . import LatestCheckpointDumper
        return LatestCheckpointDumper(output_path, resumable)
    else:
        raise NotImplementedError(f"Unknown checkpoint dumper type: {checkpoint_dumper_config['type']}")


def build_checkpoint_dumper(checkpoint_dumper_config: Union[dict, list, tuple], output_path: str, num_epochs: int) -> CheckpointDumper:
    if isinstance(checkpoint_dumper_config, (list, tuple)):
        return CheckpointDumper(_build_checkpoint_sub_dumper(sub_config, output_path, num_epochs) for sub_config in checkpoint_dumper_config)
    else:
        return CheckpointDumper((_build_checkpoint_sub_dumper(checkpoint_dumper_config, output_path, num_epochs),))
