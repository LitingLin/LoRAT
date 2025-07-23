from typing import Sequence, Mapping
from . import CheckpointDumper
from .scheduling.builder import build_checkpoint_scheduler


def build_checkpoint_dumper(checkpoint_dumper_config: Sequence[Mapping], output_path: str,
                            num_epochs: int) -> CheckpointDumper:
    epoch_based_dumpers = []
    step_based_dumpers = []
    metric_based_dumpers = []
    for sub_dumper_config in checkpoint_dumper_config:
        if sub_dumper_config['type'] == 'regular':
            from . import RegularCheckpointDumper
            epoch_selector = build_checkpoint_scheduler(sub_dumper_config.get('epoch_trigger', None))
            resumable = sub_dumper_config.get('resumable', False)
            drop_last = sub_dumper_config.get('drop_last', True)
            max_to_keep = sub_dumper_config.get('max_to_keep', -1)
            exclude_frozen_parameters = sub_dumper_config.get('exclude_frozen_parameters', True)

            epoch_based_dumpers.append(RegularCheckpointDumper(output_path, epoch_selector, drop_last,
                                                               num_epochs, resumable, max_to_keep,
                                                               is_epoch_based=True,
                                                               exclude_frozen_parameters=exclude_frozen_parameters))
        elif sub_dumper_config['type'] == 'regular_step':
            from . import RegularCheckpointDumper
            epoch_selector = build_checkpoint_scheduler(sub_dumper_config.get('step_trigger', None))
            drop_last = sub_dumper_config.get('drop_last', True)
            max_to_keep = sub_dumper_config.get('max_to_keep', -1)
            exclude_frozen_parameters = sub_dumper_config.get('exclude_frozen_parameters', True)
            # "resumable" is not supported for step-based dumper

            step_based_dumpers.append(RegularCheckpointDumper(output_path, epoch_selector, drop_last,
                                                              None, False, max_to_keep,
                                                              is_epoch_based=False,
                                                              exclude_frozen_parameters=exclude_frozen_parameters))
        elif sub_dumper_config['type'] == 'best':
            metric_name = sub_dumper_config['metric']
            top_k = sub_dumper_config.get('top_k', 1)
            resumable = sub_dumper_config.get('resumable', False)
            epoch_selector = build_checkpoint_scheduler(sub_dumper_config.get('epoch_trigger', None))
            drop_last = sub_dumper_config.get('drop_last', True)
            exclude_frozen_parameters = sub_dumper_config.get('exclude_frozen_parameters', True)

            from . import BestCheckpointDumper
            metric_based_dumpers.append(BestCheckpointDumper(output_path, metric_name, top_k, epoch_selector,
                                                             drop_last, resumable, exclude_frozen_parameters))
        elif sub_dumper_config['type'] == 'latest':
            resumable = sub_dumper_config.get('resumable', False)
            exclude_frozen_parameters = sub_dumper_config.get('exclude_frozen_parameters', True)

            from . import LatestCheckpointDumper
            epoch_based_dumpers.append(LatestCheckpointDumper(output_path, resumable, exclude_frozen_parameters))
        else:
            raise NotImplementedError(f"Unknown checkpoint dumper type: {sub_dumper_config['type']}")
    return CheckpointDumper(output_path, epoch_based_dumpers, step_based_dumpers, metric_based_dumpers)
