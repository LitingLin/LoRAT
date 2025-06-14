import math
from typing import Sequence, Optional, Iterable
from . import ExecutionCriterion, ExecutionCriterion_Unrestricted, ExecutionCriterion_SpecificValues


def build_execution_trigger(config: Optional[dict], total: int) -> ExecutionCriterion:
    if config is None:
        return ExecutionCriterion_Unrestricted(total)

    if not isinstance(total, int) or total <= 0:
        raise ValueError("total_epochs must be a positive integer.")

    specified_epochs = set()

    if config.get('last'):
        specified_epochs.add(total - 1)

    interval = config.get('interval')
    if interval:
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("'interval' must be a positive integer.")
        specified_epochs.update(range(interval - 1, total, interval))

    epoch_slice = config.get('slice')
    if epoch_slice:
        if not (isinstance(epoch_slice, Sequence) and all(isinstance(v, int) for v in epoch_slice)):
            raise ValueError("'slice' config must be a sequence of integers.")
        if len(epoch_slice) > 3:
            raise ValueError("Invalid 'slice' config. Expected format is [start,] stop[, step].")
        specified_epochs.update(range(total)[slice(*epoch_slice)])

    epoch_range = config.get('range')
    if epoch_range:
        specified_epochs.update(range(total)[epoch_range.get('start', None):
                                             epoch_range.get('stop', None):
                                             epoch_range.get('step', None)])

    range_relative_config = config.get('range_relative')
    if range_relative_config:
        specified_epochs.update(range(total)[int(math.floor(range_relative_config['start'] * total)) if 'start' in range_relative_config else None:
                                             int(math.ceil(range_relative_config['stop'] * total)) if 'stop' in range_relative_config else None:
                                             int(round(range_relative_config['step'] * total)) if 'step' in range_relative_config else None])

    custom_epochs = config.get('values')
    if custom_epochs:
        if not (isinstance(custom_epochs, Iterable) and all(isinstance(v, int) for v in custom_epochs)):
            raise ValueError("'values' must be a sequence of integers.")
        specified_epochs.update(custom_epochs)

    return ExecutionCriterion_SpecificValues(specified_epochs)
