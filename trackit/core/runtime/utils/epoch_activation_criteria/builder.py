from typing import Sequence, Optional, Iterable
from . import EpochActivationCriterion, FixedEpochActivation, UnrestrictedEpochActivation


def build_epoch_activation_criteria(config: Optional[dict], total_epochs: int) -> EpochActivationCriterion:
    if config is None:
        return UnrestrictedEpochActivation(total_epochs)

    if not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError("total_epochs must be a positive integer.")

    specified_epochs = set()

    if config.get('last'):
        specified_epochs.add(total_epochs - 1)

    interval = config.get('interval')
    if interval:
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("'interval' must be a positive integer.")
        specified_epochs.update(range(interval - 1, total_epochs, interval))

    epoch_slice = config.get('slice')
    if epoch_slice:
        if not (isinstance(epoch_slice, Sequence) and all(isinstance(v, int) for v in epoch_slice)):
            raise ValueError("'slice' config must be a sequence of integers.")
        if len(epoch_slice) > 3:
            raise ValueError("Invalid 'slice' config. Expected format is [start,] stop[, step].")
        specified_epochs.update(range(total_epochs)[slice(*epoch_slice)])

    custom_epochs = config.get('values')
    if custom_epochs:
        if not (isinstance(custom_epochs, Iterable) and all(isinstance(v, int) for v in custom_epochs)):
            raise ValueError("'values' must be a sequence of integers.")
        specified_epochs.update(custom_epochs)

    return FixedEpochActivation(specified_epochs)
