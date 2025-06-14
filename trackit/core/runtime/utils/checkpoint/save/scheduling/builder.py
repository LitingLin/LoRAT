from typing import Optional, Sequence
from . import EpochOrStepSelector, SpecificValuesSelector, RangeSelector, PeriodicSelector, UnrestrictedSelector, CompositeSelector


def build_checkpoint_scheduler(config: Optional[dict]) -> EpochOrStepSelector:
    if config is None:
        return UnrestrictedSelector()
    selectors = []
    interval = config.get('interval')
    if interval:
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("'interval' must be a positive integer.")
        selectors.append(PeriodicSelector(interval))

    slice_params = config.get('slice')
    if slice_params:
        if not (isinstance(slice_params, Sequence) and all(isinstance(v, int) for v in slice_params)):
            raise ValueError("'slice' config must be a sequence of integers.")
        if len(slice_params) == 0 or len(slice_params) > 3:
            raise ValueError("Invalid 'slice' config. Expected format is [start,] stop[, step].")
        if len(slice_params) == 1:
            selector = RangeSelector(end=slice_params[0])
        else:
            selector = RangeSelector(*slice_params)
        selectors.append(selector)

    custom_values = config.get('values')
    if custom_values:
        if not (isinstance(custom_values, Sequence) and all(isinstance(v, int) for v in custom_values)):
            raise ValueError("'values' must be a sequence of integers.")
        selectors.append(SpecificValuesSelector(custom_values))

    return CompositeSelector(selectors)
