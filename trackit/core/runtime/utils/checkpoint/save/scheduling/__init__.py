from typing import Optional, Sequence


class EpochOrStepSelector:
    def __call__(self, epoch_or_step: int) -> bool:
        """Evaluate whether the given epoch or step is selected."""
        raise NotImplementedError("Subclasses must implement this method.")


class SpecificValuesSelector(EpochOrStepSelector):
    def __init__(self, target_values: Sequence[int]):
        """Initialize with specific target values."""
        self.target_values = set(target_values)

    def __call__(self, epoch_or_step: int) -> bool:
        """Check if the epoch_or_step is in the target values."""
        return epoch_or_step in self.target_values


class RangeSelector(EpochOrStepSelector):
    def __init__(self, start: Optional[int], end: int, step: Optional[int]):
        """Initialize with a start, end, and step."""
        self.start = start
        self.end = end
        self.step = step

    def __call__(self, epoch_or_step: int) -> bool:
        """Check if the epoch_or_step is within the specified range."""
        start = self.start if self.start is not None else 0
        end = self.end
        step = self.step if self.step is not None else 1

        # Check if the epoch_or_step is within the range
        if start <= epoch_or_step < end:
            return (epoch_or_step - start) % step == 0
        return False


class PeriodicSelector(EpochOrStepSelector):
    def __init__(self, interval: int, offset: int = -1):
        """Initialize with an interval and an optional offset."""
        self.interval = interval
        self.offset = offset

    def __call__(self, epoch_or_step: int) -> bool:
        """Check if the epoch_or_step matches the periodic interval."""
        return epoch_or_step >= self.offset and (epoch_or_step - self.offset) % self.interval == 0


class UnrestrictedSelector(EpochOrStepSelector):
    def __init__(self, max_value: Optional[int] = None):
        """Initialize with an optional maximum value."""
        self.max_value = max_value

    def __call__(self, epoch_or_step: int) -> bool:
        """Allow any epoch or step up to the max_value."""
        return self.max_value is None or epoch_or_step < self.max_value


class CompositeSelector(EpochOrStepSelector):
    def __init__(self, selectors: Sequence[EpochOrStepSelector]):
        """Initialize with a sequence of selectors."""
        self.selectors = selectors

    def __call__(self, epoch_or_step: int) -> bool:
        """Evaluate if any of the selectors match the epoch_or_step."""
        return any(selector(epoch_or_step) for selector in self.selectors)
