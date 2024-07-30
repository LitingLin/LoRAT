class EpochActivationCriterion:
    def __call__(self, epoch: int) -> bool:
        raise NotImplementedError("Method __call__ must be implemented in subclass.")

    def __len__(self) -> int:
        raise NotImplementedError("Method __len__ must be implemented in subclass.")


class FixedEpochActivation(EpochActivationCriterion):
    """
    Activates on specified epochs.
    """
    def __init__(self, allowed_epochs):
        self.allowed_epochs = allowed_epochs

    def __call__(self, epoch: int) -> bool:
        return epoch in self.allowed_epochs

    def __len__(self) -> int:
        return len(self.allowed_epochs)


class UnrestrictedEpochActivation(EpochActivationCriterion):
    """
    Activates at all epochs up to a specified limit.
    """
    def __init__(self, max_epochs: int):
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        self.max_epochs = max_epochs

    def __call__(self, epoch: int) -> bool:
        return 0 <= epoch < self.max_epochs

    def __len__(self) -> int:
        return self.max_epochs
