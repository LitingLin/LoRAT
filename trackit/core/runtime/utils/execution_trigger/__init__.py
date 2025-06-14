class ExecutionCriterion:
    def should_execute(self, value: int) -> bool:
        raise NotImplementedError("Method __call__ must be implemented in subclass.")

    def total_executions(self) -> int:
        raise NotImplementedError("Method __len__ must be implemented in subclass.")


class ExecutionCriterion_SpecificValues(ExecutionCriterion):
    def __init__(self, values):
        self.values = values

    def should_execute(self, value: int) -> bool:
        return value in self.values

    def total_executions(self) -> int:
        return len(self.values)


class ExecutionCriterion_Unrestricted(ExecutionCriterion):
    def __init__(self, total: int):
        if not isinstance(total, int) or total <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        self.total = total

    def should_execute(self, value: int) -> bool:
        return 0 <= value < self.total

    def total_executions(self) -> int:
        return self.total
