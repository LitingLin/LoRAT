from . import Scheduler


class LinearScheduler(Scheduler):
    def __init__(self, initial_value, ultimate_value, begin_step, end_step, per_iteration=True):
        self.initial_value = initial_value
        self.ultimate_value = ultimate_value
        self.begin_step = begin_step
        self.end_step = end_step
        self.step_size = (ultimate_value - initial_value) / (end_step - begin_step)
        self.per_iteration = per_iteration
        self.position = 0

    def get_state(self):
        return self.position

    def set_state(self, state):
        self.position = state

    def on_epoch_begin(self, epoch: int, is_train: bool):
        if is_train and not self.per_iteration:
            self.position = epoch

    def on_iteration_end(self, is_train: bool):
        if is_train and self.per_iteration:
            self.position += 1

    def current(self):
        if self.position < self.begin_step:
            value = self.initial_value
        elif self.position > self.end_step:
            value = self.ultimate_value
        else:
            value = self.initial_value + (self.position - self.begin_step) * self.step_size
        return value
