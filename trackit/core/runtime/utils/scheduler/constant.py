from . import Scheduler


class ConstantScheduler(Scheduler):
    def __init__(self, value):
        self.value = value

    def current(self):
        return self.value
