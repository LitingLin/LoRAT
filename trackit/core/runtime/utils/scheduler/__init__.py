class Scheduler:
    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def on_iteration_begin(self, is_train: bool):
        pass

    def on_iteration_end(self, is_train: bool):
        pass

    def on_epoch_begin(self, epoch: int, is_train: bool):
        pass

    def on_epoch_end(self, epoch: int, is_train: bool):
        pass

    def current(self):
        raise NotImplementedError()
