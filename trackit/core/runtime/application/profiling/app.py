class ModelProfilingApplication:
    def __init__(self, profiler, model_manager):
        self.profiler = profiler
        self.model_manager = model_manager

    def run(self):
        if self.profiler is not None:
            self.profiler(self.model_manager)
