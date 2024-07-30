from . import EvaluationTaskTracer, EvaluationProgress


class EvaluationTaskTracer_Plain(EvaluationTaskTracer):
    def __init__(self):
        self._context = {}

    def submit(self, dataset_full_name: str, track_name: str) -> EvaluationProgress:
        if dataset_full_name not in self._context:
            self._context[dataset_full_name] = {}
        dataset_tracing_context = self._context[dataset_full_name]
        if track_name not in dataset_tracing_context:
            repeat_index = 0
        else:
            repeat_index = dataset_tracing_context[track_name] + 1
        dataset_tracing_context[track_name] = repeat_index
        return EvaluationProgress(repeat_index, None)
