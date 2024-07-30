from typing import Optional, Sequence, Tuple

from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.miscellanies.worker import SimpleWorker

from ..progress_tracer import EvaluationProgress


class EvaluationResultHandler:
    def accept(self, results: Sequence[SequenceEvaluationResult_SOT], traced_progresses: Sequence[EvaluationProgress]):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def get_metrics(self) -> Optional[Sequence[Tuple[str, float]]]:
        return None


class EvaluationResultHandlerAsyncWrapper:
    def __init__(self, handler_class, handler_args, run_async: bool = False):
        self._cls = handler_class
        self._args = handler_args
        self._instance: Optional[EvaluationResultHandler] = None
        self._run_async = run_async
        self._background_worker: Optional[SimpleWorker] = None
        self._metric_seen_index = None

    def accept(self, results: Sequence[SequenceEvaluationResult_SOT], traced_progresses: Sequence[EvaluationProgress]):
        if self._instance is None:
            self._instance = self._cls(*self._args)
            self._metric_seen_index = 0
            if self._run_async:
                self._background_worker = SimpleWorker()
                self._background_worker.start()

        if self._run_async:
            self._background_worker.submit(self._instance.accept, results, traced_progresses)
        else:
            self._instance.accept(results, traced_progresses)

    def close(self):
        if self._instance is not None:
            if self._run_async:
                self._background_worker.close()
            self._instance.close()

    def get_unseen_metrics(self) -> Optional[Sequence[Tuple[str, float]]]:
        if self._instance is None:
            return None
        metrics = self._instance.get_metrics()
        if metrics is None:
            return None
        metrics_length = len(metrics)
        if metrics_length == self._metric_seen_index:
            return None
        unseen_metrics = metrics[self._metric_seen_index: metrics_length]
        self._metric_seen_index = metrics_length
        return unseen_metrics

    def get_all_metrics(self) -> Optional[Sequence[Tuple[str, float]]]:
        if self._instance is None:
            return None
        return self._instance.get_metrics()
