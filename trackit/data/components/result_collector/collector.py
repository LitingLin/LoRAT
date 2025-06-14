from typing import Optional, Mapping, Sequence, Tuple
import itertools
from dataclasses import dataclass
from tabulate import tabulate

from trackit.miscellanies.torch.distributed import is_rank_0_process
from trackit.core.runtime.metric_logger import get_current_metric_logger, get_current_local_metric_logger
from trackit.core.runtime.context.epoch import get_current_epoch_context
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.data.utils.data_source_matcher import DataSourceMatcher
from trackit.data import MainProcessDataPipeline
from trackit.data.context.variable.eval import DatasetEvaluationTask

from .progress_tracer.predefined import EvaluationTaskTracer_Predefined
from .progress_tracer.plain import EvaluationTaskTracer_Plain
from .handler import EvaluationResultHandlerAsyncWrapper
from .progress_tracer import EvaluationTaskTracer


def print_metrics(metrics: Sequence[Tuple[str, float]], log_as_general_summary: bool):
    if len(metrics) == 0:
        return
    metrics = dict(metrics)
    if log_as_general_summary:
        get_current_metric_logger().log_summary(metrics)
    else:
        get_current_local_metric_logger().log_summary(metrics)
        get_current_metric_logger().log(external=metrics)
    print(tabulate(metrics.items(), headers=('metric', 'value'), floatfmt=".4f"), flush=True)


class EvaluationResultCollector:
    def __init__(self, progress_tracer: EvaluationTaskTracer,
                 dispatcher: Mapping[DataSourceMatcher, Sequence[EvaluationResultHandlerAsyncWrapper]],
                 log_summary: bool):
        self._progress_tracer = progress_tracer
        self._dispatcher = dispatcher
        self._log_summary = log_summary

    def collect(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT]):
        progresses = tuple(self._progress_tracer.submit(evaluation_result.sequence_info.dataset_full_name, evaluation_result.sequence_info.sequence_name)
                           for evaluation_result in evaluation_results)

        last_remaining_evaluation_results = evaluation_results
        last_remaining_progresses = progresses

        rest_evaluation_results = []
        rest_progresses = []

        for data_source_matcher, handlers in self._dispatcher.items():
            this_handler_evaluation_results = []
            this_handler_evaluation_progresses = []
            for evaluation_result, progress in zip(last_remaining_evaluation_results, last_remaining_progresses):
                if data_source_matcher(evaluation_result.sequence_info.dataset_name, evaluation_result.sequence_info.data_split):
                    this_handler_evaluation_results.append(evaluation_result)
                    this_handler_evaluation_progresses.append(progress)
                else:
                    rest_evaluation_results.append(evaluation_result)
                    rest_progresses.append(progress)

            if len(this_handler_evaluation_results) > 0:
                for handler in handlers:
                    handler.accept(this_handler_evaluation_results, this_handler_evaluation_progresses)
            last_remaining_evaluation_results = rest_evaluation_results
            last_remaining_progresses = rest_progresses
            rest_evaluation_results = []
            rest_progresses = []

        assert len(last_remaining_evaluation_results) == 0, "Some evaluation results are not handled."

    def get_metrics(self) -> Sequence[Tuple[str, float]]:
        metrics = []
        for handlers in self._dispatcher.values():
            for handler in handlers:
                this_handler_metrics = handler.get_unseen_metrics()
                if this_handler_metrics is not None:
                    metrics.extend(this_handler_metrics)
        return metrics

    def finalize(self):
        for handlers in self._dispatcher.values():
            for handler in handlers:
                handler.close()


@dataclass(frozen=True)
class SubHandlerBuildOptions:
    handler_type: str
    file_name: Optional[str]
    bbox_rasterize: bool


class EvaluationResultCollector_RuntimeIntegration(MainProcessDataPipeline):
    def __init__(self, tracker_name: str,
                 handler_build_options: Mapping[DataSourceMatcher, Sequence[SubHandlerBuildOptions]],
                 optional_predefined_evaluation_tasks: Optional[Sequence[DatasetEvaluationTask]],
                 run_async: bool, log_summary: bool):
        tracker_name = tracker_name.replace('/', '-')
        tracker_name = tracker_name.replace('\\', '-')
        if is_rank_0_process():
            self._predefined_evaluation_tasks = optional_predefined_evaluation_tasks
            self._tracker_name = tracker_name
            self._handler_build_options = handler_build_options
            self._run_async = run_async
            self._log_summary = log_summary

    def on_epoch_begin(self):
        if is_rank_0_process():
            output_path = get_current_epoch_context().get_current_epoch_output_path()
            if self._predefined_evaluation_tasks is not None:
                progress_tracer = EvaluationTaskTracer_Predefined(self._predefined_evaluation_tasks)
            else:
                progress_tracer = EvaluationTaskTracer_Plain()
            dispatcher = {}
            for data_source_matcher, handler_build_options in self._handler_build_options.items():
                handlers = []
                for handler_build_option in handler_build_options:
                    handler_cls = None
                    if handler_build_option.handler_type == 'plain':
                        if output_path is not None or handler_build_option.file_name is not None:
                            from .handler.persistence import EvaluationResultPersistence
                            handler_cls = EvaluationResultPersistence
                    elif handler_build_option.handler_type == 'one_pass_evaluation':
                        from .handler.one_pass_evaluation import EvaluationResultPersistenceWithOPEMetrics
                        handler_cls = EvaluationResultPersistenceWithOPEMetrics
                    elif handler_build_option.handler_type == 'one_pass_evaluation_compatible':
                        from .handler.one_pass_evaluation_compatible import EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools
                        handler_cls = EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools
                    elif handler_build_option.handler_type == 'external/GOT10k':
                        if output_path is not None:
                            from .handler.external_adaptors.got10k import GOT10KEvaluationToolAdaptor
                            handler_cls = GOT10KEvaluationToolAdaptor
                    elif handler_build_option.handler_type == 'external/OTB':
                        if output_path is not None:
                            from .handler.external_adaptors.otb import OTBEvaluationToolAdaptor
                            handler_cls = OTBEvaluationToolAdaptor
                    elif handler_build_option.handler_type == 'external/PyTracking':
                        if output_path is not None:
                            from .handler.external_adaptors.pytracking import PyTrackingEvaluationToolAdaptor
                            handler_cls = PyTrackingEvaluationToolAdaptor
                    elif handler_build_option.handler_type == 'external/TrackingNet':
                        if output_path is not None:
                            from .handler.external_adaptors.trackingnet import TrackingNetEvaluationToolAdaptor
                            handler_cls = TrackingNetEvaluationToolAdaptor
                    else:
                        raise ValueError(f'Unknown handler type: {handler_build_option.handler_type}')
                    if handler_cls is not None:
                        handler = EvaluationResultHandlerAsyncWrapper(handler_cls,
                                                                      (self._tracker_name,
                                                                       output_path,
                                                                       handler_build_option.file_name,
                                                                       handler_build_option.bbox_rasterize),
                                                                      self._run_async)
                        handlers.append(handler)
                dispatcher[data_source_matcher] = handlers
            self._collector = EvaluationResultCollector(progress_tracer, dispatcher, self._log_summary)
            self._duplication_check = set()

        self._local_cache = []

    def post_process(self, output_data: Optional[dict]):
        if output_data is not None:
            self._local_cache.extend(output_data['evaluated_sequences'])
        if is_rank_0_process():
            print_metrics(self._collector.get_metrics(), self._log_summary)
        return output_data

    def distributed_prepare_gathering(self) -> Sequence[SequenceEvaluationResult_SOT]:
        cached_sequences = self._local_cache
        self._local_cache = []
        return cached_sequences

    def distributed_on_gathered(self, evaluation_results_on_all_nodes: Sequence[Sequence[SequenceEvaluationResult_SOT]]) -> None:
        if is_rank_0_process():
            evaluation_results = tuple(itertools.chain.from_iterable(evaluation_results_on_all_nodes))
            if len(evaluation_results) > 0:
                for evaluation_result in evaluation_results:
                    assert evaluation_result.id not in self._duplication_check
                    self._duplication_check.add(evaluation_result.id)

                self._collector.collect(evaluation_results)

    def on_epoch_end(self):
        assert len(self._local_cache) == 0
        if is_rank_0_process():
            self._collector.finalize()
            print_metrics(self._collector.get_metrics(), self._log_summary)
            del self._collector
            del self._duplication_check
