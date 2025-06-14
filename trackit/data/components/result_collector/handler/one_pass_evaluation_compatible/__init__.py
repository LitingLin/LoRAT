import os.path
from typing import Optional, Sequence, Mapping, Tuple, MutableMapping

import numpy as np

from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from ..utils.writer import FolderWriter, ZipfileWriter
from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler
from .ope_metrics import OPEMetrics, DatasetOPEResultsBuilder, DatasetOPEResults, compute_OPE_metrics_mean, compute_one_pass_evaluation_metrics
from .report_gen import generate_dataset_one_pass_evaluation_report, generate_one_pass_evaluation_summary_report, \
    dump_sequence_tracking_results_with_groundtruth, generate_sequence_one_pass_evaluation_report
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


class EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_path: Optional[str], file_name: Optional[str], rasterize_bbox: bool):
        self._compliance = 'STARK'  # todo: options to match with evaluation results on pytracking, LaSOT toolkit, OTB toolkit
        self._tracker_name = tracker_name
        self._folder_writer = None
        if output_path is not None and file_name is not None:
            self._folder_writer = ZipfileWriter(os.path.join(output_path, file_name + '.zip'))

        self._progress_aware_sub_handler = EvaluationResultPersistenceWithOPEMetrics_ProgressAware(rasterize_bbox)
        self._live_feed_sub_handler = EvaluationResultPersistenceWithOPEMetrics_LiveFeed(rasterize_bbox)
        self._collected_metrics = []
        self._is_closed = False

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT],
               evaluation_progresses: Sequence[EvaluationProgress]):
        assert not self._is_closed
        metrics = {}
        sub_handler_1_metrics = self._progress_aware_sub_handler(self._tracker_name, self._folder_writer, evaluation_results, evaluation_progresses)
        sub_handler_2_metrics = self._live_feed_sub_handler(self._tracker_name, self._folder_writer, evaluation_results, evaluation_progresses)
        if sub_handler_1_metrics is not None:
            metrics.update(sub_handler_1_metrics)
        if sub_handler_2_metrics is not None:
            metrics.update(sub_handler_2_metrics)
        if len(metrics) != 0:
            self._collected_metrics.extend(list(metrics.items()))

    def close(self):
        assert not self._is_closed
        metrics = {}
        sub_handler_1_metrics = self._progress_aware_sub_handler.finalize(self._tracker_name, self._folder_writer)
        sub_handler_2_metrics = self._live_feed_sub_handler.finalize(self._tracker_name, self._folder_writer)
        if sub_handler_1_metrics is not None:
            metrics.update(sub_handler_1_metrics)
        if sub_handler_2_metrics is not None:
            metrics.update(sub_handler_2_metrics)
        if len(metrics) != 0:
            self._collected_metrics.extend(list(metrics.items()))
        if self._folder_writer is not None:
            self._folder_writer.close()
            del self._folder_writer
        self._is_closed = True

    def get_metrics(self) -> Optional[Sequence[Tuple[str, float]]]:
        return self._collected_metrics


class FinalOPEMetricsSummaryReportGenerator:
    def __init__(self):
        self._final_summary_metrics = {}

    def add(self, dataset_name: str, repeat_index: Optional[int], metrics: OPEMetrics):
        if repeat_index not in self._final_summary_metrics:
            self._final_summary_metrics[repeat_index] = {}
        this_repeat_summary_metrics = self._final_summary_metrics[repeat_index]
        assert dataset_name not in this_repeat_summary_metrics
        this_repeat_summary_metrics[dataset_name] = metrics

    def dump(self, folder_writer: FolderWriter, tracker_name: str):
        for repeat_index, this_repeat_summary_metrics in self._final_summary_metrics.items():
            sorted_metrics = dict(sorted(this_repeat_summary_metrics.items()))
            generate_one_pass_evaluation_summary_report(folder_writer, tracker_name, repeat_index, sorted_metrics)


def __get_summary_metric_name(metric_name: str, dataset_name: str, repeat_index: Optional[int]):
    summary_metric_name = f'{metric_name}_{dataset_name}'
    if repeat_index is not None:
        summary_metric_name += f'_{repeat_index:03d}'
    return summary_metric_name


def _generate_dataset_summary_metrics_name_value_pair(dataset_name: str, repeat_index: Optional[int], metrics: OPEMetrics):
    return {
        __get_summary_metric_name('success_score', dataset_name, repeat_index): metrics.success_score,
        __get_summary_metric_name('precision_score', dataset_name, repeat_index): metrics.precision_score,
        __get_summary_metric_name('norm_precision_score', dataset_name, repeat_index): metrics.normalized_precision_score,
        __get_summary_metric_name('success_rate_at_overlap_0_5', dataset_name, repeat_index): metrics.success_rate_at_overlap_0_5,
        __get_summary_metric_name('success_rate_at_overlap_0_75', dataset_name, repeat_index): metrics.success_rate_at_overlap_0_75,
        __get_summary_metric_name('fps', dataset_name, repeat_index): metrics.get_fps(),
    }


class EvaluationResultPersistenceWithOPEMetrics_ProgressAware:
    def __init__(self, rasterize_bbox: bool):
        self._known_tracks_metric_cache = {}
        self._multi_run_dataset_metrics_cache = {}
        self._final_summary_report_generator = FinalOPEMetricsSummaryReportGenerator()
        self._compatibility_helper = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def __call__(self, tracker_name: str, folder_writer: Optional[FolderWriter],
                 evaluation_results: Sequence[SequenceEvaluationResult_SOT],
                 evaluation_progresses: Sequence[EvaluationProgress]) -> Optional[Mapping[str, float]]:
        summary_metrics = {}

        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is None:
                continue

            assert evaluation_result.sequence_info.dataset_name is not None
            assert evaluation_result.sequence_info.dataset_full_name is not None
            assert evaluation_result.sequence_info.sequence_name is not None
            predicted_target_bounding_boxes = evaluation_result.output_box
            if self._rasterize_bbox:
                predicted_target_bounding_boxes = bbox_rasterize(predicted_target_bounding_boxes)

            metrics, frames_iou = compute_one_pass_evaluation_metrics(
                evaluation_result.sequence_info.dataset_name,
                predicted_target_bounding_boxes,
                evaluation_result.groundtruth_box,
                evaluation_result.groundtruth_object_existence_flag,
                evaluation_result.time_cost,
                self._compatibility_helper)

            print(f'{evaluation_result.sequence_info.sequence_name}: success {metrics.success_score:.04f}, prec {metrics.precision_score:.04f}, norm_pre {metrics.normalized_precision_score:.04f}')

            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset.total_repeat_times == 1:
                repeat_index = None
            cache_key = evaluation_result.sequence_info.dataset_full_name, repeat_index
            if cache_key not in self._known_tracks_metric_cache:
                dataset_ope_results_builder = DatasetOPEResultsBuilder()
                self._known_tracks_metric_cache[cache_key] = dataset_ope_results_builder
            else:
                dataset_ope_results_builder = self._known_tracks_metric_cache[cache_key]
            dataset_ope_results_builder.add_sequence(evaluation_result.sequence_info.sequence_name, metrics)
            if folder_writer is not None:
                dump_sequence_tracking_results_with_groundtruth(folder_writer,
                                                                tracker_name, repeat_index,
                                                                evaluation_result.sequence_info.dataset_full_name,
                                                                evaluation_result.sequence_info.sequence_name,
                                                                evaluation_result.evaluated_frame_indices,
                                                                evaluation_result.output_confidence,
                                                                predicted_target_bounding_boxes,
                                                                evaluation_result.groundtruth_object_existence_flag,
                                                                evaluation_result.groundtruth_box,
                                                                evaluation_result.time_cost,
                                                                frames_iou)
                generate_sequence_one_pass_evaluation_report(folder_writer,
                                                             tracker_name, repeat_index,
                                                             evaluation_result.sequence_info.dataset_full_name,
                                                             evaluation_result.sequence_info.sequence_name,
                                                             metrics)
            if evaluation_progress.this_dataset.this_repeat_all_evaluated:
                dataset_ope_results = dataset_ope_results_builder.build()
                dataset_ope_results = dataset_ope_results.sort_by_sequence_name()

                del self._known_tracks_metric_cache[cache_key]

                dataset_name = evaluation_result.sequence_info.dataset_name
                dataset_split = evaluation_result.sequence_info.data_split
                dataset_full_name = evaluation_result.sequence_info.dataset_full_name

                filtered_dataset_ope_results = self._filter_dataset_ope_results_through_dataset_attributes(
                    dataset_name, dataset_split, dataset_full_name, dataset_ope_results
                )

                for (current_dataset_name, current_dataset_ope_results,
                     should_dump_full_results, should_report_to_summary_metrics) in filtered_dataset_ope_results:
                    current_dataset_summary_metrics = current_dataset_ope_results.get_mean()
                    self._dump_dataset_ope_result(tracker_name, current_dataset_name, repeat_index,
                                                  current_dataset_ope_results, current_dataset_summary_metrics,
                                                  folder_writer,
                                                  should_dump_full_results,
                                                  True,
                                                  summary_metrics if should_report_to_summary_metrics else None)

                    if evaluation_progress.this_dataset.total_repeat_times > 1:
                        if current_dataset_name not in self._multi_run_dataset_metrics_cache:
                            dataset_all_runs_metrics = []
                            self._multi_run_dataset_metrics_cache[current_dataset_name] = dataset_all_runs_metrics
                        else:
                            dataset_all_runs_metrics = self._multi_run_dataset_metrics_cache[current_dataset_name]
                        dataset_all_runs_metrics.append(current_dataset_summary_metrics)

                        if evaluation_progress.this_dataset.all_evaluated:
                            dataset_all_runs_mean_metrics = compute_OPE_metrics_mean(dataset_all_runs_metrics)
                            del self._multi_run_dataset_metrics_cache[current_dataset_name]

                            self._dump_dataset_ope_result(tracker_name, current_dataset_name, None,
                                                          None, dataset_all_runs_mean_metrics,
                                                          folder_writer,
                                                          False,
                                                          True,
                                                          summary_metrics if should_report_to_summary_metrics else None)

        return summary_metrics

    @staticmethod
    def _filter_dataset_ope_results_through_dataset_attributes(dataset_name, data_split, dataset_full_name,
                                                               dataset_ope_results: DatasetOPEResults):
        filtered_dataset_ope_results = [(dataset_full_name, dataset_ope_results, True, True)]
        dataset_attributes = None
        if dataset_name == 'LaSOT':
            from trackit.datasets.SOT.datasets.LaSOT import get_LaSOT_sequence_attributes
            dataset_attributes = get_LaSOT_sequence_attributes()
        elif dataset_name == 'LaSOT_Extension':
            from trackit.datasets.SOT.datasets.LaSOT_Extension import get_LaSOT_extension_sequence_attributes
            dataset_attributes = get_LaSOT_extension_sequence_attributes()
        elif dataset_name == 'VastTrack' and data_split is not None and data_split[0] == 'test':
            from trackit.datasets.SOT.datasets.VastTrack import get_VastTrack_test_set_attributes
            dataset_attributes = get_VastTrack_test_set_attributes()

        if dataset_attributes is not None:
            dataset_attribute_masks = tuple(dataset_attributes.SEQUENCE_ATTRIBUTES[sequence_name]
                                            for sequence_name in dataset_ope_results.sequence_name)
            dataset_attribute_masks = np.array(dataset_attribute_masks, dtype=bool)
            for index_of_attribute, attribute_short_name in enumerate(dataset_attributes.ATTRIBUTE_SHORT_NAMES):
                filtered_dataset_ope_results.append((dataset_full_name + '_' + attribute_short_name,
                                                     dataset_ope_results.filter_sequences(
                                                         dataset_attribute_masks[:, index_of_attribute]),
                                                     False, False))
        return filtered_dataset_ope_results

    def _dump_dataset_ope_result(self,
                                 tracker_name, dataset_name, repeat_index,
                                 dataset_ope_results, dataset_summary_metrics,
                                 folder_writer,
                                 dump_full_results: bool, add_to_final_summary_report: bool, metrics_to_report: dict):
        if folder_writer is not None and dump_full_results:
            generate_dataset_one_pass_evaluation_report(folder_writer,
                                                        tracker_name, repeat_index,
                                                        dataset_name,
                                                        dataset_ope_results,
                                                        dataset_summary_metrics)
        if add_to_final_summary_report:
            self._final_summary_report_generator.add(dataset_name, repeat_index, dataset_summary_metrics)
        if metrics_to_report is not None:
            metrics_to_report.update(
                _generate_dataset_summary_metrics_name_value_pair(dataset_name,
                                                                  repeat_index,
                                                                  dataset_summary_metrics))

    def finalize(self, tracker_name: str, folder_writer: Optional[FolderWriter]) -> Optional[Mapping[str, float]]:
        if folder_writer is not None:
            self._final_summary_report_generator.dump(folder_writer, tracker_name)

        return None


class EvaluationResultPersistenceWithOPEMetrics_LiveFeed:
    def __init__(self, rasterize_bbox: bool):
        self._metric_cache: MutableMapping[Tuple[str, int], DatasetOPEResultsBuilder] = {}
        self._compatibility_helper = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def __call__(self,
                 tracker_name: str, folder_writer: Optional[FolderWriter],
                 evaluation_results: Sequence[SequenceEvaluationResult_SOT],
                 evaluation_progresses: Sequence[EvaluationProgress]) -> Optional[Mapping[str, float]]:
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is not None:
                continue
            assert evaluation_result.sequence_info.dataset_full_name is not None
            assert evaluation_result.sequence_info.sequence_name is not None
            predicted_target_bounding_boxes = evaluation_result.output_box
            metrics, frames_iou = \
                compute_one_pass_evaluation_metrics(evaluation_result.sequence_info.dataset_name,
                                                    predicted_target_bounding_boxes,
                                                    evaluation_result.groundtruth_box,
                                                    evaluation_result.groundtruth_object_existence_flag,
                                                    evaluation_result.time_cost,
                                                    self._compatibility_helper)

            if folder_writer is not None:
                dump_sequence_tracking_results_with_groundtruth(folder_writer,
                                                                tracker_name, evaluation_progress.repeat_index,
                                                                evaluation_result.sequence_info.dataset_full_name,
                                                                evaluation_result.sequence_info.sequence_name,
                                                                evaluation_result.evaluated_frame_indices,
                                                                evaluation_result.output_confidence,
                                                                predicted_target_bounding_boxes,
                                                                evaluation_result.groundtruth_box,
                                                                evaluation_result.groundtruth_object_existence_flag,
                                                                evaluation_result.time_cost,
                                                                frames_iou)
                generate_sequence_one_pass_evaluation_report(folder_writer,
                                                             tracker_name, evaluation_progress.repeat_index,
                                                             evaluation_result.sequence_info.dataset_full_name,
                                                             evaluation_result.sequence_info.sequence_name,
                                                             metrics)
            metric_cache_key = evaluation_result.sequence_info.dataset_full_name, evaluation_progress.repeat_index
            if metric_cache_key not in self._metric_cache:
                self._metric_cache[metric_cache_key] = DatasetOPEResultsBuilder()
            self._metric_cache[metric_cache_key].add_sequence(evaluation_result.sequence_info.sequence_name, metrics)
        return None

    def finalize(self, tracker_name: str, folder_writer: Optional[FolderWriter]) -> Optional[Mapping[str, float]]:
        summary_metrics = {}

        all_dataset_summary_metrics = {}

        for (dataset_full_name, repeat_index), metrics_list_builder in self._metric_cache.items():
            metrics_list = metrics_list_builder.build()
            dataset_summary_metrics = metrics_list.get_mean()
            if folder_writer is not None:
                generate_dataset_one_pass_evaluation_report(folder_writer, tracker_name, repeat_index,
                                                            dataset_full_name, metrics_list.sort_by_sequence_name(),
                                                            dataset_summary_metrics)
            if dataset_full_name not in all_dataset_summary_metrics:
                all_dataset_summary_metrics[dataset_full_name] = []
            all_dataset_summary_metrics[dataset_full_name].append(dataset_summary_metrics)

        final_summary_report_generator = FinalOPEMetricsSummaryReportGenerator()

        for dataset_full_name, dataset_summary_metrics_list in all_dataset_summary_metrics.items():
            for repeat_index, dataset_summary_metrics in enumerate(dataset_summary_metrics_list):
                summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(dataset_full_name, repeat_index, dataset_summary_metrics))
                final_summary_report_generator.add(dataset_full_name, repeat_index, dataset_summary_metrics)
            if len(dataset_summary_metrics_list) > 1:
                dataset_multirun_averaged_metrics = compute_OPE_metrics_mean(dataset_summary_metrics_list)
            else:
                dataset_multirun_averaged_metrics = dataset_summary_metrics_list[0]
            summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(dataset_full_name, None, dataset_multirun_averaged_metrics))
            final_summary_report_generator.add(dataset_full_name, None, dataset_multirun_averaged_metrics)

        if folder_writer is not None:
            final_summary_report_generator.dump(folder_writer, tracker_name)

        return summary_metrics
