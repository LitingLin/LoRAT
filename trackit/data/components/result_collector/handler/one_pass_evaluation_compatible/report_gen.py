import csv
import numpy as np
import json
import pickle
from typing import Optional, Dict, Tuple
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from ..utils.writer import FolderWriter
from .ope_metrics import DatasetOPEMetricsList, OPEMetrics


def dump_sequence_tracking_results_with_groundtruth(folder_writer: FolderWriter,
                                                    tracker_name: str,
                                                    repeat_index: Optional[int],
                                                    dataset_name: str, sequence_name: str,
                                                    frame_indices: np.ndarray,
                                                    prediction_confidence: np.ndarray,
                                                    predicted_bboxes: np.ndarray,
                                                    groundtruth_object_existence: np.ndarray,
                                                    groundtruth_bounding_boxes: np.ndarray,
                                                    time_costs: np.ndarray,
                                                    iou_of_frames: np.ndarray):
    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

    with folder_writer.open_binary_file_handle((*path, 'eval.pkl')) as f:
        pickle.dump(
            {'frame_index': frame_indices, 'confidence': prediction_confidence, 'bounding_box': predicted_bboxes,
             'time': time_costs}, f)

    sequence_length = len(frame_indices)
    eval_result_matrix = np.empty((sequence_length, 12), dtype=np.float64)
    eval_result_matrix[:, 0] = frame_indices.astype(np.float64)
    eval_result_matrix[:, 1] = groundtruth_object_existence.astype(np.float64)
    eval_result_matrix[:, 2] = prediction_confidence.astype(np.float64)
    eval_result_matrix[:, 3:7] = bbox_xyxy_to_xywh(predicted_bboxes).astype(np.float64)
    eval_result_matrix[:, 7:11] = bbox_xyxy_to_xywh(groundtruth_bounding_boxes).astype(np.float64)
    eval_result_matrix[:, 11] = iou_of_frames.astype(np.float64)

    with folder_writer.open_text_file_handle((*path, 'eval.csv')) as f:
        np.savetxt(f, eval_result_matrix, fmt='%.3f', delimiter=',',
                   header=','.join(('ind', 'gt_obj_exist', 'pred_conf', 'pred_x', 'pred_y', 'pred_w', 'pred_h', 'gt_x',
                                    'gt_y', 'gt_w', 'gt_h', 'iou')))


def generate_sequence_one_pass_evaluation_report(
        folder_writer: FolderWriter, tracker_name: str,
        repeat_index: Optional[int],
        dataset_name: str, sequence_name: str,
        ope_metrics: OPEMetrics):
    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

    sequence_report = {
        'success_score': ope_metrics.success_score,
        'precision_score': ope_metrics.precision_score,
        'normalized_precision_score': ope_metrics.normalized_precision_score,
        'success_rate_at_overlap_0.5': ope_metrics.success_rate_at_overlap_0_5,
        'success_rate_at_overlap_0.75': ope_metrics.success_rate_at_overlap_0_75,
        'success_curve': ope_metrics.success_curve.tolist(),
        'precision_curve': ope_metrics.precision_curve.tolist(),
        'normalized_precision_curve': ope_metrics.normalized_precision_curve.tolist(),
        'fps': ope_metrics.get_fps()
    }
    with folder_writer.open_text_file_handle((*path, 'performance.json')) as f:
        json.dump(sequence_report, f, indent=2)


def generate_dataset_one_pass_evaluation_report(
        folder_writer: FolderWriter, tracker_name: str,
        repeat_index: Optional[int], dataset_name: str,
        all_sequences_ope_metrics: DatasetOPEMetricsList,
        dataset_summary_ope_metrics: Optional[OPEMetrics] = None):
    if dataset_summary_ope_metrics is None:
        dataset_summary_ope_metrics = all_sequences_ope_metrics.get_mean()

    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name)
    with folder_writer.open_text_file_handle((*path, 'sequences_performance.csv')) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('Sequence Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                             'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for sequence_name, ope_metrics in all_sequences_ope_metrics:
            csv_writer.writerow((sequence_name,
                                 ope_metrics.success_score, ope_metrics.precision_score,
                                 ope_metrics.normalized_precision_score,
                                 ope_metrics.success_rate_at_overlap_0_5,
                                 ope_metrics.success_rate_at_overlap_0_75,
                                 ope_metrics.get_fps()))

    # with folder_writer.open_binary_file_handle((*path, 'success_plot.pdf')) as f:
    #     draw_success_plot(np.expand_dims(dataset_summary_ope_metrics.success_curve, axis=0), (tracker_name,), f)
    # with folder_writer.open_binary_file_handle((*path, 'precision_plot.pdf')) as f:
    #     draw_precision_plot(np.expand_dims(dataset_summary_ope_metrics.precision_curve, axis=0), (tracker_name,), f)
    # with folder_writer.open_binary_file_handle((*path, 'norm_precision_plot.pdf')) as f:
    #     draw_normalized_precision_plot(np.expand_dims(dataset_summary_ope_metrics.normalized_precision_curve, axis=0),
    #                                    (tracker_name,), f)

    dataset_report = {'success_score': dataset_summary_ope_metrics.success_score,
                      'precision_score': dataset_summary_ope_metrics.precision_score,
                      'normalized_precision_score': dataset_summary_ope_metrics.normalized_precision_score,
                      'success_rate_at_overlap_0.5': dataset_summary_ope_metrics.success_rate_at_overlap_0_5,
                      'success_rate_at_overlap_0.75': dataset_summary_ope_metrics.success_rate_at_overlap_0_75,
                      'success_curve': dataset_summary_ope_metrics.success_curve.tolist(),
                      'precision_curve': dataset_summary_ope_metrics.precision_curve.tolist(),
                      'normalized_precision_curve': dataset_summary_ope_metrics.normalized_precision_curve.tolist(),
                      'fps': dataset_summary_ope_metrics.get_fps()}
    with folder_writer.open_text_file_handle((*path, 'performance.json')) as f:
        json.dump(dataset_report, f, indent=2)


def generate_one_pass_evaluation_summary_report(folder_writer: FolderWriter, tracker_name: str,
                                                repeat_index: Optional[int],
                                                datasets_summary_ope_metrics: Dict[str, OPEMetrics]):
    with folder_writer.open_text_file_handle(
            (f'{tracker_name}_performance.csv' if repeat_index is None else f'{tracker_name}_{repeat_index:03d}_performance.csv',)) as f:
        writer = csv.writer(f)
        writer.writerow(('Dataset Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                         'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for dataset_name, dataset_summary_ope_metrics in datasets_summary_ope_metrics.items():
            writer.writerow((dataset_name,
                             dataset_summary_ope_metrics.success_score,
                             dataset_summary_ope_metrics.precision_score,
                             dataset_summary_ope_metrics.normalized_precision_score,
                             dataset_summary_ope_metrics.success_rate_at_overlap_0_5,
                             dataset_summary_ope_metrics.success_rate_at_overlap_0_75,
                             dataset_summary_ope_metrics.get_fps()))
