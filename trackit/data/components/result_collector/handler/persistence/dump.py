from typing import Optional
from ..utils.writer import FolderWriter
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
import pickle
import numpy as np


def dump_sequence_evaluation_result(folder_writer: FolderWriter, tracker_name: str,
                                    repeat_index: Optional[int],
                                    dataset_name: str, sequence_name: str,
                                    frame_indices: np.ndarray, prediction_confidences: np.ndarray,
                                    predicted_bboxes: np.ndarray, time_cost_array: np.ndarray):
    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

    assert len(predicted_bboxes) == len(time_cost_array) == len(prediction_confidences)

    with folder_writer.open_text_file_handle((*path, 'eval.pkl')) as f:
        pickle.dump({'frame_indices': frame_indices, 'confidence': prediction_confidences, 'bounding_box': predicted_bboxes, 'time': time_cost_array}, f)

    sequence_length = len(predicted_bboxes)
    eval_result_matrix = np.empty((sequence_length, 6), dtype=np.float64)
    eval_result_matrix[:, 0] = frame_indices.astype(np.float64)
    eval_result_matrix[:, 1] = prediction_confidences.astype(np.float64)
    eval_result_matrix[:, 2:6] = bbox_xyxy_to_xywh(predicted_bboxes).astype(np.float64)

    with folder_writer.open_text_file_handle((*path, 'eval.csv')) as f:
        np.savetxt(f, eval_result_matrix, fmt='%.3f', delimiter=',',
                   header=','.join(('ind', 'pred_conf', 'pred_x', 'pred_y', 'pred_w', 'pred_h')))
