import os
import io
import zipfile
from typing import Optional, Sequence
import numpy as np

from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT

from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler


class TrackingNetResultFileWriter:
    def __init__(self, output_folder: str, output_file_name: str,):
        self._zip_file_path_prefix = os.path.join(output_folder, output_file_name)
        self._zip_files = {}
        self._duplication_check = {}

    def write(self, tracker_name: str, repeat_index: Optional[int],
              sequence_name: str, predicted_bboxes: np.ndarray):
        with io.BytesIO() as result_file_content:
            np.savetxt(result_file_content, predicted_bboxes, fmt='%.2f', delimiter=',')

            if repeat_index not in self._zip_files:
                if repeat_index is None:
                    zip_file_path = self._zip_file_path_prefix + '.zip'
                else:
                    zip_file_path = self._zip_file_path_prefix + f'_{repeat_index + 1:03d}.zip'
                self._zip_files[repeat_index] = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)
                self._duplication_check[repeat_index] = set()
            zip_file = self._zip_files[repeat_index]
            assert (tracker_name, sequence_name) not in self._duplication_check[repeat_index], "duplicated sequence name detected"
            self._duplication_check[repeat_index].add((tracker_name, sequence_name))
            zip_file.writestr('/'.join((tracker_name, sequence_name, f'{sequence_name}.txt')),
                              result_file_content.getvalue())

    def close(self):
        for zip_file in self._zip_files.values():
            zip_file.close()


class TrackingNetEvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_folder: str, output_file_name: str, rasterize_bbox: bool):
        self._writer = TrackingNetResultFileWriter(output_folder, output_file_name)
        self._tracker_name = tracker_name
        self._rasterize_bbox = rasterize_bbox

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT], evaluation_progresses: Sequence[EvaluationProgress]):
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset is not None:
                repeat_index = repeat_index if evaluation_progress.this_dataset.total_repeat_times > 1 else None

            predicted_bboxes = evaluation_result.output_box
            if self._rasterize_bbox:
                predicted_bboxes = bbox_rasterize(predicted_bboxes)
            predicted_bboxes = bbox_xyxy_to_xywh(predicted_bboxes)

            self._writer.write(self._tracker_name, repeat_index,
                               evaluation_result.sequence_info.sequence_name,
                               predicted_bboxes)

    def close(self):
        self._writer.close()
