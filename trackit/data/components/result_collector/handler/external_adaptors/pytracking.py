import os
import io
import zipfile
import numpy as np
from typing import Optional, Sequence

from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize

from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


class PyTrackingAnalysisModuleTrackingResultWriter:
    def __init__(self, output_folder: str, file_name: str):
        self._zip_file = zipfile.ZipFile(os.path.join(output_folder, file_name + '.zip'), 'w', zipfile.ZIP_DEFLATED)
        self._duplication_check = set()

    def write(self, tracker_name: str, repeat_index: Optional[int], sequence_name: str, predicted_bboxes: np.ndarray):
        assert (tracker_name, repeat_index, sequence_name) not in self._duplication_check
        self._duplication_check.add((tracker_name, repeat_index, sequence_name))
        tracker_folder_path = tracker_name
        if repeat_index is not None:
            tracker_folder_path += f'_{(repeat_index + 1):03d}'
        with io.StringIO() as f:
            np.savetxt(f, predicted_bboxes, fmt='%.3f', delimiter='\t')
            self._zip_file.writestr(f'{tracker_folder_path}/{sequence_name}.txt', f.getvalue())

    def close(self):
        self._zip_file.close()


class PyTrackingEvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_folder: str, file_name: str, rasterize_bbox: bool = True):
        self._writer = PyTrackingAnalysisModuleTrackingResultWriter(output_folder, file_name)
        self._tracker_name = tracker_name
        self._result_adjuster = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def close(self):
        self._writer.close()

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT], evaluation_progresses: Sequence[EvaluationProgress]):
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            track_name, predicted_bboxes = self._result_adjuster.adjust_for_pytracking(
                evaluation_result.sequence_info.dataset_name, evaluation_result.sequence_info.sequence_name, evaluation_result.output_box)
            if self._rasterize_bbox:
                predicted_bboxes = bbox_rasterize(predicted_bboxes)
            predicted_bboxes = bbox_xyxy_to_xywh(predicted_bboxes)
            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset is not None:
                repeat_index = repeat_index if evaluation_progress.this_dataset.total_repeat_times > 1 else None
            self._writer.write(self._tracker_name, repeat_index, track_name, predicted_bboxes)
