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


class GOT10kEvaluationToolFileWriter:
    def __init__(self, output_folder: str, output_file_name: str):
        self._output_file_path_prefix = os.path.join(output_folder, output_file_name)
        self._zipfile = zipfile.ZipFile(self._output_file_path_prefix + '.zip', 'w', zipfile.ZIP_DEFLATED)
        self._single_run_zipfiles = {}
        self._duplication_check = set()

    def write(self, tracker_name: str, sequence_name: str,
              predicted_bboxes: np.ndarray, time_costs: np.ndarray,
              repeat_index: Optional[int] = None):
        single_run_zipfile = None
        if repeat_index is not None:
            if repeat_index not in self._single_run_zipfiles:
                self._single_run_zipfiles[repeat_index] = zipfile.ZipFile(self._output_file_path_prefix + f'{repeat_index + 1:03d}.zip', 'w', zipfile.ZIP_DEFLATED)
            single_run_zipfile = self._single_run_zipfiles[repeat_index]

        if repeat_index is None:
            repeat_index = 0

        assert (tracker_name, repeat_index, sequence_name) not in self._duplication_check, "duplicated sequence name detected"
        self._duplication_check.add((tracker_name, repeat_index, sequence_name))

        with io.BytesIO() as result_file_content:
            np.savetxt(result_file_content, predicted_bboxes, fmt='%.4f',
                       delimiter=',')
            self._zipfile.writestr('/'.join((tracker_name, sequence_name, f'{sequence_name}_{repeat_index + 1:03d}.txt')),
                                   result_file_content.getvalue())
            if single_run_zipfile is not None:
                single_run_zipfile.writestr('/'.join((tracker_name, sequence_name, f'{sequence_name}_001.txt')),
                                            result_file_content.getvalue())
        if repeat_index == 0 or single_run_zipfile is not None:
            with io.BytesIO() as time_file_content:
                np.savetxt(time_file_content, time_costs, fmt='%.8f')
                if repeat_index == 0:
                    self._zipfile.writestr('/'.join((tracker_name, sequence_name, f'{sequence_name}_time.txt')), time_file_content.getvalue())
                if single_run_zipfile is not None:
                    single_run_zipfile.writestr('/'.join((tracker_name, sequence_name, f'{sequence_name}_time.txt')), time_file_content.getvalue())

    def close(self):
        for zip_file in self._single_run_zipfiles.values():
            zip_file.close()
        self._zipfile.close()


class GOT10KEvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str,  output_folder: str, file_name: str = 'GOT10k', rasterize_bbox: bool = True):
        self._writer = GOT10kEvaluationToolFileWriter(output_folder, file_name)
        self._tracker_name = tracker_name
        self._rasterize_bbox = rasterize_bbox

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT], evaluation_progresses: Sequence[EvaluationProgress]):
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            predicted_bboxes = evaluation_result.output_box
            if self._rasterize_bbox:
                predicted_bboxes = bbox_rasterize(predicted_bboxes)
            predicted_bboxes = bbox_xyxy_to_xywh(predicted_bboxes)
            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset is not None:
                repeat_index = repeat_index if evaluation_progress.this_dataset.total_repeat_times > 1 else None

            self._writer.write(self._tracker_name, evaluation_result.sequence_info.sequence_name,
                               predicted_bboxes, evaluation_result.time_cost,
                               repeat_index)

    def close(self):
        self._writer.close()
