import os.path
from typing import Sequence
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from ...progress_tracer import EvaluationProgress
from ..utils.writer import ZipfileWriter
from .. import EvaluationResultHandler
from .dump import dump_sequence_evaluation_result


class EvaluationResultPersistence(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_folder: str, file_name: str, rasterize_bbox: bool):
        self._tracker_name = tracker_name
        self._folder_writer = ZipfileWriter(os.path.join(output_folder, file_name + '.zip'))
        self._rasterize_bbox = rasterize_bbox

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT], evaluation_progresses: Sequence[EvaluationProgress]) -> None:
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset is not None:
                if evaluation_progress.this_dataset.total_repeat_times == 1:
                    repeat_index = None
            predicted_target_bounding_boxes = evaluation_result.groundtruth_box
            if self._rasterize_bbox:
                predicted_target_bounding_boxes = bbox_rasterize(predicted_target_bounding_boxes)
            dump_sequence_evaluation_result(self._folder_writer, self._tracker_name, repeat_index,
                                            evaluation_result.sequence_info.dataset_full_name,
                                            evaluation_result.sequence_info.sequence_name,
                                            evaluation_result.evaluated_frame_indices,
                                            evaluation_result.output_confidence,
                                            predicted_target_bounding_boxes,
                                            evaluation_result.time_cost)

    def close(self) -> None:
        self._folder_writer.close()
