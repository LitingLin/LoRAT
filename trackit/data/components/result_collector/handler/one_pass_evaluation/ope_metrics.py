from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import numpy as np
from trackit.core.operator.numpy.bbox.iou import bbox_compute_iou
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.miscellanies.numpy_array_builder import NumpyArrayBuilder
from trackit.miscellanies.argsort import argsort


_bins_of_center_location_error = 51
_bins_of_normalized_center_location_error = 51
_bins_of_intersection_of_union = 21
_bin_index_precision_score = 20
_bin_index_normalized_precision_score = 20


def calculate_center_location_error(pred_bb: np.ndarray, anno_bb: np.ndarray, normalized: bool=False):
    pred_center = (pred_bb[:, :2] + pred_bb[:, 2:]) / 2.
    anno_center = (anno_bb[:, :2] + anno_bb[:, 2:]) / 2.

    if normalized:
        anno_wh = anno_bb[:, 2:] - anno_bb[:, :2]
        pred_center = pred_center / anno_wh
        anno_center = anno_center / anno_wh

    err_center = np.sqrt(((pred_center - anno_center)**2).sum(1))
    return err_center


def _calc_curves(all_frame_iou: np.ndarray, all_frame_center_error: np.ndarray, all_frame_norm_center_error: np.ndarray):
    all_frame_iou = all_frame_iou[:, np.newaxis]
    all_frame_center_error = all_frame_center_error[:, np.newaxis]
    all_frame_norm_center_error = all_frame_norm_center_error[:, np.newaxis]

    thr_iou = np.linspace(0, 1, _bins_of_intersection_of_union)[np.newaxis, :]
    thr_ce = np.arange(0, _bins_of_center_location_error)[np.newaxis, :]
    thr_nce = np.linspace(0, 0.5, _bins_of_normalized_center_location_error)[np.newaxis, :]

    bin_iou = np.greater_equal(all_frame_iou, thr_iou)
    bin_ce = np.less_equal(all_frame_center_error, thr_ce)
    bin_nce = np.less_equal(all_frame_norm_center_error, thr_nce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_nce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve


def _calc_ao_sr(frames_iou: np.ndarray) -> Tuple[float, float, float]:
    assert frames_iou.shape[0] > 0
    assert frames_iou.ndim == 1
    return np.mean(frames_iou).item(), np.mean(frames_iou >= 0.5).item(), np.mean(frames_iou >= 0.75).item()


def compute_one_pass_evaluation_metrics(predicted_bounding_boxes: np.ndarray,
                                        groundtruth_bounding_boxes: np.ndarray,
                                        bounding_box_validity_flags: Optional[np.ndarray],
                                        time_costs: np.ndarray):
    frames_iou_ = bbox_compute_iou(predicted_bounding_boxes, groundtruth_bounding_boxes)
    assert not (frames_iou_ > 1.).any()

    groundtruth_bboxes_validity = bbox_is_valid(groundtruth_bounding_boxes)
    if bounding_box_validity_flags is None:
        bounding_box_validity_flags = groundtruth_bboxes_validity
    else:
        bounding_box_validity_flags = np.logical_and(groundtruth_bboxes_validity, bounding_box_validity_flags)

    frames_iou = frames_iou_.copy()
    if bounding_box_validity_flags is not None:
        frames_iou[~bounding_box_validity_flags] = 0.

    all_frame_center_location_errors = calculate_center_location_error(predicted_bounding_boxes, groundtruth_bounding_boxes, False)
    all_frame_normalized_center_location_errors = calculate_center_location_error(predicted_bounding_boxes, groundtruth_bounding_boxes, True)

    predicted_bboxes_validity = bbox_is_valid(predicted_bounding_boxes)
    frames_iou_[~predicted_bboxes_validity] = -1.0
    all_frame_center_location_errors[~predicted_bboxes_validity] = float('inf')
    all_frame_normalized_center_location_errors[~predicted_bboxes_validity] = float('inf')

    if bounding_box_validity_flags is not None:
        frames_iou_ = frames_iou_[bounding_box_validity_flags]
        all_frame_center_location_errors = all_frame_center_location_errors[bounding_box_validity_flags]
        all_frame_normalized_center_location_errors = all_frame_normalized_center_location_errors[bounding_box_validity_flags]

    frames_iou_[frames_iou_ == 0] = -1.0

    succ_curve, prec_curve, norm_prec_curve = _calc_curves(frames_iou_, all_frame_center_location_errors,
                                                           all_frame_normalized_center_location_errors)
    frames_iou_[frames_iou_ < 0.] = 0
    ao, sr_at_0_5, sr_at_0_75 = _calc_ao_sr(frames_iou_)

    average_time_cost = time_costs.mean().item()

    return OPEMetrics(ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve, average_time_cost), frames_iou


@dataclass(frozen=True)
class OPEMetrics:
    average_overlap: float
    success_rate_at_iou_0_5: float
    success_rate_at_iou_0_75: float
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: float

    @property
    def success_score(self) -> float:
        return np.mean(self.success_curve).item()

    @property
    def precision_score(self) -> float:
        return self.precision_curve[_bin_index_precision_score].item()

    @property
    def normalized_precision_score(self) -> float:
        return self.normalized_precision_curve[_bin_index_normalized_precision_score].item()

    def get_fps(self) -> float:
        return 1. / self.time_cost


def compute_OPE_metrics_mean(metrics: Sequence[OPEMetrics]) -> OPEMetrics:
    return OPEMetrics(
        average_overlap=np.mean([m.average_overlap for m in metrics]).item(),
        success_rate_at_iou_0_5=np.mean([m.success_rate_at_iou_0_5 for m in metrics]).item(),
        success_rate_at_iou_0_75=np.mean([m.success_rate_at_iou_0_75 for m in metrics]).item(),
        success_curve=np.mean([m.success_curve for m in metrics], axis=0),
        precision_curve=np.mean([m.precision_curve for m in metrics], axis=0),
        normalized_precision_curve=np.mean([m.normalized_precision_curve for m in metrics], axis=0),
        time_cost=np.mean([m.time_cost for m in metrics]).item()
    )


@dataclass(frozen=True)
class DatasetOPEMetricsList:
    sequence_name: Sequence[str]
    average_overlap: np.ndarray
    success_rate_at_iou_0_5: np.ndarray
    success_rate_at_iou_0_75: np.ndarray
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: np.ndarray

    def __getitem__(self, index: int):
        return self.sequence_name[index], \
            OPEMetrics(self.average_overlap[index].item(), self.success_rate_at_iou_0_5[index].item(),
                            self.success_rate_at_iou_0_75[index].item(), self.success_curve[index], self.precision_curve[index],
                            self.normalized_precision_curve[index], self.time_cost[index].item())

    def __len__(self):
        return len(self.sequence_name)

    def sort_by_sequence_name(self):
        sorted_indices = np.array(argsort(self.sequence_name))

        return DatasetOPEMetricsList(
            tuple(self.sequence_name[index] for index in sorted_indices),
            self.average_overlap[sorted_indices],
            self.success_rate_at_iou_0_5[sorted_indices],
            self.success_rate_at_iou_0_75[sorted_indices],
            self.success_curve[sorted_indices],
            self.precision_curve[sorted_indices],
            self.normalized_precision_curve[sorted_indices],
            self.time_cost[sorted_indices])

    def get_mean(self):
        return OPEMetrics(np.mean(self.average_overlap).item(), np.mean(self.success_rate_at_iou_0_5).item(),
                            np.mean(self.success_rate_at_iou_0_75).item(), np.mean(self.success_curve, axis=0),
                            np.mean(self.precision_curve, axis=0), np.mean(self.normalized_precision_curve, axis=0),
                            np.mean(self.time_cost).item())

    def get_success_score(self) -> np.ndarray:
        return np.mean(self.success_curve, axis=1)

    def get_precision_score(self) -> np.ndarray:
        return self.precision_curve[:, _bin_index_precision_score]

    def get_normalized_precision_score(self) -> np.ndarray:
        return self.normalized_precision_curve[:, _bin_index_normalized_precision_score]


class DatasetOPEMetricsListBuilder:
    def __init__(self):
        self._sequence_name = []
        self._average_overlap = NumpyArrayBuilder(np.float64)
        self._success_rate_at_iou_0_5 = NumpyArrayBuilder(np.float64)
        self._success_rate_at_iou_0_75 = NumpyArrayBuilder(np.float64)
        self._success_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_intersection_of_union,))
        self._precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_center_location_error,))
        self._norm_precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_normalized_center_location_error,))
        self._average_time_cost = NumpyArrayBuilder(np.float64)

    def append(self, sequence_name: str, metrics: OPEMetrics):
        self._sequence_name.append(sequence_name)
        self._average_overlap.append(metrics.average_overlap)
        self._success_rate_at_iou_0_5.append(metrics.success_rate_at_iou_0_5)
        self._success_rate_at_iou_0_75.append(metrics.success_rate_at_iou_0_75)
        self._success_curve.append(metrics.success_curve)
        self._precision_curve.append(metrics.precision_curve)
        self._norm_precision_curve.append(metrics.normalized_precision_curve)
        self._average_time_cost.append(metrics.time_cost)

    def build(self):
        return DatasetOPEMetricsList(self._sequence_name, self._average_overlap.build(),
                                     self._success_rate_at_iou_0_5.build(),
                                     self._success_rate_at_iou_0_75.build(), self._success_curve.build(),
                                     self._precision_curve.build(), self._norm_precision_curve.build(),
                                     self._average_time_cost.build())
