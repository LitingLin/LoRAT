# modify from https://github.com/visionml/pytracking/blob/master/pytracking/analysis/extract_results.py
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import numpy as np
from trackit.miscellanies.numpy_array_builder import NumpyArrayBuilder
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.miscellanies.argsort import argsort
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


_plot_bin_gap = 0.05

_bins_of_intersection_of_union = 21
_bins_of_center_location_error = 51
_threshold_set_overlap = np.linspace(0, 1, _bins_of_intersection_of_union, dtype=np.float64)
_threshold_set_center = np.arange(0, _bins_of_center_location_error, dtype=np.float64)
_threshold_set_center_norm = np.arange(0, _bins_of_center_location_error, dtype=np.float64) / 100.0
_precision_score_bin_index = 20
_success_rate_at_overlap_0_5_bin_index = np.where(_threshold_set_overlap == 0.5)[0].item()
_success_rate_at_overlap_0_75_bin_index = np.where(_threshold_set_overlap == 0.75)[0].item()


def calc_err_center(pred_bb: np.ndarray, anno_bb: np.ndarray, normalized: bool=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = np.sqrt(((pred_center - anno_center)**2).sum(1))
    return err_center


def calc_iou_overlap(pred_bb: np.ndarray, anno_bb: np.ndarray):
    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clip(min=0)

    # Area
    intersection = sz.prod(axis=1)
    union = pred_bb[:, 2:].prod(axis=1) + anno_bb[:, 2:].prod(axis=1) - intersection

    return intersection / union


def calc_seq_err_robust(dataset_name: str, pred_bb: np.ndarray, anno_bb: np.ndarray, target_visible: Optional[np.ndarray]=None, stark_behavior:bool=True):
    if stark_behavior:
        valid = ((anno_bb[:, 2:] > 0.0).sum(axis=1) == 2)
    else:
        valid = ((anno_bb > 0.0).sum(axis=1) == 4)
    if target_visible is not None and dataset_name in ('LaSOT', 'LaSOT_Extension', 'VastTrack'):
        target_visible = target_visible.astype(bool)
        valid = valid & target_visible
    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset_name.startswith('UAV'):
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset_name in ('LaSOT', 'VastTrack'):
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    return err_overlap, err_center, err_center_normalized, valid


def compute_one_pass_evaluation_metrics(dataset_name: str, pred_bb: np.ndarray, anno_bb: np.ndarray,
                                        target_visible: Optional[np.ndarray],
                                        time_costs: np.ndarray,
                                        compatibility_helper: ExternalToolkitCompatibilityHelper,
                                        stark_behavior:bool=True,
                                        exclude_invalid_frames: bool = False):
    pred_bb = compatibility_helper.adjust(dataset_name, pred_bb)
    anno_bb = compatibility_helper.adjust(dataset_name, anno_bb)
    pred_bb = bbox_xyxy_to_xywh(pred_bb)
    anno_bb = bbox_xyxy_to_xywh(anno_bb)
    err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(dataset_name, pred_bb, anno_bb, target_visible, stark_behavior)

    if exclude_invalid_frames:
        seq_length = valid_frame.long().sum()
    else:
        seq_length = anno_bb.shape[0]

    ave_success_rate_plot_overlap = ((err_overlap.reshape(-1, 1) > _threshold_set_overlap.reshape(1, -1)).sum(0).astype(np.float64) / seq_length).astype(np.float32)
    ave_success_rate_plot_center = ((err_center.reshape(-1, 1) <= _threshold_set_center.reshape(1, -1)).sum(0).astype(np.float64) / seq_length).astype(np.float32)
    ave_success_rate_plot_center_norm = ((err_center_normalized.reshape(-1, 1) <= _threshold_set_center_norm.reshape(1, -1)).sum(0).astype(np.float64) / seq_length).astype(np.float32)
    return OPEMetrics(ave_success_rate_plot_overlap, ave_success_rate_plot_center, ave_success_rate_plot_center_norm, time_costs.mean().item()), err_overlap


@dataclass(frozen=True)
class OPEMetrics:
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: float

    @property
    def success_score(self) -> float:
        return np.mean(self.success_curve).item()

    @property
    def precision_score(self) -> float:
        return self.precision_curve[20].item()

    @property
    def normalized_precision_score(self) -> float:
        return self.normalized_precision_curve[20].item()

    @property
    def success_rate_at_overlap_0_5(self) -> float:
        return self.success_curve[_success_rate_at_overlap_0_5_bin_index].item()

    @property
    def success_rate_at_overlap_0_75(self) -> float:
        return self.success_curve[_success_rate_at_overlap_0_75_bin_index].item()

    def get_fps(self) -> float:
        return 1. / self.time_cost


def compute_OPE_metrics_mean(metrics: Sequence[OPEMetrics]) -> OPEMetrics:
    return OPEMetrics(
        success_curve=np.mean([m.success_curve for m in metrics], axis=0),
        precision_curve=np.mean([m.precision_curve for m in metrics], axis=0),
        normalized_precision_curve=np.mean([m.normalized_precision_curve for m in metrics], axis=0),
        time_cost=np.mean([m.time_cost for m in metrics]).item()
    )


@dataclass(frozen=True)
class DatasetOPEMetricsList:
    sequence_name: Sequence[str]
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: np.ndarray

    def __getitem__(self, index: int):
        return self.sequence_name[index], \
            OPEMetrics(self.success_curve[index], self.precision_curve[index], self.normalized_precision_curve[index], self.time_cost[index].item())

    def __len__(self):
        return len(self.sequence_name)

    def sort_by_sequence_name(self):
        sorted_indices = np.array(argsort(self.sequence_name))

        return DatasetOPEMetricsList(
            tuple(self.sequence_name[index] for index in sorted_indices),
            self.success_curve[sorted_indices],
            self.precision_curve[sorted_indices],
            self.normalized_precision_curve[sorted_indices],
            self.time_cost[sorted_indices])

    def get_mean(self):
        return OPEMetrics(np.mean(self.success_curve, axis=0),
                          np.mean(self.precision_curve, axis=0),
                          np.mean(self.normalized_precision_curve, axis=0),
                          np.mean(self.time_cost).item())

    def get_success_score(self) -> np.ndarray:
        return np.mean(self.success_curve, axis=1)

    def get_precision_score(self) -> np.ndarray:
        return self.precision_curve[:, _precision_score_bin_index]

    def get_normalized_precision_score(self) -> np.ndarray:
        return self.normalized_precision_curve[:, _precision_score_bin_index]


class DatasetOPEMetricsListBuilder:
    def __init__(self):
        self._sequence_name = []
        self._average_overlap = NumpyArrayBuilder(np.float64)
        self._success_rate_at_iou_0_5 = NumpyArrayBuilder(np.float64)
        self._success_rate_at_iou_0_75 = NumpyArrayBuilder(np.float64)
        self._success_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_intersection_of_union,))
        self._precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_center_location_error,))
        self._norm_precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_center_location_error,))
        self._average_time_cost = NumpyArrayBuilder(np.float64)

    def append(self, sequence_name: str, metrics: OPEMetrics):
        self._sequence_name.append(sequence_name)
        self._success_curve.append(metrics.success_curve)
        self._precision_curve.append(metrics.precision_curve)
        self._norm_precision_curve.append(metrics.normalized_precision_curve)
        self._average_time_cost.append(metrics.time_cost)

    def build(self):
        return DatasetOPEMetricsList(self._sequence_name, self._success_curve.build(),
                                     self._precision_curve.build(), self._norm_precision_curve.build(),
                                     self._average_time_cost.build())
