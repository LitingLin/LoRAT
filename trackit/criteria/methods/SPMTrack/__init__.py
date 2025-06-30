import torch
import torch.nn as nn
from trackit.criteria import CriterionOutput
from trackit.criteria.modules.iou_loss import bbox_overlaps
from trackit.miscellanies.torch.distributed.reduce_mean import reduce_mean_


class SPMTrackCriteria(nn.Module):
    def __init__(self, cls_loss: nn.Module, bbox_reg_loss: nn.Module,
                 iou_aware_classification_score: bool,
                 cls_loss_weight: float, bbox_reg_loss_weight: float,
                 cls_loss_display_name: str, bbox_reg_loss_display_name: str, warmup_epochs: int):
        super().__init__()
        self.cls_loss = cls_loss
        self.bbox_reg_loss = bbox_reg_loss
        self.iou_aware_classification_score = iou_aware_classification_score
        self.cls_loss_weight = cls_loss_weight
        self.bbox_reg_loss_weight = bbox_reg_loss_weight
        self.cls_loss_display_name = cls_loss_display_name
        self.bbox_reg_loss_display_name = bbox_reg_loss_display_name
        self._across_all_nodes_normalization = True
        self._warmup_epochs = warmup_epochs

    def forward(self, outputs: dict, targets: dict):
        metrics = {}
        extra_metrics = {}
        total_loss = 0
        epoch = targets['epoch']

        for i, output in enumerate(outputs):
            num_positive_samples = targets[f'num_positive_samples_{i}']
            assert isinstance(num_positive_samples, torch.Tensor)

            reduce_mean_(num_positive_samples)  # caution: inplace update
            num_positive_samples.clamp_(min=1.)

            predicted_score_map = output['score_map'].to(torch.float)
            predicted_bboxes = output['boxes'].to(torch.float)
            groundtruth_bboxes = targets[f'boxes_{i}']

            N, H, W = predicted_score_map.shape

            # shape: (num_positive_samples, )
            positive_sample_batch_dim_index = targets[f'positive_sample_batch_dim_indices_{i}']
            # shape: (num_positive_samples, )
            positive_sample_feature_map_dim_index = targets[f'positive_sample_map_dim_indices_{i}']

            has_positive_samples = positive_sample_batch_dim_index is not None

            if has_positive_samples:
                predicted_bboxes = predicted_bboxes.view(N, H * W, 4)
                predicted_bboxes = predicted_bboxes[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]
                groundtruth_bboxes = groundtruth_bboxes[positive_sample_batch_dim_index]

            with torch.no_grad():
                groundtruth_response_map = torch.zeros((N, H * W),  dtype=torch.float32, device=predicted_score_map.device)
                if self.iou_aware_classification_score:
                    groundtruth_response_map.index_put_(
                        (positive_sample_batch_dim_index, positive_sample_feature_map_dim_index),
                        bbox_overlaps(groundtruth_bboxes, predicted_bboxes, is_aligned=True))
                else:
                    groundtruth_response_map[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] = 1.

            cls_loss = self.cls_loss(predicted_score_map.view(N, -1), groundtruth_response_map).sum() / num_positive_samples

            if has_positive_samples:
                reg_loss = self.bbox_reg_loss(predicted_bboxes, groundtruth_bboxes).sum() / num_positive_samples
            else:
                reg_loss = predicted_bboxes.mean() * 0

            if self.cls_loss_weight != 1.:
                cls_loss = cls_loss * self.cls_loss_weight

            bbox_reg_loss_weight = self.bbox_reg_loss_weight
            if bbox_reg_loss_weight != 1. or (self._warmup_epochs > 0 and 0 <= epoch < self._warmup_epochs):
                if self._warmup_epochs > 0 and 0 <= epoch < self._warmup_epochs:
                    bbox_reg_loss_weight = bbox_reg_loss_weight * 10
                reg_loss = reg_loss * bbox_reg_loss_weight

            cls_loss_cpu = cls_loss.detach().cpu().item()
            reg_loss_cpu = reg_loss.detach().cpu().item()

            metrics.update({f'Loss/{self.cls_loss_display_name}_{i}': cls_loss_cpu, f'Loss/{self.bbox_reg_loss_display_name}_{i}': reg_loss_cpu})
            extra_metrics.update({f'Loss/{self.cls_loss_display_name}_unscale_{i}': cls_loss_cpu / self.cls_loss_weight, f'Loss/{self.bbox_reg_loss_display_name}_unscale_{i}': reg_loss_cpu / bbox_reg_loss_weight})

            total_loss += cls_loss
            total_loss += reg_loss

        return CriterionOutput(total_loss, metrics, extra_metrics)
