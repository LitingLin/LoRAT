import torch
from typing import Tuple
from . import TrackerOutputPostProcess


class PostProcessing_BoxWithScoreMap(TrackerOutputPostProcess):
    def __init__(self, device: torch.device,
                 response_map_size: Tuple[int, int],
                 search_region_size: Tuple[int, int],
                 window_penalty_ratio: float = 0.):
        self._response_map_size = response_map_size
        self._search_region_size = search_region_size
        self._device = device
        self._enable_gaussian_score_map_penalty = window_penalty_ratio > 0.
        self._window_penalty_ratio = window_penalty_ratio

    def start(self):
        self._scale_factor = torch.tensor((self._search_region_size[0], self._search_region_size[1]), dtype=torch.float).to(self._device)
        if self._enable_gaussian_score_map_penalty:
            self._window = torch.flatten(torch.outer(torch.hann_window(self._response_map_size[1], periodic=False),
                                                     torch.hann_window(self._response_map_size[0], periodic=False))).to(self._device)

    def stop(self):
        del self._scale_factor
        if self._enable_gaussian_score_map_penalty:
            del self._window

    def __call__(self, output):
        # shape: (N, H, W), (N, H, W, 4)
        predicted_score_map = output['score_map'].detach().float().sigmoid()
        predicted_bbox = output['boxes'].detach().float()

        N, H, W = predicted_score_map.shape
        predicted_score_map = predicted_score_map.view(N, H * W)

        if self._enable_gaussian_score_map_penalty:
            # window penalty
            score_map_with_penalty = predicted_score_map * (1 - self._window_penalty_ratio) + \
                     self._window.view(1, H * W) * self._window_penalty_ratio
            _, best_idx = torch.max(score_map_with_penalty, 1, keepdim=True)
            confidence_score = torch.gather(predicted_score_map, 1, best_idx)
        else:
            confidence_score, best_idx = torch.max(predicted_score_map, 1, keepdim=True)

        confidence_score = confidence_score.squeeze(1)
        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        bounding_box = torch.gather(predicted_bbox, 1, best_idx.view(N, 1, 1).expand(-1, -1, 4)).squeeze(1)
        bounding_box = (bounding_box.view(N, 2, 2) * self._scale_factor.view(1, 1, 2)).view(N, 4)
        return {'box': bounding_box, 'confidence': confidence_score}
