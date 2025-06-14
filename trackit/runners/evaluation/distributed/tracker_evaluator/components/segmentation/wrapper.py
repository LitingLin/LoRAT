import torch
import numpy as np
from typing import Tuple, List, Sequence

from .interface import MaskGenerator
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, scale_siamfc_cropping_params, reverse_siamfc_cropping_params, apply_siamfc_cropping_subpixel


class Segmentify_PostProcessor:
    def __init__(self, search_region_size: Tuple[int, int],
                 area_factor: float, device: torch.device,
                 search_region_based_post_process: MaskGenerator,
                 interpolation_mode: str, interpolation_align_corners: bool):
        self.search_region_size = search_region_size
        self.area_factor = area_factor
        self.search_region_based_post_process = search_region_based_post_process
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.device = device

    def start(self, max_batch_size: int):
        self.search_region_based_post_process.start()
        self.search_region_image_cache = torch.empty((max_batch_size, 3, self.search_region_size[1], self.search_region_size[0]),
                                                     dtype=torch.float32, device=self.device)

    def stop(self):
        self.search_region_based_post_process.stop()
        del self.search_region_image_cache

    def __call__(self, image: Sequence[torch.Tensor], prompt_bbox: np.ndarray) -> List[np.ndarray]:
        assert prompt_bbox.ndim == 2 and prompt_bbox.shape[1] == 4, 'prompt_bbox must in shape (N, 4)'
        assert len(image) == prompt_bbox.shape[0], 'image and prompt_bbox must have the same batch size'
        siamfc_curation_parameter = []
        prompt_bbox_on_search_region = []
        image_size = []

        for index, (curr_image, curr_bbox) in enumerate(zip(image, prompt_bbox)):
            C, H, W = curr_image.shape
            assert C == 3, 'image must have 3 channels'
            current_siamfc_curation_parameter = get_siamfc_cropping_params(curr_bbox, self.area_factor, np.array(self.search_region_size))
            _, _, current_siamfc_curation_parameter = \
                apply_siamfc_cropping(curr_image.to(torch.float32), np.array(self.search_region_size), current_siamfc_curation_parameter,
                                      self.interpolation_mode, self.interpolation_align_corners, out_image=self.search_region_image_cache[index])
            siamfc_curation_parameter.append(current_siamfc_curation_parameter)
            prompt_bbox_on_search_region.append(apply_siamfc_cropping_to_boxes(curr_bbox, current_siamfc_curation_parameter))
            image_size.append((W, H))
        predicted_mask_on_search_region = self.search_region_based_post_process(
            self.search_region_image_cache[:len(image)],
            torch.from_numpy(np.stack(prompt_bbox_on_search_region, axis=0)).to(torch.float32).to(device=self.device))
        mask_prediction_on_full_search_image = []
        for curr_mask, curr_siamfc_curation_parameter, curr_image_size in zip(predicted_mask_on_search_region, siamfc_curation_parameter, image_size):
            mask_h, mask_w = curr_mask.shape
            curr_siamfc_curation_parameter = scale_siamfc_cropping_params(curr_siamfc_curation_parameter, np.array(self.search_region_size), np.array((mask_w, mask_h)))
            mask_prediction_on_full_search_image.append(
                apply_siamfc_cropping_subpixel(curr_mask.to(torch.float32).unsqueeze(0), np.array(curr_image_size),
                                               reverse_siamfc_cropping_params(curr_siamfc_curation_parameter),
                                               self.interpolation_mode, self.interpolation_align_corners).squeeze(0).to(torch.bool).cpu().numpy())
        return mask_prediction_on_full_search_image
