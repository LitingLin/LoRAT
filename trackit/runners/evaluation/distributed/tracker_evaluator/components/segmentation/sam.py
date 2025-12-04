from typing import List, Tuple, Optional
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import clip_boxes_to_image

from trackit.core.operator.bbox.scale import bbox_scale
from trackit.core.operator.bbox.rasterize import rasterize_bbox_torch_

from .interface import MaskGenerator


checkpoint_urls = {
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
}


class SAM_BoxToMaskPostProcess(MaskGenerator):
    def __init__(self, model_type: str, model_name: str, device: torch.device,
                 interpolation_mode: str, interpolation_align_corners: bool,
                 mask_threshold: float, enable_data_parallel: bool):
        self._model_type = model_type
        if self._model_type == 'fast':
            import os
            os.environ['SEGMENT_ANYTHING_FAST_USE_FLASH_4'] = '0'
        self._model_name = model_name
        self._device = device
        self._interpolation_mode = interpolation_mode
        self._interpolation_align_corners = interpolation_align_corners
        self._mask_threshold = mask_threshold
        self._mask_restrict_to_bbox = False
        self._enable_data_parallel = enable_data_parallel

    def start(self):
        if self._model_type == 'fast':
            from .segment_anything_fast.build_sam import sam_model_registry
        elif self._model_type == 'official':
            from .segment_anything.build_sam import sam_model_registry
        else:
            raise ValueError(f'Unknown model type: {self._model_type}')

        self._sam = sam_model_registry[self._model_name]().to(self._device)
        self._sam.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_urls[self._model_name], map_location=self._device))
        self._dtype = torch.float32
        if self._model_type == 'fast':
            from .segment_anything_fast.build_sam import _apply_eval_dtype_sam
            _apply_eval_dtype_sam(self._sam, torch.bfloat16)
            self._dtype = torch.bfloat16
        self._pixel_mean = self._sam.pixel_mean
        self._pixel_std = self._sam.pixel_std
        self._sam = SAM(self._sam, self._mask_threshold)
        self._sam.eval()
        if self._enable_data_parallel:
            self._sam.image_encoder = torch.nn.DataParallel(self._sam.image_encoder, output_device=self._device)
            print('[SAM] DataParallel is enabled.')

    def stop(self):
        del self._sam
        del self._dtype
        del self._pixel_mean
        del self._pixel_std
        gc.collect()

    def __call__(self, search_region: torch.Tensor, predicted_bbox: torch.Tensor) -> List[torch.Tensor]:
        assert search_region.dim() == 4, 'search_region must have in shape (N, C, H, W)'
        assert predicted_bbox.ndim == 2, 'predicted_bbox must have in shape (N, 4)'
        # note that search region must be in original (0 - 255) RGB format, not normalized
        h, w = search_region.shape[-2:]

        with torch.inference_mode():
            search_region.sub_(self._pixel_mean).div_(self._pixel_std)
            if search_region.shape[-2:] != (1024, 1024):
                predicted_bbox = bbox_scale(predicted_bbox, torch.tensor(((1024 / h), (1024 / w)), device=predicted_bbox.device, dtype=torch.float32))

                search_region = F.interpolate(search_region, (1024, 1024), mode=self._interpolation_mode, align_corners=self._interpolation_align_corners)

            search_region = search_region.to(self._dtype)
            rasterize_bbox_torch_(predicted_bbox)
            predicted_bbox = clip_boxes_to_image(predicted_bbox, (1024, 1024))
            all_search_region_mask = self._sam(search_region, predicted_bbox)
        return all_search_region_mask

    def get_recommended_input_resolution(self) -> Optional[Tuple[int, int]]:
        return 1024, 1024


def _box_to_mask(box: torch.Tensor, image_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    box = box.to(torch.long)
    mask = torch.zeros((image_size[1], image_size[0]), dtype=torch.bool, device=device)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


class SAM(nn.Module):
    def __init__(self, sam: nn.Module, mask_threshold: float):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.register_buffer('mask_threshold', torch.as_tensor(mask_threshold, device=sam.device, dtype=torch.float32))
        self._mask_restrict_to_bbox = False

    def forward(self, image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.image_encoder(image)
        all_search_region_mask = []
        for curr_embedding, curr_bbox in zip(image_embeddings, bbox):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                boxes=curr_bbox.unsqueeze(0),
                points=None, masks=None
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = low_res_masks > self.mask_threshold
            masks = masks.squeeze(0).squeeze(0)

            if self._mask_restrict_to_bbox:
                masks = torch.logical_and(masks, _box_to_mask(curr_bbox, (1024, 1024), self._device))

            all_search_region_mask.append(masks)
        return torch.stack(all_search_region_mask, dim=0)
