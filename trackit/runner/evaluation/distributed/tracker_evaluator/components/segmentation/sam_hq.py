from typing import List

import torch
import torch.nn.functional as F

from trackit.core.operator.bbox.scale import bbox_scale
from trackit.core.operator.bbox.rasterize import rasterize_bbox_torch_

from .interface import MaskGenerator
from .segment_anything_hq.build_sam import sam_model_registry as sam_model_registry_hq
from .segment_anything_hq.modeling.sam import Sam


_sam_models = {
    'sam_hq_vit_b': {
        'build_fn': sam_model_registry_hq['vit_b'],
        'checkpoint_url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth'
    },
    'sam_hq_vit_h': {
        'build_fn': sam_model_registry_hq['vit_h'],
        'checkpoint_url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth'
    },
    'sam_hq_vit_l': {
        'build_fn': sam_model_registry_hq['vit_l'],
        'checkpoint_url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth'
    },
    'sam_hq_vit_t': {
        'build_fn': sam_model_registry_hq['vit_tiny'],
        'checkpoint_url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth'
    }
}


class SAMHQ_BoxToMaskPostProcess(MaskGenerator):
    def __init__(self, model_name: str, device: torch.device,
                 interpolation_mode: str, interpolation_align_corners: bool,
                 mask_threshold: float = 0.5, hq_token_only=False):
        self._model_name = model_name
        self._device = device
        self._interpolation_mode = interpolation_mode
        self._interpolation_align_corners = interpolation_align_corners
        self._mask_threshold = mask_threshold
        self._hq_token_only = hq_token_only

    def start(self):
        self._sam: Sam = _sam_models[self._model_name]['build_fn']().to(self._device)
        self._sam.load_state_dict(torch.hub.load_state_dict_from_url(_sam_models[self._model_name]['checkpoint_url'], map_location=self._device))

    def stop(self):
        del self._sam

    def __call__(self, search_region: torch.Tensor, predicted_bbox: torch.Tensor) -> torch.Tensor:
        assert search_region.dim() == 4, 'search_region must have in shape (N, C, H, W)'
        assert predicted_bbox.ndim == 2, 'predicted_bbox must have 2 dimensions'
        # note that search region must be in original (0 - 255) RGB format, not normalized
        h, w = search_region.shape[-2:]

        image_encoder = self._sam.image_encoder
        prompt_encoder = self._sam.prompt_encoder
        mask_decoder = self._sam.mask_decoder

        pixel_mean = self._sam.pixel_mean
        pixel_std = self._sam.pixel_std

        with torch.inference_mode():
            predicted_bbox = bbox_scale(predicted_bbox,
                                        torch.tensor(((1024 / h), (1024 / w)), device=predicted_bbox.device,
                                                     dtype=torch.float32))
            rasterize_bbox_torch_(predicted_bbox)

            scaled_search_region = F.interpolate(search_region, (1024, 1024), mode=self._interpolation_mode,
                                                 align_corners=self._interpolation_align_corners)

            scaled_search_region.sub_(pixel_mean).div_(pixel_std)
            image_embeddings, interm_embeddings = image_encoder(scaled_search_region)
            interm_embeddings = interm_embeddings[0]
            all_search_region_mask = []
            for curr_embedding, curr_bbox, curr_interm in zip(image_embeddings, predicted_bbox, interm_embeddings):
                sparse_embeddings, dense_embeddings = prompt_encoder(
                    boxes=curr_bbox.unsqueeze(0)
                )
                low_res_masks, iou_predictions = mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    hq_token_only=self._hq_token_only,
                    interm_embeddings=curr_interm.unsqueeze(0).unsqueeze(0),
                )
                masks = low_res_masks > self._mask_threshold
                masks = masks.squeeze(0).squeeze(0)
                all_search_region_mask.append(masks)
            return torch.stack(all_search_region_mask)
