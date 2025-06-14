import numpy as np
import torch
import os
from typing import Optional
from trackit.core.runtime.context.task import get_current_task_context
from trackit.miscellanies.image.io import write_image
from trackit.miscellanies.image.draw import draw_box_on_image_
from PIL import Image


def visualize_tracking_result(dataset_name: str, track_name: str, frame_index: int, search_region: torch.Tensor,
                              predicted_box: np.ndarray,
                              predicted_mask: Optional[np.ndarray],
                              predicted_mask_on_full_search_image: Optional[Image.Image]):
    output_path = get_current_task_context().get_current_epoch_output_path()
    if output_path is None:
        return
    output_path = os.path.join(output_path, 'tracking_result_visualization')
    os.makedirs(output_path, exist_ok=True)
    output_image_path = os.path.join(output_path, f'{dataset_name}_{track_name}_{frame_index}_x.png')
    search_region = search_region.permute(1, 2, 0).to(torch.uint8).cpu().numpy(force=True)
    if predicted_box is not None:
        draw_box_on_image_(search_region, predicted_box, color=(0, 255, 0), thickness=2)

    write_image(search_region, output_image_path)
    if predicted_mask is not None:
        output_image_path = os.path.join(output_path, f'{dataset_name}_{track_name}_{frame_index}_x_mask.png')
        write_image(predicted_mask, output_image_path)
    if predicted_mask_on_full_search_image is not None:
        output_image_path = os.path.join(output_path, f'{dataset_name}_{track_name}_{frame_index}_x_mask_full.png')
        predicted_mask_on_full_search_image.save(output_image_path)
