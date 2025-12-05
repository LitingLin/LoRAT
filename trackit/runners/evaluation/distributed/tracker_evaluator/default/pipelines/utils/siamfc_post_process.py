from typing import Tuple

import torch
import numpy as np
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params


def siamfc_post_process(post_process_output: dict, curation_parameters: np.ndarray) -> Tuple:
    predicted_scores, predicted_bounding_boxes = (post_process_output['confidence'], post_process_output['box'])
    predicted_scores = predicted_scores.cpu()
    assert torch.all(torch.isfinite(predicted_scores))
    predicted_scores = predicted_scores.numpy()
    predicted_bounding_boxes = predicted_bounding_boxes.cpu()
    assert torch.all(torch.isfinite(predicted_bounding_boxes))
    predicted_bounding_boxes = predicted_bounding_boxes.to(torch.float64).numpy()

    predicted_bounding_boxes = apply_siamfc_cropping_to_boxes(
        predicted_bounding_boxes,
        reverse_siamfc_cropping_params(curation_parameters))

    return predicted_scores, predicted_bounding_boxes
