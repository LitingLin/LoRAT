import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from trackit.core.operator.numpy.bbox.scale_and_translate import bbox_scale_and_translate
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize_


def reverse_scale_and_translation_parameters(scale: np.ndarray, translation: np.ndarray):
    return 1. / scale, -translation / scale


def scale_and_translate(img: torch.Tensor, output_size: np.ndarray,
                        scale: np.ndarray, translation: np.ndarray,
                        background_color: Optional[torch.Tensor] = None,
                        mode: str = 'bilinear', align_corners: bool = False,
                        output_img: Optional[torch.Tensor] = None,
                        return_adjusted_params: bool = False) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray, np.ndarray]]:
    """
    Args:
        img (torch.Tensor): (n, c, h, w) or (c, h, w)
        output_size (np.ndarray): (2)
        scale (np.ndarray): (n, 2) or (2)
        translation (np.ndarray): (n, 2) or (2)
        background_color (torch.Tensor | None): (n, c) or (n, 1) or (c)
        mode (str): interpolation algorithm
        align_corners (bool): interpolation align at corners or half pixel centers (refers to the source image)
        output_img (torch.Tensor | None): (n, c, h, w) or (c, h, w)
        return_adjusted_params (bool): Whether to return the adjusted scale and translation factor due to the pixel aligned cropping
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray, np.ndarray]]:
            - If return_adjusted_params is False: Transformed image
            - If return_adjusted_params is True: (Transformed image, adjusted scale, adjusted translation)
    """
    img_dtype = img.dtype
    device = img.device
    bbox_dtype = scale.dtype
    if bbox_dtype not in (np.float32, np.float64):
        bbox_dtype = np.float32
    assert img.ndim in (3, 4)
    batch_mode = (img.ndim == 4)
    if not batch_mode:
        img = img.unsqueeze(0)
    n, c, h, w = img.shape
    if output_img is not None:
        if batch_mode:
            assert output_img.ndim == 4
        else:
            assert output_img.ndim in (3, 4)
            if output_img.ndim == 4:
                assert output_img.shape[0] == 1
            else:
                output_img = output_img.unsqueeze(0)
    else:
        output_img = torch.empty((n, c, output_size[1], output_size[0]), dtype=img_dtype, device=device)

    if background_color is not None:
        if background_color.ndim == 1:
            output_img[:] = background_color.view(1, -1, 1, 1)
        elif background_color.ndim == 2:
            b_n, b_c = background_color.shape
            assert b_n == n
            output_img[:] = background_color.view(b_n, b_c, 1, 1)
        else:
            raise RuntimeError(f"Incompatible background_color shape")
    else:
        output_img.zero_()

    output_bbox = bbox_scale_and_translate(np.asarray((0, 0, w, h), dtype=bbox_dtype), scale, translation)
    reverse_scale, reverse_translation = reverse_scale_and_translation_parameters(scale, translation)
    bbox_clip_to_image_boundary_(output_bbox, output_size)
    bbox_rasterize_(output_bbox)
    input_bbox = bbox_scale_and_translate(output_bbox, reverse_scale, reverse_translation)
    bbox_rasterize_(input_bbox)
    bbox_clip_to_image_boundary_(input_bbox, np.asarray((w, h), dtype=bbox_dtype))
    output_bbox_validity = bbox_is_valid(output_bbox)  # x1 < x2, y1 < y2 is valid
    output_bbox = torch.from_numpy(output_bbox).to(torch.long)
    input_bbox = torch.from_numpy(input_bbox).to(torch.long)

    assert output_bbox.ndim in (1, 2)

    if output_bbox.ndim == 2:
        assert output_bbox.shape[0] == n
        for i_n in range(n):
            if not output_bbox_validity[i_n]:
                continue
            output_img[i_n: i_n + 1, :, output_bbox[i_n, 1]: output_bbox[i_n, 3], output_bbox[i_n, 0]: output_bbox[i_n, 2]] = F.interpolate(
                img[i_n: i_n + 1, :, input_bbox[i_n, 1]: input_bbox[i_n, 3], input_bbox[i_n, 0]: input_bbox[i_n, 2]],
                (output_bbox[i_n, 3] - output_bbox[i_n, 1], output_bbox[i_n, 2] - output_bbox[i_n, 0]),
                mode=mode,
                align_corners=align_corners)
    else:
        if output_bbox_validity:
            for i_n in range(n):
                output_img[i_n: i_n + 1, :, output_bbox[1]: output_bbox[3], output_bbox[0]: output_bbox[2]] = F.interpolate(
                    img[i_n: i_n + 1, :, input_bbox[1]: input_bbox[3], input_bbox[0]: input_bbox[2]],
                    (output_bbox[3] - output_bbox[1], output_bbox[2] - output_bbox[0]),
                    mode=mode,
                    align_corners=align_corners)
    if not batch_mode:
        output_img = output_img.squeeze(0)

    if return_adjusted_params:
        input_bbox_float = input_bbox.float()
        output_bbox_float = output_bbox.float()

        real_scale = (output_bbox_float[..., 2:] - output_bbox_float[..., :2]) / (
                    input_bbox_float[..., 2:] - input_bbox_float[..., :2])
        real_translation = output_bbox_float[..., :2] - input_bbox_float[..., :2] * real_scale

        return output_img, real_scale.numpy(), real_translation.numpy()

    return output_img


# generally aligned with tf.raw_ops.ScaleAndTranslate. note that our input is in (w, h) format
# cost about 1.7x cpu times compared with the pixel aligned implementation (above)
def scale_and_translate_subpixel(image: torch.Tensor, size: Tuple[int, int],
                                 scale: Union[torch.Tensor, np.ndarray], translation: Union[torch.Tensor, np.ndarray],
                                 interpolation_mode: str = 'bilinear', align_corners: bool = False,
                                 padding_mode: str = 'zeros') -> torch.Tensor:
    # Ensure the image is a 4D tensor (batch, channels, height, width)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if isinstance(scale, np.ndarray):
        scale = torch.from_numpy(scale).to(torch.float32).to(image.device)
    if isinstance(translation, np.ndarray):
        translation = torch.from_numpy(translation).to(torch.float32).to(image.device)

    scale = scale.to(torch.float32)
    translation = translation.to(torch.float32)

    batch_size, channels, height, width = image.shape
    out_width, out_height = size
    scale = scale.view(-1, 1, 1, 2)
    translation = translation.view(-1, 1, 1, 2)

    x = torch.linspace(0, out_width / width, out_width, dtype=scale.dtype, device=scale.device)
    y = torch.linspace(0, out_height / height, out_height, dtype=scale.dtype, device=scale.device)

    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    grid.div_(scale)
    translation = torch.div(translation, scale)
    translation.select(-1, 0).div_(width)
    translation.select(-1, 1).div_(height)
    grid.sub_(translation)
    grid.mul_(2).sub_(1)

    # Perform the transformation using grid_sample
    transformed_image = F.grid_sample(image, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)

    return transformed_image.squeeze(0)  # Remove batch dimension if input was 3D
