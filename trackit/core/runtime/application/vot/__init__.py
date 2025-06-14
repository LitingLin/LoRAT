import os
import uuid
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import trax
import torch.amp
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.third_party.vot.vot_integration import VOT
from trackit.miscellanies.image.io import read_image_with_auto_retry
from trackit.miscellanies.image.pil_interop import to_pil_image


class VOTTest:
    def __init__(self):
        from trackit.core.runtime.global_constant import get_global_constant
        self.path = get_global_constant('VOTS2023_PATH')
        self.path = os.path.join(self.path, 'bear-6', 'color')
        self.images = os.listdir(self.path)
        self.images = [image for image in self.images if image.endswith('.jpg')]
        self.index = 0

    def frame(self):
        if self.index >= len(self.images):
            return None
        path = os.path.join(self.path, self.images[self.index])
        self.index += 1
        return path

    def objects(self):
        return [np.zeros((255, 255), dtype=np.uint8), np.ones((255, 255), dtype=np.uint8)]

    def report(self, _):
        pass

    def quit(self):
        pass


class VOTEvaluationApplication:
    def __init__(self, model: nn.Module, stats_transform_name: str, device: torch.device,
                 dtype: torch.dtype,
                 auto_mixed_precision_dtype: Optional[torch.dtype] = None,
                 visualize: bool = False,
                 output_dir: Optional[str] = None,
                 test=False):
        self.model = model
        self.model.eval()
        self.norm_stats_transform = get_dataset_norm_stats_transform(stats_transform_name, inplace=True)
        self.device = device
        self.dtype = dtype
        self.auto_mixed_precision_dtype = auto_mixed_precision_dtype
        self.visualize = visualize
        self.output_dir = output_dir
        self.test = test

    def run(self):
        run_uuid = uuid.uuid1()
        output_dir = None
        if self.output_dir is not None and self.visualize:
            output_dir = os.path.join(self.output_dir, str(run_uuid))
            os.makedirs(output_dir, exist_ok=False)

        if self.test:
            vot = VOTTest()
        else:
            vot = VOT(trax.Region.MASK, multiobject=True)

        # do init
        with torch.amp.autocast(self.device.type,
                                enabled=self.auto_mixed_precision_dtype is not None,
                                dtype=self.auto_mixed_precision_dtype):
            self.model(**init(vot, self.device, self.dtype, self.norm_stats_transform, output_dir))

        frame_index = 1
        # do tracking
        while True:
            track_kwargs = track(vot, self.device, self.dtype, self.norm_stats_transform)
            if track_kwargs is None:
                break
            with torch.amp.autocast(self.device.type,
                                    enabled=self.auto_mixed_precision_dtype is not None,
                                    dtype=self.auto_mixed_precision_dtype):
                object_ids, predicted_masks = self.model(**track_kwargs)
            predicted_masks = predicted_masks.cpu().to(torch.float32).numpy()
            for index, object_id in enumerate(object_ids):
                assert index == object_id, f'object_id {object_id} does not match index {index}'
                if self.visualize:
                    mask_path = os.path.join(output_dir, f'mask_frame_{frame_index}_obj_{object_id}.png')
                    to_pil_image(predicted_masks[index]).save(mask_path)
            predicted_masks = predicted_masks.astype(np.uint8)
            assert predicted_masks.ndim == 3, f'predicted_masks.ndim {predicted_masks.ndim} != 3'
            predicted_masks = list(mask for mask in predicted_masks)
            vot.report(predicted_masks)
            frame_index += 1


def _get_number_of_frames(image_path: str):
    image_file_names = os.listdir(os.path.dirname(image_path))
    image_file_names = [x for x in image_file_names if x.endswith(('.jpg', '.png', '.jpeg'))]
    assert len(image_file_names) > 0, f'No image files found in {os.path.dirname(image_path)}'
    return len(image_file_names)


def init(vot, device: torch.device, dtype: torch.dtype, transform_, output_dir: Optional[str]):
    frame_path = vot.frame()
    number_of_frames = _get_number_of_frames(frame_path)
    image = read_image_with_auto_retry(frame_path)
    image = torch.from_numpy(image).permute(2, 0, 1).to(device).to(torch.float32)
    image.div_(255.0)
    video_height, video_width = image.shape[-2:]
    assert image.ndim == 3, f'image.ndim {image.ndim} != 3'
    assert image.shape[0] == 3, f'image.shape[0] {image.shape[0]} != 3'
    transform_(image)
    image = image.to(dtype)
    objects = []
    for index, object_mask in enumerate(vot.objects()):
        object_mask = object_mask.astype(bool)
        assert object_mask.ndim == 2, f'object_mask.ndim {object_mask.ndim} != 2'
        assert object_mask.shape[0] <= video_height and object_mask.shape[1] <= video_width, \
            f'{object_mask.shape} vs {(video_height, video_width)}'
        object_mask = np.pad(object_mask, ((0, video_height - object_mask.shape[0]),
                                           (0, video_width - object_mask.shape[1])), mode='constant')

        objects.append({'object_id': index, 'mask': object_mask})
    if output_dir is not None:
        for curr_object in objects:
            mask = curr_object['mask']
            mask_path = os.path.join(output_dir, f'mask_frame_0_obj_{curr_object["object_id"]}.png')
            to_pil_image(mask).save(mask_path)

    return {'action': 'init', 'frame': image, 'context': {'total_num_frames': number_of_frames,
                                                          'task_id': 0,
                                                          'objects': objects}}


def track(vot, device: torch.device, dtype: torch.dtype, transform_):
    frame_path = vot.frame()
    if frame_path is None:
        return None
    image = read_image_with_auto_retry(frame_path)
    image = torch.from_numpy(image).permute(2, 0, 1).to(device).to(torch.float32)
    image.div_(255.0)
    transform_(image)
    image = image.to(dtype)
    return {'action': 'track', 'frame': image, 'context': {'task_id': 0}}
