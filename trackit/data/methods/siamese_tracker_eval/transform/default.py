import torch
import numpy as np
from typing import Tuple

from trackit.data.protocol.eval_input import TrackerEvalData_TaskDesc, TrackerEvalData_FrameData
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform

from .. import SiameseTrackerEvalDataWorker_Task
from . import SiameseTrackerEval_DataTransform


class SiameseTrackerEval_DefaultDataTransform(SiameseTrackerEval_DataTransform):
    def __init__(self, template_size: Tuple[int, int], template_area_factor: float,
                 with_full_template_image: bool,
                 interpolation_mode: str, interpolation_align_corners: bool,
                 norm_stats_dataset_name: str, device: torch.device = torch.device('cpu')):
        self.template_size = np.array(template_size)
        self.template_area_factor = template_area_factor
        self.with_full_template_image = with_full_template_image
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.image_normalize_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.device = device

    def __call__(self, task: SiameseTrackerEvalDataWorker_Task) -> TrackerEvalData_TaskDesc:
        init_frame_data = None
        if task.do_tracker_init is not None:
            z = task.do_tracker_init.get_image()
            z = torch.from_numpy(z)
            z = torch.permute(z, (2, 0, 1))
            z = z.to(self.device)

            z_bbox = task.do_tracker_init.gt_bbox

            template_curation_parameter = get_siamfc_cropping_params(z_bbox, self.template_area_factor, self.template_size)

            z_curated, z_image_mean, template_curation_parameter = apply_siamfc_cropping(
                z.to(torch.float32), self.template_size, template_curation_parameter,
                self.interpolation_mode, self.interpolation_align_corners)

            z_curated.div_(255.)
            self.image_normalize_transform_(z_curated)

            input_data = {'curated_image': z_curated, 'image_mean': z_image_mean,
                          'curation_parameter': template_curation_parameter}
            if self.with_full_template_image:
                input_data['image'] = z
            init_frame_data = TrackerEvalData_FrameData(task.do_tracker_init.frame_index, z_bbox, None, input_data)

        track_frame_data = None
        if task.do_tracker_track is not None:
            x = task.do_tracker_track.get_image()
            x = torch.from_numpy(x)
            x = torch.permute(x, (2, 0, 1))
            x = x.to(self.device)
            track_frame_data = TrackerEvalData_FrameData(task.do_tracker_track.frame_index,
                                                         task.do_tracker_track.gt_bbox,
                                                         None, {'image': x})
        return TrackerEvalData_TaskDesc(task.task_index, task.do_task_creation,
                                        init_frame_data, track_frame_data,
                                        task.do_task_finalization)
