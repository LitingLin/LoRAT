import torch
import numpy as np

from trackit.data.protocol.eval_input import TrackerEvalData_TaskDesc, TrackerEvalData_FrameData
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.miscellanies.image.pil_interop import from_pil_image

from .. import SiameseTrackerEvalDataWorker_Task
from . import SiameseTrackerEval_DataTransform


class SiameseTrackerEval_PlainDataTransform(SiameseTrackerEval_DataTransform):
    def __init__(self, norm_stats_dataset_name: str,
                 device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        self.image_normalize_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.device = device
        self.dtype = dtype

    def __call__(self, task: SiameseTrackerEvalDataWorker_Task) -> TrackerEvalData_TaskDesc:
        init_frame_data = None
        if task.do_tracker_init is not None:
            z = task.do_tracker_init.get_image()
            z = torch.from_numpy(z)
            z = torch.permute(z, (2, 0, 1))
            z = z.to(self.device).to(torch.float32)
            z.div_(255.)
            self.image_normalize_transform_(z)
            z = z.to(self.dtype)

            input_data = {'image': z}
            image_height, image_width = z.shape[1:]
            if task.do_tracker_init.gt_mask is not None:
                mask = torch.from_numpy(from_pil_image(task.do_tracker_init.gt_mask))
                input_data['mask'] = mask
            else:
                z_bbox = bbox_clip_to_image_boundary(task.do_tracker_init.gt_bbox, np.array((image_width, image_height)))
                z_bbox = bbox_rasterize(z_bbox)
                z_bbox = torch.from_numpy(z_bbox)
                input_data['bbox'] = z_bbox

            init_frame_data = TrackerEvalData_FrameData(task.do_tracker_init.frame_index,
                                                        task.do_tracker_init.gt_bbox,
                                                        task.do_tracker_init.gt_mask,
                                                        input_data)

        track_frame_data = None
        if task.do_tracker_track is not None:
            x = task.do_tracker_track.get_image()
            x = torch.from_numpy(x)
            x = torch.permute(x, (2, 0, 1))
            x = x.to(self.device).to(torch.float32)
            x.div_(255.)
            self.image_normalize_transform_(x)
            x = x.to(self.dtype)
            track_frame_data = TrackerEvalData_FrameData(task.do_tracker_track.frame_index,
                                                         task.do_tracker_track.gt_bbox,
                                                         task.do_tracker_track.gt_mask,
                                                         {'image': x})
        return TrackerEvalData_TaskDesc(task.task_index, task.do_task_creation,
                                        init_frame_data, track_frame_data,
                                        task.do_task_finalization)
