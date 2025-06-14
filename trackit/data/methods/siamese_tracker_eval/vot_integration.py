import uuid
import numpy as np
import torch.utils.data

import trax

from trackit.core.third_party.vot.vot_integration import VOT, Rectangle
from trackit.datasets.base.operator.bbox.transform.compile import (compile_bbox_transform, BoundingBoxFormat,
                                                                   BoundingBoxCoordinateSystem)
from trackit.miscellanies.image.io import read_image_with_auto_retry
from trackit.miscellanies.image.pil_interop import from_pil_image, to_pil_image
from trackit.data.protocol.eval_input import TrackerEvalData, SequenceInfo
from trackit.data.protocol.eval_output import FrameEvaluationResult_SOT
from .transform import SiameseTrackerEval_DataTransform
from . import SiameseTrackerEvalDataWorker_Task, SiameseTrackerEvalDataWorker_FrameContext
from ... import MainProcessDataPipeline


class SiameseTrackerEvaluation_VOTToolkitIntegrator(torch.utils.data.dataset.IterableDataset, MainProcessDataPipeline):
    def __init__(self, vot: VOT, vot_region_format: trax.Region,
                 transform: SiameseTrackerEval_DataTransform):
        self._vot = vot
        self._vot_region_format = vot_region_format
        self._transform = transform

        self.vot_box_format_to_ours = compile_bbox_transform(
            BoundingBoxFormat.XYWH, BoundingBoxFormat.XYXY,
            BoundingBoxCoordinateSystem.Discrete, BoundingBoxCoordinateSystem.Continuous
        )
        self.ours_box_format_to_vot = compile_bbox_transform(
            BoundingBoxFormat.XYXY, BoundingBoxFormat.XYWH,
            BoundingBoxCoordinateSystem.Continuous, BoundingBoxCoordinateSystem.Discrete,
        )
        self._exhausted = False
        self._sequence_uuid = str(uuid.uuid1())

    def __iter__(self):
        self._index = 0
        assert not self._exhausted
        return self

    def __next__(self) -> TrackerEvalData:
        if self._exhausted:
            raise StopIteration

        template_image = None
        if self._index == 0:
            template_image = read_image_with_auto_retry(self._vot.frame())
            self._index += 1

        search_region_image = self._vot.frame()
        if search_region_image is not None:
            search_region_image = read_image_with_auto_retry(search_region_image)

        batch = []

        for index_of_object in range(len(self._vot.objects())):
            sequence_info = None
            init_context = None
            if template_image is not None:
                vot_init_region = self._vot.objects()[index_of_object]
                if self._vot_region_format == trax.Region.MASK:
                    template_box = _rect_from_mask(vot_init_region)
                    template_mask = vot_init_region.astype(bool)
                    template_mask = to_pil_image(template_mask)
                elif self._vot_region_format == trax.Region.RECTANGLE:
                    template_box = (vot_init_region.x, vot_init_region.y, vot_init_region.width, vot_init_region.height)
                    template_mask = None
                else:
                    raise NotImplementedError()
                template_box = np.array(self.vot_box_format_to_ours(template_box), dtype=np.float64)
                sequence_info = SequenceInfo('vot', None, 'vot',
                                             self._sequence_uuid + '-' + str(index_of_object), None, None)
                init_context = SiameseTrackerEvalDataWorker_FrameContext(self._index - 1,  # 0
                                                                         lambda: template_image,
                                                                         template_box,
                                                                         template_mask)

            if search_region_image is not None:
                do_task_finalization = False
                tracking_context = SiameseTrackerEvalDataWorker_FrameContext(self._index,
                                                                             lambda: search_region_image,
                                                                             None, None)
            else:
                do_task_finalization = True
                tracking_context = None
                self._exhausted = True

            worker_task = SiameseTrackerEvalDataWorker_Task(index_of_object, sequence_info, init_context,
                                                            tracking_context, do_task_finalization)
            data = self._transform(worker_task)
            batch.append(data)

        if search_region_image is not None:
            self._index += 1

        return TrackerEvalData(batch, {})

    def post_process(self, output_data: dict) -> dict:
        evaluated_frames = output_data['evaluated_frames']
        if len(evaluated_frames) == 0:
            assert self._exhausted, "Only if the data is exhausted the output can be empty."
        outputs = []
        for evaluated_frame in evaluated_frames:
            assert isinstance(evaluated_frame, FrameEvaluationResult_SOT)
            if evaluated_frame.output_mask is not None:
                output_mask = evaluated_frame.output_mask
                output_mask = from_pil_image(output_mask)
                output_mask = output_mask.astype(np.uint8)
                outputs.append(output_mask)
            else:
                assert isinstance(evaluated_frame.output_box, np.ndarray)
                box = self.ours_box_format_to_vot(evaluated_frame.output_box.tolist())
                outputs.append(Rectangle(*box))

        self._vot.report(outputs)
        return output_data


def _rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_)).item()
    x1 = np.max(np.nonzero(x_)).item()
    y0 = np.min(np.nonzero(y_)).item()
    y1 = np.max(np.nonzero(y_)).item()
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1
