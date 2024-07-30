from trackit.data.protocol.eval_input import TrackerEvalData_TaskDesc
from .. import SiameseTrackerEvalDataWorker_Task


class SiameseTrackerEval_DataTransform:
    def __call__(self, task: SiameseTrackerEvalDataWorker_Task) -> TrackerEvalData_TaskDesc:
        raise NotImplementedError()
