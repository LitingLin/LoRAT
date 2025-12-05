from trackit.data import MainProcessDataPipeline
from trackit.data.protocol.train_input import TrainData


class AddParam_MainProcessDataPipeline(MainProcessDataPipeline):
    def __init__(self, params):
        self.params = params

    def pre_process(self, input_data: TrainData) -> TrainData:
        input_data.input.update(self.params)
        return input_data


def build_add_param_data_pipeline(pipeline_config: dict):
    return None, None, AddParam_MainProcessDataPipeline(pipeline_config['parameters'])
