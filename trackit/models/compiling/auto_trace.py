import torch
import torch.nn as nn
from typing import Dict, Any
from trackit.models.schema.conversion.trace_friendly import TraceFriendlyDataAdaptor, get_trace_friendly_data_adaptor
from trackit.models.schema.input.auto_unpack import auto_unpack_and_call
from trackit.models import ModelInputDataSelfDescriptionMixin, ModelInputDataSelfDescriptionMixin_MultiPath


class ExternalInferenceEngineWrapper:
    def __init__(self, module, input_data_adaptor: TraceFriendlyDataAdaptor, output_data_adaptor: TraceFriendlyDataAdaptor):
        self.module = module
        self.input_data_adaptor = input_data_adaptor
        self.output_data_adaptor = output_data_adaptor

    def get_input_name(self):
        return self.input_data_adaptor.get_data_names()

    def get_output_name(self):
        return self.output_data_adaptor.get_data_names()

    def __call__(self, data):
        return self.output_data_adaptor.restore(self.module(*self.input_data_adaptor.flatten(data)))


class ExternalInferenceEngineWrapper_MultiPath:
    def __init__(self, modules: Dict[str, Any], common_input_data_flatten_fn,
                 input_data_adaptors: Dict[str, TraceFriendlyDataAdaptor],
                 output_data_adaptors: Dict[str, TraceFriendlyDataAdaptor],
                 input_pattern_path_name_mapping: Dict[Any, str]):
        self.modules = modules
        self.common_input_data_flatten_fn = common_input_data_flatten_fn
        self.input_data_adaptors = input_data_adaptors
        self.output_data_adaptors = output_data_adaptors
        self.input_pattern_path_name_mapping = input_pattern_path_name_mapping

    def __call__(self, data):
        flattened_input_data, data_pattern = self.common_input_data_flatten_fn(data)
        path_name = self.input_pattern_path_name_mapping[data_pattern]
        module = self.modules[path_name]
        flattened_output_data = module(*flattened_input_data)
        output_data_adaptor = self.output_data_adaptors[path_name]
        return output_data_adaptor.restore(flattened_output_data)

    def get_input_name(self, data):
        _, data_pattern = self.common_input_data_flatten_fn(data)
        return self.input_data_adaptors[self.input_pattern_path_name_mapping[data_pattern]].get_data_names()

    def get_output_name(self, data):
        _, data_pattern = self.common_input_data_flatten_fn(data)
        return self.output_data_adaptors[self.input_pattern_path_name_mapping[data_pattern]].get_data_names()

    def get_all_path_name(self):
        return self.modules.keys()


class TraceFriendlyModuleAdaptor(nn.Module):
    def __init__(self, module: nn.Module, input_data_adaptor, output_data_adaptor):
        super(TraceFriendlyModuleAdaptor, self).__init__()
        self.module = module
        self.input_data_adaptor = input_data_adaptor
        self.output_data_adaptor = output_data_adaptor

    def forward(self, *args: torch.Tensor):
        restored_input_data = self.input_data_adaptor.restore(args)
        output = auto_unpack_and_call(restored_input_data, self.module)
        return self.output_data_adaptor.flatten(output)


def enable_inference_engine_by_auto_tracing(module: nn.Module,
                                            trace_fn, max_batch_size, device: torch.device, dtype: torch.dtype):
    if isinstance(module, ModelInputDataSelfDescriptionMixin):
        input_data = module.get_sample_data(max_batch_size, device, dtype, None)
        input_data_adaptor, flattened_input_data = get_trace_friendly_data_adaptor(input_data)

        output_data = auto_unpack_and_call(input_data, module)
        output_data_adaptor, _ = get_trace_friendly_data_adaptor(output_data)

        trace_friendly_module = TraceFriendlyModuleAdaptor(module, input_data_adaptor, output_data_adaptor)
        traced_module = trace_fn(trace_friendly_module, flattened_input_data, input_data_adaptor.get_data_names(), output_data_adaptor.get_data_names())
        return ExternalInferenceEngineWrapper(traced_module, input_data_adaptor, output_data_adaptor)
    elif isinstance(module, ModelInputDataSelfDescriptionMixin_MultiPath):
        common_input_data_adaptor_fn = None
        pattern_name_mapping = {}

        traced_modules = {}
        input_data_adaptors = {}
        output_data_adaptors = {}
        for path_name in module.get_data_path_names(with_train=False):
            input_data = module.get_sample_data(path_name, max_batch_size, device, dtype, None)
            output_data = auto_unpack_and_call(input_data, module)

            input_data_adaptor, flattened_input_data = get_trace_friendly_data_adaptor(input_data)
            output_data_adaptor, _ = get_trace_friendly_data_adaptor(output_data)

            if common_input_data_adaptor_fn is None:
                common_input_data_adaptor_fn = input_data_adaptor.flatten_fn
            else:
                assert common_input_data_adaptor_fn == input_data_adaptor.flatten_fn, "Auto compiling for multipath module needs all paths have same input data schema"
            assert input_data_adaptor.restore_context not in pattern_name_mapping, "Auto compiling for multipath module needs input data have different pattern"
            pattern_name_mapping[input_data_adaptor.restore_context] = path_name

            trace_friendly_module = TraceFriendlyModuleAdaptor(module, input_data_adaptor, output_data_adaptor)

            traced_modules[path_name] = trace_fn(trace_friendly_module, flattened_input_data, input_data_adaptor.get_data_names(), output_data_adaptor.get_data_names())
            input_data_adaptors[path_name] = input_data_adaptor
            output_data_adaptors[path_name] = output_data_adaptor
        return ExternalInferenceEngineWrapper_MultiPath(traced_modules, common_input_data_adaptor_fn, input_data_adaptors, output_data_adaptors, pattern_name_mapping)
    else:
        raise TypeError(f"Module must implement ModelInputDataSelfDescriptionMixin or ModelInputDataSelfDescriptionMixin_MultiPath.")
