import torch
import torch.nn as nn
from typing import Optional, Any
from torch.nn.parallel import DistributedDataParallel
from trackit.models.schema.input.auto_unpack import auto_unpack_and_call
from ..common.distributed_data_parallel import DistributedDataParallelOption
from .utils import criterion_has_parameters


class ModelWithCriterion(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithCriterion, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, samples, targets) -> Any:
        output = auto_unpack_and_call(samples, self.model)
        output = self.criterion(output, targets)
        return output


def build_model_wrapper(model: nn.Module, criterion: nn.Module,
                        distributed_data_parallel_option: Optional[DistributedDataParallelOption],
                        torch_compile_options: Optional[dict]):
    in_computational_graph_criterion = criterion_has_parameters(criterion)
    wrapped_model = ModelWithCriterion(model, criterion)

    if distributed_data_parallel_option is not None:
        if distributed_data_parallel_option.convert_sync_batchnorm:
            if in_computational_graph_criterion:
                wrapped_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wrapped_model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                wrapped_model.model = model
        if distributed_data_parallel_option.model_params_and_buffers_to_ignore is not None or distributed_data_parallel_option.criterion_params_and_buffers_to_ignore is not None:
            params_and_buffer_to_ignore = []
            if distributed_data_parallel_option.model_params_and_buffers_to_ignore is not None:
                if in_computational_graph_criterion:
                    params_and_buffer_to_ignore.extend('model.' + name for name in
                                                       distributed_data_parallel_option.model_params_and_buffers_to_ignore)
                else:
                    params_and_buffer_to_ignore.extend(name for name in
                                                       distributed_data_parallel_option.model_params_and_buffers_to_ignore)
            if distributed_data_parallel_option.criterion_params_and_buffers_to_ignore is not None:
                params_and_buffer_to_ignore.extend('criterion.' + name for name in
                                                   distributed_data_parallel_option.criterion_params_and_buffers_to_ignore)
            if in_computational_graph_criterion:
                DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(wrapped_model,
                                                                                    params_and_buffer_to_ignore)
            else:
                DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model,
                                                                                    params_and_buffer_to_ignore)
        if in_computational_graph_criterion:
            wrapped_model = DistributedDataParallel(wrapped_model,
                                                    find_unused_parameters=distributed_data_parallel_option.find_unused_parameters,
                                                    gradient_as_bucket_view=distributed_data_parallel_option.gradient_as_bucket_view,
                                                    static_graph=distributed_data_parallel_option.static_graph)
        else:
            model = DistributedDataParallel(model,
                                            find_unused_parameters=distributed_data_parallel_option.find_unused_parameters,
                                            gradient_as_bucket_view=distributed_data_parallel_option.gradient_as_bucket_view,
                                            static_graph=distributed_data_parallel_option.static_graph)
            wrapped_model.model = model
        print('DistributedDataParallel is enabled.')

    if torch_compile_options is not None:
        if in_computational_graph_criterion:
            wrapped_model = torch.compile(wrapped_model, **torch_compile_options)
        else:
            model = torch.compile(model, **torch_compile_options)
            wrapped_model.model = model
        message = 'torch.compile is enabled'
        if in_computational_graph_criterion:
            message += ' with criterion'
        message += '.'
        if len(torch_compile_options) > 0:
            message += f' parameters: {torch_compile_options}.'
        print(message)
    return wrapped_model


def build_model_wrapper_eval(model: nn.Module, criterion: nn.Module):
    return ModelWithCriterion(model, criterion)
