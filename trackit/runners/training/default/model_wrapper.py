import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from torch.nn.parallel import DistributedDataParallel
from trackit.models.schema.input.auto_unpack import auto_unpack_and_call
from trackit.criteria import CriterionOutput
from ..common.distributed_data_parallel import DistributedDataParallelOption
from .utils import criterion_has_parameters
from ..common.torch_compile import TorchCompileOptions


class ModelWithCriterion(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithCriterion, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, samples, targets) -> Tuple[Any, CriterionOutput]:
        output = auto_unpack_and_call(samples, self.model)
        criterion_output = self.criterion(output, targets)
        return output, criterion_output


def _compile_model(wrapped_model: ModelWithCriterion, torch_compile_options: TorchCompileOptions,
                   in_computational_graph_criterion: bool):
    if not torch_compile_options.enabled:
        return wrapped_model
    torch_compile_parameters = torch_compile_options.parameters
    if torch_compile_parameters is None:
        torch_compile_parameters = {}
    if in_computational_graph_criterion:
        wrapped_model = torch.compile(wrapped_model, **torch_compile_parameters)
    else:
        wrapped_model.model = torch.compile(wrapped_model.model, **torch_compile_parameters)
    message = 'torch.compile is enabled'
    if in_computational_graph_criterion:
        message += ' with criterion'
    message += '.'
    if len(torch_compile_parameters) > 0:
        message += f' parameters: {torch_compile_parameters}.'
    print(message)
    return wrapped_model


def build_model_wrapper(model: nn.Module, criterion: nn.Module,
                        distributed_data_parallel_option: Optional[DistributedDataParallelOption],
                        torch_compile_options: TorchCompileOptions):
    ddp_no_sync_fn = None
    in_computational_graph_criterion = criterion_has_parameters(criterion)
    wrapped_model = ModelWithCriterion(model, criterion)
    
    if distributed_data_parallel_option is not None:
        if distributed_data_parallel_option.convert_sync_batchnorm:
            if in_computational_graph_criterion:
                wrapped_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wrapped_model)
            else:
                wrapped_model.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed_data_parallel_option is not None:
        if torch_compile_options.compiled_autograd.enabled:
            if torch._dynamo.config.optimize_ddp in (True, "ddp_optimizer"):
                print('workaround: compiled autograd is enabled with DDP. applied torch._dynamo.config.optimize_ddp = "python_reducer"')
                torch._dynamo.config.optimize_ddp = "python_reducer"
        
        if distributed_data_parallel_option.model_params_and_buffers_to_ignore is not None:
            DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                model,
                distributed_data_parallel_option.model_params_and_buffers_to_ignore)
        if distributed_data_parallel_option.criterion_params_and_buffers_to_ignore is not None and in_computational_graph_criterion:
            DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                criterion,
                distributed_data_parallel_option.criterion_params_and_buffers_to_ignore)
        if in_computational_graph_criterion:
            ddp_module = wrapped_model
        else:
            ddp_module = wrapped_model.model
        ddp_module = DistributedDataParallel(ddp_module,
                                                find_unused_parameters=distributed_data_parallel_option.find_unused_parameters,
                                                gradient_as_bucket_view=distributed_data_parallel_option.gradient_as_bucket_view,
                                                static_graph=distributed_data_parallel_option.static_graph,
                                                bucket_cap_mb=100)
        ddp_no_sync_fn = ddp_module.no_sync
        if in_computational_graph_criterion:
            wrapped_model = ddp_module
        else:
            wrapped_model.model = ddp_module
        print('DistributedDataParallel is enabled.')

    wrapped_model = _compile_model(wrapped_model, torch_compile_options, in_computational_graph_criterion)
    return wrapped_model, ddp_no_sync_fn


def build_model_wrapper_eval(model: nn.Module, criterion: nn.Module):
    return ModelWithCriterion(model, criterion)
