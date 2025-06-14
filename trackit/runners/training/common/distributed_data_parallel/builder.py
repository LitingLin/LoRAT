from typing import Optional
import torch
import re

from trackit.miscellanies.torch.distributed import is_dist_initialized
from . import DistributedDataParallelOption


def _parse_params_and_buffers_to_ignore(params_and_buffers_to_ignore_rules, model_parameters):
    params_and_buffers_to_ignore = []
    for rule in params_and_buffers_to_ignore_rules:
        regex_matcher = None
        name = None
        if 'name_regex' in rule:
            regex_matcher = re.compile(rule['name_regex'])
        elif 'name' in rule:
            name = rule['name']
        else:
            raise RuntimeError('invalid rule')
        for model_parameter_name in list(model_parameters.keys()):
            if regex_matcher is not None:
                if (regex_matcher is not None and regex_matcher.search(model_parameter_name) is not None) or (
                        name is not None and model_parameter_name == name):
                    params_and_buffers_to_ignore.append(model_parameter_name)
                    model_parameters.pop(model_parameter_name)
    if len(params_and_buffers_to_ignore) == 0:
        raise RuntimeError('no param or buffer matched')
    return params_and_buffers_to_ignore


def get_distributed_data_parallel_option(config: dict, model: torch.nn.Module, criterion: Optional[torch.nn.Module],
                                         device: torch.device,
                                         grad_accumulation_steps: int
                                         ) -> Optional[DistributedDataParallelOption]:
    if not is_dist_initialized():
        return None
    is_cuda = device.type == 'cuda'
    enable_sync_bn = is_cuda
    find_unused_parameters = False
    gradient_as_bucket_view = True
    static_graph = True
    params_and_buffers_to_ignore = None
    criterion_params_and_buffers_to_ignore = None
    if 'distributed_data_parallel' in config:
        distributed_data_parallel_config = config['distributed_data_parallel']
        if 'sync_batchnorm' in distributed_data_parallel_config and is_cuda:
            enable_sync_bn = distributed_data_parallel_config['sync_batchnorm']
        if 'gradient_as_bucket_view' in distributed_data_parallel_config:
            gradient_as_bucket_view = distributed_data_parallel_config['gradient_as_bucket_view']
        if 'static_graph' in distributed_data_parallel_config:
            static_graph = distributed_data_parallel_config['static_graph']
        if 'find_unused_parameters' in distributed_data_parallel_config:
            find_unused_parameters = distributed_data_parallel_config['find_unused_parameters']
        if 'params_and_buffers_to_ignore' in distributed_data_parallel_config:
            params_and_buffers_to_ignore_rules = distributed_data_parallel_config['params_and_buffers_to_ignore']
            model_parameters = dict(model.named_parameters())
            params_and_buffers_to_ignore = _parse_params_and_buffers_to_ignore(params_and_buffers_to_ignore_rules, model_parameters)
        if 'criterion_params_and_buffers_to_ignore' in distributed_data_parallel_config:
            criterion_params_and_buffers_to_ignore_rules = distributed_data_parallel_config['criterion_params_and_buffers_to_ignore']
            assert criterion is not None, "criterion is None but config criterion_params_and_buffers_to_ignore has been set"

            criterion_parameters = dict(criterion.named_parameters())
            assert len(criterion_parameters) > 0, "criterion has no parameters but config criterion_params_and_buffers_to_ignore has been set"
            criterion_params_and_buffers_to_ignore = _parse_params_and_buffers_to_ignore(criterion_params_and_buffers_to_ignore_rules, criterion_parameters)

    if grad_accumulation_steps > 1:
        static_graph = False

    return DistributedDataParallelOption(find_unused_parameters, gradient_as_bucket_view, static_graph, enable_sync_bn,
                                         params_and_buffers_to_ignore, criterion_params_and_buffers_to_ignore)
