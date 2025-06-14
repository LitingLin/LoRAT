from typing import Optional

import torch
import torch.nn as nn
import time
from trackit.models import ModelManager
from trackit.models.schema.input.auto_unpack import auto_unpack_and_call
from trackit.miscellanies.torch.amp_autocast import get_torch_amp_autocast_fn
from ._execution_path_iterator import iterate_model_execution_paths
from . import InferencePrecision


def _test_model_latency(model: nn.Module, data,
                        loops: int, warmup_loops: int,
                        device: torch.device,
                        auto_mixed_precision_dtype: Optional[torch.dtype] = None):
    autocast_fn = get_torch_amp_autocast_fn(device.type, auto_mixed_precision_dtype is not None, auto_mixed_precision_dtype)
    if warmup_loops > 0:
        with torch.inference_mode(), autocast_fn():
            for _ in range(warmup_loops):
                auto_unpack_and_call(data, model)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    begin = time.perf_counter()
    with torch.inference_mode(), autocast_fn():
        for _ in range(loops):
            auto_unpack_and_call(data, model)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - begin) / loops * 1000


def _test_model_latency_cuda(model: nn.Module, data,
                             loops: int, warmup_loops: int,
                             auto_mixed_precision_dtype: Optional[torch.dtype] = None):
    autocast_fn = get_torch_amp_autocast_fn('cuda',
                                            auto_mixed_precision_dtype is not None, auto_mixed_precision_dtype)
    if warmup_loops > 0:
        with torch.inference_mode(), autocast_fn():
            for _ in range(warmup_loops):
                auto_unpack_and_call(data, model)
    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.inference_mode(), autocast_fn():
        for _ in range(loops):
            start_event.record()
            auto_unpack_and_call(data, model)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
    return sum(timings) / loops


def get_model_latency_for_all_paths(model_manager: ModelManager,
                                    device: torch.device,
                                    inference_precision: InferencePrecision,
                                    train_mode_inference_precision: Optional[InferencePrecision] = None,
                                    batch_size: int = 1, loops: int = 100, warmup_loops: int = 10):
    for execution_path in iterate_model_execution_paths(model_manager, batch_size, device,
                                                        inference_precision,
                                                        train_mode_inference_precision):
        if device.type == 'cuda':
            model_latency = _test_model_latency_cuda(execution_path.model, execution_path.sample_input,
                                                     loops, warmup_loops, execution_path.auto_mixed_precision_dtype)
        else:
            model_latency = _test_model_latency(execution_path.model, execution_path.sample_input,
                                                loops, warmup_loops,
                                                device, execution_path.auto_mixed_precision_dtype)
        yield execution_path.path, model_latency
