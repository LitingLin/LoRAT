from typing import Any, Iterable, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from trackit.models import (ModelManager,
                            ModelInputDataSelfDescriptionMixin, ModelInputDataSelfDescriptionMixin_MultiPath)

from . import InferencePrecision


@dataclass(frozen=True)
class ModelExecutionPath:
    is_train: bool
    path: str
    model: nn.Module
    sample_input: Any
    auto_mixed_precision_dtype: Optional[torch.dtype]


def iterate_model_execution_paths(model_manager: ModelManager, batch_size: int,
                                  device: torch.device,
                                  inference_precision: InferencePrecision = InferencePrecision(),
                                  train_mode_inference_precision: Optional[InferencePrecision] = None,
                                  torch_jit_trace_compatible: bool = False) -> Iterable[ModelExecutionPath]:
    if train_mode_inference_precision is None:
        train_mode_inference_precision = inference_precision
    train_mode_dtype = train_mode_inference_precision.dtype
    train_mode_amp_dtype = train_mode_inference_precision.auto_mixed_precision_dtype if train_mode_inference_precision.auto_mixed_precision_enabled else None
    eval_mode_dtype = inference_precision.dtype
    eval_mode_amp_dtype = inference_precision.auto_mixed_precision_dtype if inference_precision.auto_mixed_precision_enabled else None

    train_model_impl_suggestions = {
        'dtype': train_mode_dtype,
        'torch_jit_trace_compatible': torch_jit_trace_compatible,
        'optimize_for_inference': False,
        'load_pretrained': False}
    eval_model_impl_suggestions = {
        'dtype': eval_mode_dtype,
        'torch_jit_trace_compatible': torch_jit_trace_compatible,
        'optimize_for_inference': True,
        'load_pretrained': False}

    train_model = model_manager.create(device, **train_model_impl_suggestions)
    if isinstance(train_model.model, ModelInputDataSelfDescriptionMixin):
        if train_model.fingerprint_string != model_manager.get_fingerprint_string(device, **eval_model_impl_suggestions):
            yield ModelExecutionPath(True, 'train', train_model.model.train(),
                                     train_model.model.get_sample_data(batch_size, device,
                                                                       train_mode_dtype,
                                                                       train_mode_amp_dtype),
                                     train_mode_amp_dtype)
            del train_model
            eval_model = model_manager.create(device, **eval_model_impl_suggestions)
            assert isinstance(eval_model.model, (ModelInputDataSelfDescriptionMixin, ModelInputDataSelfDescriptionMixin_MultiPath))
            if isinstance(eval_model.model, ModelInputDataSelfDescriptionMixin):
                yield ModelExecutionPath(False, 'eval', eval_model.model.eval(),
                                         eval_model.model.get_sample_data(batch_size, device,
                                                                          eval_mode_dtype,
                                                                          eval_mode_amp_dtype),
                                         eval_mode_amp_dtype)
            elif isinstance(eval_model.model, ModelInputDataSelfDescriptionMixin_MultiPath):
                for path in eval_model.model.get_data_path_names(with_train=False, with_eval=True):
                    yield ModelExecutionPath(False, '{path}-eval', eval_model.model,
                                             eval_model.model.get_sample_data(path, batch_size,
                                                                              device, eval_mode_dtype,
                                                                              eval_mode_amp_dtype),
                                             eval_mode_amp_dtype)
            else:
                raise NotImplementedError
        else:
            yield ModelExecutionPath(False, '', train_model.model.eval(),
                                     train_model.model.get_sample_data(batch_size, device,
                                                                       eval_mode_dtype,
                                                                       eval_mode_amp_dtype),
                                     eval_mode_amp_dtype)
    elif isinstance(train_model.model, ModelInputDataSelfDescriptionMixin_MultiPath):
        train_paths = train_model.model.get_data_path_names(with_train=True, with_eval=False)
        diff_implement = train_model.fingerprint_string != model_manager.get_fingerprint_string(device, **eval_model_impl_suggestions)
        for train_path in train_paths:
            yield ModelExecutionPath(True, train_path + '-train' if diff_implement else train_path,
                                     train_model.model,
                                     train_model.model.get_sample_data(train_path, batch_size, device,
                                                                       train_mode_dtype,
                                                                       train_mode_amp_dtype),
                                     train_mode_amp_dtype)
        if diff_implement:
            del train_model
            eval_model = model_manager.create(device, **eval_model_impl_suggestions)
        else:
            eval_model = train_model
        if isinstance(eval_model.model, ModelInputDataSelfDescriptionMixin):
            eval_paths = []
        elif isinstance(eval_model.model, ModelInputDataSelfDescriptionMixin_MultiPath):
            eval_paths = eval_model.model.get_data_path_names(with_train=False, with_eval=True)
        else:
            raise NotImplementedError
        if len(eval_paths) > 0:
            for eval_path in eval_paths:
                yield ModelExecutionPath(False, eval_path + '-eval' if diff_implement else eval_path,
                                         eval_model.model,
                                         eval_model.model.get_sample_data(eval_path, batch_size, device,
                                                                          eval_mode_dtype, eval_mode_amp_dtype),
                                         eval_mode_amp_dtype)
        else:
            yield ModelExecutionPath(False, 'eval', eval_model.model.eval(),
                                     eval_model.model.get_sample_data(batch_size, device,
                                                                      eval_mode_dtype, eval_mode_amp_dtype),
                                     eval_mode_amp_dtype)
    else:
        raise NotImplementedError
