from contextlib import contextmanager
from typing import Any, Iterable, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from trackit.models import (ModelManager,
                            ModelInputDataSelfDescriptionMixin, ModelInputDataSelfDescriptionMixin_MultiPath,
                            ModelCacheSelfContainedMixin)

from . import InferencePrecision


@dataclass(frozen=True)
class ModelExecutionPath:
    is_train: bool
    path: str
    model: nn.Module
    sample_input: Any
    auto_mixed_precision_dtype: Optional[torch.dtype]


@contextmanager
def _managed_model_cache(model: nn.Module, batch_size: int, dtype: torch.dtype, amp_dtype: Optional[torch.dtype]):
    """A context manager to safely allocate and destroy the model cache."""
    if isinstance(model, ModelCacheSelfContainedMixin):
        model.allocate_cache(batch_size, batch_size, batch_size, dtype, amp_dtype)
        try:
            yield
        finally:
            model.destroy_cache()
    else:
        yield


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

    has_separate_eval_impl = (
        model_manager.get_fingerprint_string(device, **train_model_impl_suggestions) !=
        model_manager.get_fingerprint_string(device, **eval_model_impl_suggestions)
    )


    if has_separate_eval_impl:
        # --- Training Paths ---
        train_model_wrapper = model_manager.create(device, **train_model_impl_suggestions)
        model = train_model_wrapper.model
        with _managed_model_cache(model, batch_size, train_mode_dtype, train_mode_amp_dtype):
            model.train()
            if isinstance(model, ModelInputDataSelfDescriptionMixin_MultiPath):
                for path in model.get_data_path_names(with_train=True, with_eval=False):
                    yield ModelExecutionPath(
                        is_train=True, path=f"{path}-train", model=model,
                        sample_input=model.get_sample_data(path, batch_size, device, train_mode_dtype, train_mode_amp_dtype),
                        auto_mixed_precision_dtype=train_mode_amp_dtype)
            elif isinstance(model, ModelInputDataSelfDescriptionMixin):
                yield ModelExecutionPath(
                    is_train=True, path='train', model=model,
                    sample_input=model.get_sample_data(batch_size, device, train_mode_dtype, train_mode_amp_dtype),
                    auto_mixed_precision_dtype=train_mode_amp_dtype)
            else:
                raise NotImplementedError("model must implement ModelInputDataSelfDescriptionMixin or ModelInputDataSelfDescriptionMixin_MultiPath to support auto profiling")
        del train_model_wrapper

        # --- Evaluation Paths ---
        eval_model_wrapper = model_manager.create(device, **eval_model_impl_suggestions)
        model = eval_model_wrapper.model
        with _managed_model_cache(model, batch_size, eval_mode_dtype, eval_mode_amp_dtype):
            model.eval()
            if isinstance(model, ModelInputDataSelfDescriptionMixin_MultiPath):
                for path in model.get_data_path_names(with_train=False, with_eval=True):
                    yield ModelExecutionPath(
                        is_train=False, path=f"{path}-eval", model=model,
                        sample_input=model.get_sample_data(path, batch_size, device, eval_mode_dtype, eval_mode_amp_dtype),
                        auto_mixed_precision_dtype=eval_mode_amp_dtype)
            elif isinstance(model, ModelInputDataSelfDescriptionMixin):
                yield ModelExecutionPath(
                    is_train=False, path='eval', model=model,
                    sample_input=model.get_sample_data(batch_size, device, eval_mode_dtype, eval_mode_amp_dtype),
                    auto_mixed_precision_dtype=eval_mode_amp_dtype)
            else:
                raise NotImplementedError("model must implement ModelInputDataSelfDescriptionMixin or ModelInputDataSelfDescriptionMixin_MultiPath to support auto profiling")

    else:
        model_wrapper = model_manager.create(device, **train_model_impl_suggestions)
        model = model_wrapper.model

        if isinstance(model, ModelInputDataSelfDescriptionMixin_MultiPath):
            with _managed_model_cache(model, batch_size, train_mode_dtype, train_mode_amp_dtype):
                model.train()
                for path in model.get_data_path_names(with_train=True, with_eval=False):
                    yield ModelExecutionPath(
                        is_train=True, path=path, model=model,
                        sample_input=model.get_sample_data(path, batch_size, device, train_mode_dtype, train_mode_amp_dtype),
                        auto_mixed_precision_dtype=train_mode_amp_dtype)

            with _managed_model_cache(model, batch_size, eval_mode_dtype, eval_mode_amp_dtype):
                model.eval()
                for path in model.get_data_path_names(with_train=False, with_eval=True):
                    yield ModelExecutionPath(
                        is_train=False, path=path, model=model,
                        sample_input=model.get_sample_data(path, batch_size, device, eval_mode_dtype, eval_mode_amp_dtype),
                        auto_mixed_precision_dtype=eval_mode_amp_dtype)

        elif isinstance(model, ModelInputDataSelfDescriptionMixin):
            # Special case: for single-path, only yield the eval path.
            with _managed_model_cache(model, batch_size, eval_mode_dtype, eval_mode_amp_dtype):
                model.eval()
                yield ModelExecutionPath(
                    is_train=False, path='', model=model,
                    sample_input=model.get_sample_data(batch_size, device, eval_mode_dtype, eval_mode_amp_dtype),
                    auto_mixed_precision_dtype=eval_mode_amp_dtype)
        else:
            raise NotImplementedError("model must implement ModelInputDataSelfDescriptionMixin or ModelInputDataSelfDescriptionMixin_MultiPath to support auto profiling")