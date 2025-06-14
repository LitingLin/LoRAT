import os.path

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Protocol, Optional, Callable, Mapping, Any, Sequence, List, Union, Iterable
from weakref import WeakValueDictionary
import safetensors.torch

from trackit.core.runtime.global_constant import get_global_constant
from trackit.miscellanies.torch.check_version import is_torch_version_greater_or_equal


@dataclass(frozen=True)
class ModelImplementationSuggestions:
    device: torch.device
    dtype: torch.dtype = torch.float32
    torch_jit_trace_compatible: bool = False
    optimize_for_inference: bool = False
    load_pretrained: bool = True


class ModelFactory(Protocol):
    def __call__(self, optim_advice: ModelImplementationSuggestions) -> nn.Module:
        ...


class ModelBuildFingerprintFn(Protocol):
    """Return a deterministic string that uniquely identifies
    a model build given a set of options (used for caching, logging, etc.)."""
    def __call__(self, optim_advice: ModelImplementationSuggestions) -> str:
        ...


@dataclass(frozen=True)
class ModelBuildContext:
    create_fn: ModelFactory
    fingerprint_fn: ModelBuildFingerprintFn


class ModelInstance:
    def __init__(self, model: nn.Module,
                 device: torch.device, dtype: torch.dtype,
                 version: int,
                 fingerprint_string: str, parent: 'ModelManager'):
        self._model = model
        self._device = device
        self._dtype = dtype
        self._version = version
        self._fingerprint_string = fingerprint_string
        self._parent = parent

    def notify_update(self):
        assert self._parent._version == self._version, "only the latest model can be updated"
        self._version += 1
        self._parent._latest_model_state.update_state(self._model)
        self._parent._version = self._version

    @property
    def fingerprint_string(self) -> str:
        return self._fingerprint_string

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


class _custom_state_dict_fn(Protocol):
    def __call__(self, exclude_frozen_parameters: bool) -> dict[str, Any]:
        ...


class _custom_model_state_save_fn(Protocol):
    def __call__(self, folder_path: str, exclude_frozen_parameters: bool) -> str:
        ...


@dataclass
class ModelState:
    _model: Optional[nn.Module] = None
    _state_dict_fn: Optional[_custom_state_dict_fn] = None
    _model_state_save_fn: Optional[_custom_model_state_save_fn] = None
    _prefer_safetensors: bool = True
    _safetensors_file_name: str = 'model.safetensors'
    _model_state_file_name: str = 'model.pth'

    def update_state(self, model: nn.Module):
        self._model = model
        self._state_dict_fn = None
        self._model_state_save_fn = None

    def update_state_with_custom_state_dict_fn(self,
                                               state_dict_fn: _custom_state_dict_fn,
                                               model_state_save_fn: Optional[_custom_model_state_save_fn] = None):
        self._model = None
        self._state_dict_fn = state_dict_fn
        self._model_state_save_fn = model_state_save_fn

    def state_dict(self, exclude_frozen_parameters: bool = False) -> Optional[dict[str, Any]]:
        if self._model is not None:
            if exclude_frozen_parameters:
                return _state_dict_exclude_frozen_parameters(self._model.state_dict(keep_vars=True))
            else:
                return self._model.state_dict()
        elif self._state_dict_fn is not None:
            return self._state_dict_fn(exclude_frozen_parameters)
        else:
            raise RuntimeError('ModelState: state_dict() called without available context')

    def clear(self):
        self._model = None
        self._state_dict_fn = None
        self._model_state_save_fn = None

    def updated(self):
        return self._model is not None or self._state_dict_fn is not None

    def offload(self):
        if self._model is not None:
            offloaded_model_state = _OffloadedModelState(self._model, self._prefer_safetensors, self._model_state_file_name)
            self._state_dict_fn = offloaded_model_state.state_dict
            self._model_state_save_fn = offloaded_model_state.save

    def save(self, folder_path: str, exclude_frozen_parameters: bool = False):
        if self._model is not None:
            if self._prefer_safetensors:
                file_path = os.path.join(folder_path, self._safetensors_file_name)
                if exclude_frozen_parameters:
                    original_state_dict_fn = self._model.state_dict
                    self._model.state_dict = _state_dict_fn_exclude_frozen_parameters.__get__(self._model, self._model.__class__)
                    safetensors.torch.save_model(self._model, file_path)
                    self._model.state_dict = original_state_dict_fn
                else:
                    safetensors.torch.save_model(self._model, file_path)
            else:
                file_path = os.path.join(folder_path, self._model_state_file_name)
                if exclude_frozen_parameters:
                    state_dict = _state_dict_exclude_frozen_parameters(self._model.state_dict(keep_vars=True))
                else:
                    state_dict = self._model.state_dict()
                torch.save(state_dict, file_path)
            return file_path
        elif self._model_state_save_fn is not None:
            return self._model_state_save_fn(folder_path, exclude_frozen_parameters)
        elif self._state_dict_fn is not None:
            state_dict = self._state_dict_fn(exclude_frozen_parameters)
            if self._prefer_safetensors:
                file_path = os.path.join(folder_path, self._safetensors_file_name)
            else:
                file_path = os.path.join(folder_path, self._model_state_file_name)
            _default_save_dict_fn(state_dict, file_path, self._prefer_safetensors)
            return file_path
        else:
            raise RuntimeError('ModelState: save() called without available context')


def _default_save_dict_fn(state_dict: dict[str, Any], file_path: str, use_safetensors: bool = True):
    if use_safetensors:
        safetensors.torch.save_file(state_dict, file_path)
    else:
        torch.save(state_dict, file_path)


class _OffloadedModelState:
    def __init__(self, model: nn.Module, use_safetensors: bool, model_state_file_name: str):
        state_dict = model.state_dict(keep_vars=True)
        frozen_parameters = tuple(k for k, v in state_dict.items() if isinstance(v, torch.Tensor) and not v.requires_grad)
        state_dict = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        self._state_dict = state_dict
        self._frozen_parameters = frozen_parameters
        self._use_safetensors = use_safetensors
        self._model_state_file_name = model_state_file_name

    def save(self, folder_path: str, exclude_frozen_parameters: bool):
        state_dict = self.state_dict(exclude_frozen_parameters)
        file_path = os.path.join(folder_path, self._model_state_file_name)
        _default_save_dict_fn(state_dict, file_path, self._use_safetensors)
        return file_path

    def state_dict(self, exclude_frozen_parameters: bool):
        if exclude_frozen_parameters:
            state_dict = {k: v for k, v in self._state_dict.items() if k not in self._frozen_parameters}
        else:
            state_dict = self._state_dict
        return state_dict


class ExternalModelUpdater:
    def __init__(self, model_state_dict_fn: _custom_state_dict_fn,
                 model_state_save_fn: Optional[_custom_model_state_save_fn],
                 parent: 'ModelManager'):
        self._model_state_dict_fn = model_state_dict_fn
        self._model_state_save_fn = model_state_save_fn
        self._parent = parent

    def notify_update(self):
        self._parent._latest_model_state.update_state_with_custom_state_dict_fn(self._model_state_dict_fn, self._model_state_save_fn)
        self._parent._version += 1


class ModelManager:
    def __init__(self, model_building_context: ModelBuildContext):
        self._create_fn = model_building_context.create_fn
        self._fingerprint_generator = model_building_context.fingerprint_fn
        self._model_instance_cache = WeakValueDictionary()
        self._version: int = 0
        self._latest_model_state = ModelState()
        self._states_to_be_load: List[Callable[[nn.Module, torch.device], None]] = []
        self._prefer_safetensors = get_global_constant('use_safetensors', default=True)

    def get_fingerprint_string(self, device: torch.device, dtype: torch.dtype = torch.float32,
                               torch_jit_trace_compatible: bool = False,
                               optimize_for_inference: bool = False,
                               load_pretrained: bool = True) -> str:
        return self._fingerprint_generator(ModelImplementationSuggestions(device, dtype,
                                                                          torch_jit_trace_compatible, optimize_for_inference,
                                                                          load_pretrained))

    def create(self, device: torch.device, dtype: torch.dtype = torch.float32,
               torch_jit_trace_compatible: bool = False,
               optimize_for_inference: bool = False,
               load_pretrained: bool = True) -> ModelInstance:
        impl_suggestions = ModelImplementationSuggestions(device, dtype, torch_jit_trace_compatible, optimize_for_inference,
                                                          load_pretrained)
        fingerprint = self._fingerprint_generator(impl_suggestions)
        model_instance = self._model_instance_cache.get(fingerprint)
        if model_instance is not None:
            if model_instance._version != self._version:
                for load_fn in self._states_to_be_load:
                    load_fn(model_instance._model, device)
                _load_state_dict(model_instance._model, self._latest_model_state.state_dict(), strict=False, print_missing=True)
                model_instance._version = self._version
        else:
            model = self._create_fn(impl_suggestions)
            for load_fn in self._states_to_be_load:
                load_fn(model, device)
            if self._version > 0:
                _load_state_dict(model, self._latest_model_state.state_dict(), strict=False, print_missing=True)
            model_instance = ModelInstance(model, device, dtype, self._version, fingerprint, self)
            self._model_instance_cache[fingerprint] = model_instance

        return model_instance

    def create_unmanaged(self, device: torch.device, dtype: torch.dtype = torch.float32,
                         torch_jit_trace_compatible: bool = False,
                         optimize_for_inference: bool = False,
                         load_pretrained: bool = True) -> nn.Module:
        impl_suggestions = ModelImplementationSuggestions(device, dtype, torch_jit_trace_compatible, optimize_for_inference,
                                                          load_pretrained)
        model = self._create_fn(impl_suggestions)
        for load_fn in self._states_to_be_load:
            load_fn(model, device)
        if self._version > 0:
            _load_state_dict(model, self._latest_model_state.state_dict(), strict=False, print_missing=True)
        return model

    def create_external_updater(self, model_state_dict_fn: _custom_state_dict_fn,
                                model_state_save_fn: Optional[_custom_model_state_save_fn] = None) -> ExternalModelUpdater:
        return ExternalModelUpdater(model_state_dict_fn, model_state_save_fn, self)

    def load_state_dict(self, state_dict: dict, strict: bool = False, print_missing: bool = True):
        assert self._version == 0
        self._states_to_be_load.append(lambda model, _: _load_state_dict(model, state_dict, strict, print_missing))
        for model in self._model_instance_cache.values():
            _load_state_dict(model._model, state_dict, strict, print_missing)

    def load_state_dict_from_file(self, state_file: str, strict: bool = False, print_missing: bool = True):
        assert self._version == 0
        self._states_to_be_load.append(lambda model, device: _load_state_dict_from_file(model, state_file, device, strict, print_missing))
        for model in self._model_instance_cache.values():
            _load_state_dict_from_file(model._model, state_file, model._device, strict, print_missing)

    def state_dict(self, exclude_frozen_parameters: bool = False) -> Optional[dict[str, Any]]:
        return self._latest_model_state.state_dict(exclude_frozen_parameters)

    def save(self, folder_path: str, exclude_frozen_parameters: bool = False):
        return self._latest_model_state.save(folder_path, exclude_frozen_parameters)

    @property
    def version(self) -> int:
        return self._version

    def release(self):
        self._model_instance_cache.clear()
        self._latest_model_state.offload()


def _state_dict_exclude_frozen_parameters(state_dict: dict):
    return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items() if not isinstance(v, torch.Tensor) or v.requires_grad}


def _state_dict_fn_exclude_frozen_parameters(self: nn.Module, **kwargs):
    kwargs['keep_vars'] = True
    state_dict = super(self.__class__, self).state_dict(**kwargs)
    state_dict = _state_dict_exclude_frozen_parameters(state_dict)
    return state_dict


def _default_model_state_save_fn(model: nn.Module, folder_path: str, exclude_frozen_parameters: bool, use_safetensors: bool = True):
    if use_safetensors:
        file_path = os.path.join(folder_path, 'model.safetensors')
        if exclude_frozen_parameters:
            original_state_dict_fn = model.state_dict
            model.state_dict = _state_dict_fn_exclude_frozen_parameters.__get__(model, model.__class__)
            safetensors.torch.save_model(model, file_path)
            model.state_dict = original_state_dict_fn
        else:
            safetensors.torch.save_model(model, file_path)
    else:
        file_path = os.path.join(folder_path, 'model.pth')
        if exclude_frozen_parameters:
            state_dict = _state_dict_exclude_frozen_parameters(model.state_dict(keep_vars=True))
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, file_path)
    return file_path


def _get_state_dict(model: nn.Module, exclude_frozen_parameters: bool):
    if exclude_frozen_parameters:
        state_dict = model.state_dict(keep_vars=True)
        state_dict = {k: v.detach() for k, v in state_dict.items() if v.requires_grad}
    else:
        state_dict = model.state_dict()
    return state_dict


def _load_state_dict(model: nn.Module, state_dict: dict[str, Any],
                     strict: bool = False, print_missing: bool = True):
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if len(missing) > 0 and print_missing:
        print(f'missing keys in state_dict: {missing}')
    if len(unexpected) > 0:
        raise RuntimeError(f'unexpected keys in state_dict: {unexpected}')


def _load_state_dict_from_file(model: nn.Module, state_file: str, device: torch.device,
                               strict: bool = False, print_missing: bool = True):
    if state_file.endswith(('.safetensors', '.bin')):
        if isinstance(device, torch.device):
            device = str(device)
        missing, unexpected = safetensors.torch.load_model(model, state_file, strict=strict, device=device)
    else:
        if is_torch_version_greater_or_equal((1, 13)):
            state_dict = torch.load(state_file, map_location=device, weights_only=True)
        else:
            state_dict = torch.load(state_file, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if len(missing) > 0 and print_missing:
        print(f'missing keys in state_dict: {missing}')
    if len(unexpected) > 0:
        raise RuntimeError(f'unexpected keys in state_dict: {unexpected}')


class ModelAuxiliaryBranchStateSavingMixin:
    def aux_state_dict(self):
        raise NotImplementedError()


class ModelInputDataSelfDescriptionMixin:
    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, auto_mixed_precision_dtype: Optional[torch.dtype]) -> Union[torch.Tensor, Iterable, Mapping]:
        raise NotImplementedError()


class ModelInputDataSelfDescriptionMixin_MultiPath:
    def get_sample_data(self, name: str, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, auto_mixed_precision_dtype: Optional[torch.dtype]) -> Union[torch.Tensor, Iterable, Mapping]:
        raise NotImplementedError()

    def get_data_path_names(self, with_train: bool = True, with_eval: bool = True) -> Sequence[str]:
        raise NotImplementedError()
