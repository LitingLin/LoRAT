import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Protocol, Optional, Union, Callable, Mapping, Any, Sequence, MutableSequence, List
from weakref import WeakValueDictionary
from trackit.models.schema.input.dummy_data_generation import SampleInputDataGeneratorInterface, SampleInputDataGeneratorInterface_MultiPath
import safetensors.torch


@dataclass(frozen=True)
class ModelImplSuggestions:
    torch_jit_trace_compatible: bool = False
    optimize_for_inference: bool = False


class ModelBuilder(Protocol):
    def __call__(self, optim_advice: ModelImplSuggestions) -> nn.Module:
        ...


class ModelBuildStringGenerator(Protocol):
    def __call__(self, optim_advice: ModelImplSuggestions) -> str:
        ...


@dataclass(frozen=True)
class ModelBuildingContext:
    create_fn: ModelBuilder
    build_string_generator: ModelBuildStringGenerator
    sample_input_data_generator: Optional[Union[SampleInputDataGeneratorInterface, SampleInputDataGeneratorInterface_MultiPath]] = None


class ModelInstance:
    def __init__(self, model: nn.Module, device: torch.device, version: int,
                 build_string: str, parent: 'ModelManager'):
        self._model = model
        self._device = device
        self._version = version
        self._build_string = build_string
        self._parent = parent

    def notify_update(self):
        assert self._parent._version == self._version, "only the newest model can be updated"
        self._version += 1
        self._parent._newest_model = self._model
        self._parent._version = self._version

    @property
    def build_string(self) -> str:
        return self._build_string

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device


class ModelManager:
    def __init__(self, model_building_context: ModelBuildingContext):
        self._create_fn = model_building_context.create_fn
        self._build_string_generator = model_building_context.build_string_generator
        self._sample_input_data_generator = model_building_context.sample_input_data_generator
        self._model_instance_cache = WeakValueDictionary()
        self._version: int = 0
        self._newest_model: Optional[nn.Module] = None
        self._offloaded_model_state_dict: Optional[Mapping[str, Any]] = None
        self._states_to_be_load: List[Callable[[nn.Module], None]] = []

    def get_build_string(self, impl_suggestions: ModelImplSuggestions) -> str:
        return self._build_string_generator(impl_suggestions)

    @property
    def sample_input_data_generator(self) -> Optional[Union[SampleInputDataGeneratorInterface, SampleInputDataGeneratorInterface_MultiPath]]:
        return self._sample_input_data_generator

    def create(self, device: torch.device, impl_suggestions: ModelImplSuggestions = ModelImplSuggestions()) -> ModelInstance:
        build_string = self._build_string_generator(impl_suggestions)
        model_instance = self._model_instance_cache.get((build_string, device))
        if model_instance is not None:
            if model_instance._version != self._version:
                _load_state_dict(model_instance._model, self._newest_model.state_dict(), strict=False, print_missing=True)
                model_instance._version = self._version
        else:
            model = self._create_fn(impl_suggestions)
            model = model.to(device)
            if self._newest_model is None:
                self._newest_model = model
                if self._offloaded_model_state_dict is not None:
                    _load_state_dict(model, self._offloaded_model_state_dict, strict=False, print_missing=True)
                    self._offloaded_model_state_dict = None
                if len(self._states_to_be_load) > 0:
                    for load_fn in self._states_to_be_load:
                        load_fn(model)
                    self._version += 1
                    self._states_to_be_load = []
            elif self._version > 0:
                _load_state_dict(model, self._newest_model.state_dict(), strict=False, print_missing=True)
            model_instance = ModelInstance(model, device, self._version, build_string, self)
            self._model_instance_cache[(build_string, device)] = model_instance

        return model_instance

    def load_state_dict(self, state_dict: dict, strict: bool = False, print_missing: bool = True):
        if self._newest_model is None:
            self._states_to_be_load.append(lambda model: _load_state_dict(model, state_dict, strict, print_missing))
        else:
            _load_state_dict(self._newest_model, state_dict, strict, print_missing)
            self._version += 1

    def load_state_dict_from_file(self, state_file: str, strict: bool = False, print_missing: bool = True, use_safetensors: bool = True):
        if self._newest_model is None:
            self._states_to_be_load.append(lambda model: _load_state_dict_from_file(model, state_file, strict, print_missing, use_safetensors))
        else:
            _load_state_dict_from_file(self._newest_model, state_file, strict, print_missing, use_safetensors)
            self._version += 1

    def state_dict(self) -> Optional[Mapping[str, Any]]:
        if self._offloaded_model_state_dict is not None:
            return self._offloaded_model_state_dict
        if self._newest_model is None:
            raise RuntimeError('ModelManager: state_dict() called before create()')
        return self._newest_model.state_dict()

    def save_state_dict_to_file(self, state_file: str, use_safetensors: bool = True):
        if use_safetensors:
            safetensors.torch.save_model(self._newest_model, state_file)
        else:
            torch.save(self._newest_model.state_dict(), state_file)

    @property
    def latest_model(self) -> Optional[nn.Module]:
        return self._newest_model

    @property
    def version(self) -> int:
        return self._version

    def release(self):
        self._model_instance_cache.clear()
        if self._newest_model is not None:
            state_dict = self._newest_model.cpu().state_dict()
            self._offloaded_model_state_dict = state_dict
        self._newest_model = None


def _load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool = False, print_missing: bool = True):
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if len(missing) > 0 and print_missing:
        print(f'missing keys in state_dict: {missing}')
    if len(unexpected) > 0:
        raise RuntimeError(f'unexpected keys in state_dict: {unexpected}')


def _load_state_dict_from_file(model: nn.Module, state_file: str, strict: bool = False, print_missing: bool = True, use_safetensors: bool = True):
    if use_safetensors:
        missing, unexpected = safetensors.torch.load_model(model, state_file, strict=strict)
    else:
        missing, unexpected = model.load_state_dict(torch.load(state_file, map_location='cpu'), strict=strict)
    if len(missing) > 0 and print_missing:
        print(f'missing keys in state_dict: {missing}')
    if len(unexpected) > 0:
        raise RuntimeError(f'unexpected keys in state_dict: {unexpected}')
