from dataclasses import dataclass, field
import os
import shutil
import torch
from typing import Dict, Any, Optional, List, Callable, Mapping, Iterable
from collections import OrderedDict
import safetensors.torch
from trackit.miscellanies.slugify import slugify
from trackit.core.runtime.utils.epoch_activation_criteria import EpochActivationCriterion
from trackit.miscellanies.torch.distributed import is_main_process


@dataclass
class _DumpedFilePaths:
    model: List[str] = field(default_factory=list)
    app_state: List[str] = field(default_factory=list)


def _create_folder(folder_path: str):
    if is_main_process():
        os.makedirs(folder_path, exist_ok=True)


def _copy_file(src: str, dst: str, try_hardlink_first: bool = True):
    if try_hardlink_first:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copyfile(src, dst)


def _get_model_weight_file_name(model_name: str = 'model', use_safetensors: bool = True) -> str:
    return model_name + '.bin' if use_safetensors else model_name + '.pth'


def _get_app_state_file_name() -> str:
    return 'state.pth'


class _CheckpointDumper:
    def __call__(self, epoch: int,
                 model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]], application_state_getter: Optional[Callable[[], Any]],
                 dumped_file_paths: _DumpedFilePaths):
        raise NotImplementedError()


class _CheckpointDumper_with_metrics:
    def __call__(self, epoch: int, metrics: Dict[str, float],
                 model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]], application_state_getter: Optional[Callable[[], Any]],
                 dumped_file_paths: _DumpedFilePaths):
        raise NotImplementedError()


def _save_model_weight(model_state_dict_getter: Callable[[], Mapping[str, Any]], file_path: str,
                       dumped_file_paths: List[str],
                       force_contiguous: bool = True, use_safetensors: bool = True):
    if file_path in dumped_file_paths:
        return
    if len(dumped_file_paths) > 1:
        if is_main_process():
            _copy_file(dumped_file_paths[0], file_path)
            print(f'checkpoint: model weight copied from {dumped_file_paths[0]} to {file_path}', flush=True)
    else:
        if is_main_process():
            state_dict = model_state_dict_getter()
            if force_contiguous:
                state_dict = OrderedDict((key, value.contiguous() if isinstance(value, torch.Tensor) else value) for key, value in state_dict.items())
            if use_safetensors:
                safetensors.torch.save_file(state_dict, file_path)
            else:
                torch.save(state_dict, file_path)
            print(f'checkpoint: model weight --> {file_path}', flush=True)
    dumped_file_paths.append(file_path)


def _save_application_state(application_state_getter: Callable[[], Any], file_path: str,
                            dumped_file_paths: List[str]):
    if file_path in dumped_file_paths:
        return
    if len(dumped_file_paths) > 1:
        _copy_file(dumped_file_paths[0], file_path)
        print(f'checkpoint: app state copied from {dumped_file_paths[0]} to {file_path}', flush=True)
    else:
        application_state = application_state_getter()
        if is_main_process():
            torch.save(application_state, file_path)
            print(f'checkpoint: app state --> {file_path}', flush=True)
    dumped_file_paths.append(file_path)


class EpochEventCheckpointDumper(_CheckpointDumper):
    def __init__(self, output_path: str, epoch_activation_criteria: EpochActivationCriterion, max_epoch: int, resumable: bool, max_to_keep: int):
        self._output_path = output_path
        self._epoch_activation_criteria = epoch_activation_criteria
        self._epoch_digit_max_length = len(str(max_epoch))
        self._resumable = resumable
        self._max_to_keep = max_to_keep
        self._saved_epochs = set()

    def __call__(self, epoch: int,
                 model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]], application_state_getter: Optional[Callable[[], Any]],
                 dumped_file_paths: _DumpedFilePaths):
        if not self._epoch_activation_criteria(epoch):
            return

        if self._max_to_keep > 0:
            self._saved_epochs.add(epoch)
            if len(self._saved_epochs) > self._max_to_keep:
                epoch_to_remove = min(self._saved_epochs)
                folder_to_remove = os.path.join(self._output_path, self._get_folder_name(epoch_to_remove))
                if is_main_process():
                    shutil.rmtree(folder_to_remove)
                    print('checkpoint: max_to_keep activated, removed folder', folder_to_remove, flush=True)
                self._saved_epochs.remove(epoch_to_remove)

        folder_path = os.path.join(self._output_path, self._get_folder_name(epoch))
        _create_folder(folder_path)

        # dump model state
        if model_state_dict_getter is not None:
            model_state_file_path = os.path.join(folder_path, _get_model_weight_file_name())
            _save_model_weight(model_state_dict_getter, model_state_file_path, dumped_file_paths.model)

        if self._resumable and application_state_getter is not None:
            # dump application state
            application_state_file_path = os.path.join(folder_path, _get_app_state_file_name())
            _save_application_state(application_state_getter, application_state_file_path, dumped_file_paths.app_state)

    def _get_folder_name(self, epoch: int):
        return f"epoch_{epoch:0{self._epoch_digit_max_length}}"


class BestCheckpointDumper(_CheckpointDumper_with_metrics):
    def __init__(self, output_path: str, metric_name: str, top_k: int, epoch_activation_criteria: EpochActivationCriterion, resumable: bool):
        self._output_path = output_path
        self._metric_name = metric_name
        self._metric_name_slugified = slugify(metric_name, allow_unicode=True)
        self._top_k = top_k
        self._top_k_metric_values = [None for _ in range(top_k)]
        self._epoch_activation_criteria = epoch_activation_criteria
        self._resumable = resumable

    def __call__(self, epoch: int, metrics: Dict[str, float],
                 model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]], application_state_getter: Optional[Callable[[], Any]],
                 dumped_file_paths: _DumpedFilePaths):
        if self._metric_name not in metrics:
            return

        if not self._epoch_activation_criteria(epoch):
            return

        metric_value = metrics[self._metric_name]
        rank = self._get_rank(metric_value)
        if rank is None:
            return

        folder_path = os.path.join(self._output_path, self._get_top_n_folder_name(rank))
        model_state_file_path = os.path.join(folder_path, _get_model_weight_file_name())

        for i in range(self._top_k - 1, rank - 1, -1):
            if self._top_k_metric_values[i] is None:
                continue

            this_rank_folder_path = os.path.join(self._output_path, self._get_top_n_folder_name(i))

            if i == self._top_k - 1:
                if is_main_process():
                    shutil.rmtree(this_rank_folder_path)

            if i - 1 >= rank:
                higher_rank_folder_path = os.path.join(self._output_path, self._get_top_n_folder_name(i - 1))
                if is_main_process():
                    os.rename(higher_rank_folder_path, this_rank_folder_path)
                self._top_k_metric_values[i] = self._top_k_metric_values[i - 1]

        self._top_k_metric_values[rank] = metric_value

        _create_folder(folder_path)

        if model_state_dict_getter is not None:
            _save_model_weight(model_state_dict_getter, model_state_file_path, dumped_file_paths.model)

        if self._resumable and application_state_getter is not None:
            application_state_file_path = os.path.join(folder_path, _get_app_state_file_name())
            _save_application_state(application_state_getter, application_state_file_path, dumped_file_paths.app_state)

    def _get_rank(self, metric_value):
        for i in range(self._top_k):
            if self._top_k_metric_values[i] is None:
                return i
            if metric_value > self._top_k_metric_values[i]:
                return i
        return None

    def _get_top_n_folder_name(self, rank: int):
        folder_name = f"best@{self._metric_name_slugified}"
        if rank > 1:
            folder_name += f"_top_{rank + 1:0{len(str(self._top_k))}}"
        return folder_name


class LatestCheckpointDumper(_CheckpointDumper):
    def __init__(self, output_path: str, resumable: bool):
        self._output_path = output_path
        self._resumable = resumable

    def __call__(self, epoch: int,
                 model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]], application_state_getter: Optional[Callable[[], Any]],
                 dumped_file_paths: _DumpedFilePaths):
        folder_path = os.path.join(self._output_path, 'latest')
        _create_folder(folder_path)

        if model_state_dict_getter is not None:
            model_state_file_path = os.path.join(folder_path, _get_model_weight_file_name())
            _save_model_weight(model_state_dict_getter, model_state_file_path, dumped_file_paths.model)

        if self._resumable and application_state_getter is not None:
            application_state_file_path = os.path.join(folder_path, _get_app_state_file_name())
            _save_application_state(application_state_getter, application_state_file_path, dumped_file_paths.app_state)


class CheckpointDumper:
    def __init__(self, checkpoint_dumpers: Iterable[_CheckpointDumper]):
        self._checkpoint_dumpers = tuple(dumper for dumper in checkpoint_dumpers if not isinstance(dumper, _CheckpointDumper_with_metrics))
        self._metric_based_checkpoint_dumpers = tuple(dumper for dumper in checkpoint_dumpers if isinstance(dumper, _CheckpointDumper_with_metrics))
        self._model_weight_dedup_cache: Dict[int, List[str]] = {}  # model weight version -> file path
        self._app_state_dedup_cache: Dict[int, List[str]] = {}  # epoch -> file path

    def temporary_dump(self, epoch: int, model_weight_version: int,
                       model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]]):
        if model_state_dict_getter is None:
            return
        saved_file_paths = _DumpedFilePaths(self._model_weight_dedup_cache.get(model_weight_version, []), None)
        for dumper in self._checkpoint_dumpers:
            dumper(epoch, model_state_dict_getter, None, saved_file_paths)
        if len(saved_file_paths.model) > 0:
            self._model_weight_dedup_cache[model_weight_version] = saved_file_paths.model

    def dump(self, epoch: int, metrics: Dict[str, float], model_weight_version: int,
             model_state_dict_getter: Optional[Callable[[], Mapping[str, Any]]], application_state_getter: Optional[Callable[[], Any]]):
        if model_state_dict_getter is None and application_state_getter is None:
            return
        saved_file_paths = _DumpedFilePaths(self._model_weight_dedup_cache.get(model_weight_version, []), self._app_state_dedup_cache.get(epoch, []))
        for dumper in self._checkpoint_dumpers:
            dumper(epoch, model_state_dict_getter, application_state_getter, saved_file_paths)
        for dumper in self._metric_based_checkpoint_dumpers:
            dumper(epoch, metrics, model_state_dict_getter, application_state_getter, saved_file_paths)
        if len(saved_file_paths.model) > 0:
            self._model_weight_dedup_cache[model_weight_version] = saved_file_paths.model
        if len(saved_file_paths.app_state) > 0:
            self._app_state_dedup_cache[epoch] = saved_file_paths.app_state
