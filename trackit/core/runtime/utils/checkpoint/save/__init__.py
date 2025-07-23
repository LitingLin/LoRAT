from dataclasses import dataclass, field
import os
import shutil
import torch
from typing import Dict, Optional, List, Sequence

from trackit.miscellanies.slugify import slugify
from trackit.miscellanies.torch.distributed import is_rank_0_process
from .scheduling import EpochOrStepSelector
from .. import model_state_save_fn, application_state_save_fn
from ..recovery import write_recovery_file

@dataclass
class _DumpedFilePaths:
    model_state_path: Optional[str] = None
    model_state_output_folder_paths: List[str] = field(default_factory=list)
    app_state_file_paths: List[str] = field(default_factory=list)


def _create_folder(folder_path: str):
    if is_rank_0_process():
        os.makedirs(folder_path, exist_ok=True)


def _copy_file(src: str, dst: str, try_hardlink_first: bool = True):
    if try_hardlink_first:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copyfile(src, dst)


def _copy_to_folder(src_file_path: str, dst_folder_path: str, try_link_first: bool = True):
    is_directory = os.path.isdir(src_file_path)
    basename = os.path.basename(src_file_path)
    target_path = os.path.join(dst_folder_path, basename)

    if try_link_first:
        if not is_directory:
            try:
                os.link(src_file_path, target_path)
                return target_path
            except OSError:
                pass

    if is_directory:
        shutil.copytree(src_file_path, target_path)
    else:
        shutil.copyfile(src_file_path, target_path)
    return target_path

_app_state_file_name = 'state.tar'


class _CheckpointDumper:
    def __call__(self, epoch_or_step: int, is_last_epoch_or_step: bool,
                 model_state_saver: Optional[model_state_save_fn],
                 application_state_getter: Optional[application_state_save_fn],
                 dumped_file_paths: _DumpedFilePaths):
        raise NotImplementedError()


class _CheckpointDumper_MetricAware:
    def __call__(self, epoch_or_step: int, is_last_epoch_or_step: bool,
                 metrics: Dict[str, float],
                 model_state_saver: Optional[model_state_save_fn],
                 application_state_getter: Optional[application_state_save_fn],
                 dumped_file_paths: _DumpedFilePaths):
        raise NotImplementedError()


def _save_model_weight(model_state_saver: model_state_save_fn,
                       output_folder_path: str,
                       dumped_file_paths :_DumpedFilePaths,
                       exclude_frozen_parameters: bool):
    if output_folder_path in dumped_file_paths.model_state_output_folder_paths:
        return
    if is_rank_0_process():
        if dumped_file_paths.model_state_path is not None:
            output_path = _copy_to_folder(dumped_file_paths.model_state_path, output_folder_path)
            print(f'checkpoint: model weight copied from {dumped_file_paths.model_state_path} to {output_path}', flush=True)
        else:
            output_path = model_state_saver(output_folder_path, exclude_frozen_parameters)
            print(f'checkpoint: model weight --> {output_path}', flush=True)
            dumped_file_paths.model_state_path = output_path
    dumped_file_paths.model_state_output_folder_paths.append(output_folder_path)


def _save_application_state(application_state_getter: application_state_save_fn, output_folder_path: str,
                            dumped_file_paths: _DumpedFilePaths):
    state_file_path = os.path.join(output_folder_path, _app_state_file_name)
    if state_file_path in dumped_file_paths.app_state_file_paths:
        return
    if len(dumped_file_paths.app_state_file_paths) > 1:
        if is_rank_0_process():
            _copy_file(dumped_file_paths.app_state_file_paths[0], state_file_path)
            print(f'checkpoint: app state copied from {dumped_file_paths.app_state_file_paths[0]} to {state_file_path}', flush=True)
    else:
        application_state = application_state_getter(output_folder_path)
        if is_rank_0_process():
            torch.save(application_state, state_file_path)
            print(f'checkpoint: app state --> {state_file_path}', flush=True)
    dumped_file_paths.app_state_file_paths.append(state_file_path)


class RegularCheckpointDumper(_CheckpointDumper):
    def __init__(self, output_path: str, epoch_or_step_selector: EpochOrStepSelector, dump_last: bool,
                 max_epoch_or_step: Optional[int], resumable: bool, max_to_keep: int, is_epoch_based: bool,
                 exclude_frozen_parameters: bool
                 ):
        self._output_path = output_path
        self._epoch_or_step_selector = epoch_or_step_selector
        self._file_name_digit_max_length = len(str(max_epoch_or_step)) if max_epoch_or_step is not None else 0
        self._resumable = resumable
        self._dump_last = dump_last
        self._max_to_keep = max_to_keep
        self._saved_epochs_or_steps = set()
        self._is_epoch_based = is_epoch_based
        self._exclude_frozen_parameters = exclude_frozen_parameters

    def __call__(self, epoch_or_step: int, is_last: bool,
                 model_state_saver: Optional[model_state_save_fn],
                 application_state_getter: Optional[application_state_save_fn],
                 dumped_file_paths: _DumpedFilePaths):
        need_to_run = (is_last and self._dump_last) or self._epoch_or_step_selector(epoch_or_step)
        if not need_to_run:
            return

        if self._max_to_keep > 0:
            self._saved_epochs_or_steps.add(epoch_or_step)
            remove_old = False
            this_epoch_has_model_state = model_state_saver is not None or dumped_file_paths.model_state_path is not None
            this_epoch_has_app_state = application_state_getter is not None or len(dumped_file_paths.app_state_file_paths) > 0
            this_epoch_state_has_saved = this_epoch_has_model_state and (not self._resumable or this_epoch_has_app_state)
            if this_epoch_state_has_saved:
                if len(self._saved_epochs_or_steps) > self._max_to_keep:
                    remove_old = True
            if remove_old:
                epoch_to_remove = min(self._saved_epochs_or_steps)
                folder_to_remove = os.path.join(self._output_path, self._get_folder_name(epoch_to_remove))
                if is_rank_0_process():
                    shutil.rmtree(folder_to_remove)
                    print('checkpoint: max_to_keep activated, removed folder', folder_to_remove, flush=True)
                self._saved_epochs_or_steps.remove(epoch_to_remove)

        folder_path = os.path.join(self._output_path, self._get_folder_name(epoch_or_step))
        _create_folder(folder_path)

        # dump model state
        if model_state_saver is not None:
            _save_model_weight(model_state_saver, folder_path, dumped_file_paths, self._exclude_frozen_parameters)

        if self._resumable and application_state_getter is not None:
            # dump application state
            _save_application_state(application_state_getter, folder_path, dumped_file_paths)

    def _get_folder_name(self, epoch_or_step: int):
        if self._is_epoch_based:
            return f"epoch_{epoch_or_step:0{self._file_name_digit_max_length}}"
        else:
            return f"step_{epoch_or_step:0{self._file_name_digit_max_length}}"


class BestCheckpointDumper(_CheckpointDumper_MetricAware):
    def __init__(self, output_path: str, metric_name: str, top_k: int,
                 epoch_or_step_selector: EpochOrStepSelector, dump_last: bool,
                 resumable: bool, exclude_frozen_parameters: bool):
        self._output_path = output_path
        self._metric_name = metric_name
        self._metric_name_slugified = slugify(metric_name, allow_unicode=True)
        self._top_k = top_k
        self._top_k_metric_values = [None for _ in range(top_k)]
        self._epoch_or_step_selector = epoch_or_step_selector
        self._dump_last = dump_last
        self._resumable = resumable
        self._exclude_frozen_parameters = exclude_frozen_parameters

    def __call__(self, epoch_or_step: int, is_last: bool,
                 metrics: Dict[str, float],
                 model_state_saver: Optional[model_state_save_fn],
                 application_state_getter: Optional[application_state_save_fn],
                 dumped_file_paths: _DumpedFilePaths):
        need_to_run = (is_last and self._dump_last) or self._epoch_or_step_selector(epoch_or_step)
        if not need_to_run:
            return
        if self._metric_name not in metrics:
            return

        metric_value = metrics[self._metric_name]
        rank = self._get_rank(metric_value)
        if rank is None:
            return

        folder_path = os.path.join(self._output_path, self._get_top_n_folder_name(rank))

        for i in range(self._top_k - 1, rank - 1, -1):
            if self._top_k_metric_values[i] is None:
                continue

            this_rank_folder_path = os.path.join(self._output_path, self._get_top_n_folder_name(i))

            if i == self._top_k - 1:
                if is_rank_0_process():
                    shutil.rmtree(this_rank_folder_path)

            if i - 1 >= rank:
                higher_rank_folder_path = os.path.join(self._output_path, self._get_top_n_folder_name(i - 1))
                if is_rank_0_process():
                    os.rename(higher_rank_folder_path, this_rank_folder_path)
                self._top_k_metric_values[i] = self._top_k_metric_values[i - 1]

        self._top_k_metric_values[rank] = metric_value

        _create_folder(folder_path)

        if model_state_saver is not None:
            _save_model_weight(model_state_saver, folder_path, dumped_file_paths, self._exclude_frozen_parameters)

        if self._resumable and application_state_getter is not None:
            _save_application_state(application_state_getter, folder_path, dumped_file_paths)

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
    def __init__(self, output_path: str, resumable: bool, exclude_frozen_parameters: bool):
        self._output_path = output_path
        self._resumable = resumable
        self._exclude_frozen_parameters = exclude_frozen_parameters

    def __call__(self, epoch: int, is_last: bool,
                 model_state_saver: Optional[model_state_save_fn],
                 application_state_getter: Optional[application_state_save_fn],
                 dumped_file_paths: _DumpedFilePaths):
        folder_path = os.path.join(self._output_path, 'latest')
        _create_folder(folder_path)

        if model_state_saver is not None:
            _save_model_weight(model_state_saver, folder_path, dumped_file_paths, self._exclude_frozen_parameters)

        if self._resumable and application_state_getter is not None:
            _save_application_state(application_state_getter, folder_path, dumped_file_paths)


class CheckpointDumper:
    def __init__(self, output_path: str,
                 epoch_based_checkpoint_dumpers: Sequence[_CheckpointDumper],
                 step_based_checkpoint_dumpers: Sequence[_CheckpointDumper],
                 metric_based_checkpoint_dumpers: Sequence[_CheckpointDumper_MetricAware]):
        self._output_path = output_path
        self._epoch_based_checkpoint_dumpers = epoch_based_checkpoint_dumpers
        self._step_based_checkpoint_dumpers = step_based_checkpoint_dumpers
        self._metric_based_checkpoint_dumpers = metric_based_checkpoint_dumpers
        self._model_weight_file_path_cache: Dict[int, Optional[str]] = {}  # model version -> model weight file path
        self._model_weight_output_paths_dedup_cache: Dict[int, List[str]] = {}  # model version -> output paths
        self._app_state_dedup_cache: Dict[int, List[str]] = {}  # epoch -> file path

    def has_dumpers(self):
        return len(self._epoch_based_checkpoint_dumpers) > 0 or len(self._step_based_checkpoint_dumpers) > 0 or len(self._metric_based_checkpoint_dumpers) > 0

    def dump(self, epoch: int, is_last: bool, metrics: Optional[Dict[str, float]], model_weight_version: int,
             model_state_saver: Optional[model_state_save_fn],
             application_state_getter: Optional[application_state_save_fn]):
        if model_weight_version == 0:
            model_state_saver = None
        if model_state_saver is None and application_state_getter is None:
            return
        saved_file_paths = _DumpedFilePaths(self._model_weight_file_path_cache.get(model_weight_version, None),
                                            self._model_weight_output_paths_dedup_cache.get(model_weight_version, []),
                                            self._app_state_dedup_cache.get(epoch, []))
        for dumper in self._epoch_based_checkpoint_dumpers:
            dumper(epoch, is_last, model_state_saver, application_state_getter, saved_file_paths)
        if metrics is not None:
            for dumper in self._metric_based_checkpoint_dumpers:
                dumper(epoch, is_last, metrics, model_state_saver, application_state_getter, saved_file_paths)
        if len(saved_file_paths.model_state_output_folder_paths) > 0:
            self._model_weight_file_path_cache[model_weight_version] = saved_file_paths.model_state_path
            self._model_weight_output_paths_dedup_cache[model_weight_version] = saved_file_paths.model_state_output_folder_paths
        if len(saved_file_paths.app_state_file_paths) > 0:
            self._app_state_dedup_cache[epoch] = saved_file_paths.app_state_file_paths
        if self._model_weight_file_path_cache.get(model_weight_version) is not None and self._app_state_dedup_cache.get(epoch) is not None:
            if is_rank_0_process():
                recovery_file_path = write_recovery_file(self._output_path,
                                                         self._model_weight_file_path_cache[model_weight_version],
                                                         self._app_state_dedup_cache[epoch][-1])
                print(f'checkpoint: recovery -> {recovery_file_path}', flush=True)

    def dump_step_based(self, step: int, is_last: bool, model_weight_version: int,
                        model_state_saver: Optional[model_state_save_fn]):
        if model_state_saver is None or model_weight_version == 0:
            return
        saved_file_paths = _DumpedFilePaths(self._model_weight_file_path_cache.get(model_weight_version, None),
                                            self._model_weight_output_paths_dedup_cache.get(model_weight_version, []))
        for dumper in self._step_based_checkpoint_dumpers:
            dumper(step, is_last, model_state_saver, None, saved_file_paths)
        if saved_file_paths.model_state_path is not None:
            self._model_weight_file_path_cache[model_weight_version] = saved_file_paths.model_state_path
            self._model_weight_output_paths_dedup_cache[model_weight_version] = saved_file_paths.model_state_output_folder_paths
