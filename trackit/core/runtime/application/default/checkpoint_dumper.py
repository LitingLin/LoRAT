from typing import Optional, Iterable, Sequence

from trackit.core.runtime.metric_logger.epoch_metric import EpochMetrics
from trackit.core.runtime.utils.checkpoint.save import CheckpointDumper
from trackit.models import ModelManager
from .local_context import GlobalIterationCounter


class ApplicationCheckpointDumper:
    def __init__(self, checkpoint_dumper: Optional[CheckpointDumper],
                 global_step_getter: GlobalIterationCounter,
                 model_manager: ModelManager, application_state_getter,
                 epoch_metric_holder: EpochMetrics):
        self._checkpoint_dumper = checkpoint_dumper
        self._global_step_getter = global_step_getter
        self._model_manager = model_manager
        self._application_state_getter = application_state_getter
        self._epoch_metric_holder = epoch_metric_holder

    def dump_every_epoch(self, epoch_iterator: Iterable[int]) -> Iterable[int]:
        if self._checkpoint_dumper is None:
            for epoch in epoch_iterator:
                yield epoch
                print('Output path is not set. Skip checkpoint saving.')
            return
        total_epochs = len(epoch_iterator)
        self._total_epochs = total_epochs
        for epoch in epoch_iterator:
            self._epoch = epoch
            self._is_last_epoch = epoch == total_epochs - 1
            yield epoch

            self._checkpoint_dumper.dump(epoch, self._is_last_epoch,
                                         self._epoch_metric_holder.get(epoch - 1),
                                         self._model_manager.version,
                                         self._model_manager.save,
                                         self._application_state_getter)

    def dump_model_state(self):
        if self._checkpoint_dumper is None:
            return
        self._checkpoint_dumper.dump(self._epoch, self._is_last_epoch,
                                     None, self._model_manager.version,
                                     self._model_manager.save,
                                     None)

    def dump_every_iteration(self, data_loader: Iterable):
        if self._checkpoint_dumper is None:
            for obj in data_loader:
                yield obj
            return
        if hasattr(data_loader, '__len__'):
            total_iterations = len(data_loader)
            for index, obj in enumerate(data_loader):
                yield obj
                is_last = index == total_iterations - 1 and self._is_last_epoch
                self._checkpoint_dumper.dump_step_based(self._global_step_getter.get_iteration(), is_last,
                                                        self._model_manager.version,
                                                        self._model_manager.save)
        else:
            global_step = None
            for obj in data_loader:
                if global_step is not None:
                    self._checkpoint_dumper.dump_step_based(global_step, False,
                                                            self._model_manager.version,
                                                            self._model_manager.save)
                yield obj
                global_step = self._global_step_getter.get_iteration()
            self._checkpoint_dumper.dump_step_based(global_step, self._is_last_epoch,
                                                    self._model_manager.version,
                                                    self._model_manager.save)
