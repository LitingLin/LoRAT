from typing import Optional, Iterable

from trackit.core.runtime.metric_logger.epoch_metric import EpochMetrics
from trackit.core.runtime.utils.checkpoint.save import CheckpointDumper
from trackit.models import ModelManager


class ApplicationCheckpointDumper:
    def __init__(self, checkpoint_dumper: Optional[CheckpointDumper],
                 model_manager: ModelManager, application_state_getter,
                 epoch_metric_holder: EpochMetrics):
        self._checkpoint_dumper = checkpoint_dumper
        self._model_manager = model_manager
        self._application_state_getter = application_state_getter
        self._epoch_metric_holder = epoch_metric_holder
        self._epoch = 0
        self._global_step = 0
        self._total_epochs = None

    def initialize(self, epoch: int, global_step: int):
        self._epoch = epoch
        self._global_step = global_step
        self._total_epochs = None

    def dump_every_epoch(self, iterable: Iterable):
        if self._checkpoint_dumper is None:
            for obj in iterable:
                yield obj
                print('Output path is not set. Skip checkpoint saving.')
            return
        if hasattr(iterable, '__len__'):
            total_epochs = len(iterable)
            self._total_epochs = total_epochs
            for obj in iterable:
                yield obj

                self._checkpoint_dumper.dump(self._epoch, self._epoch == total_epochs - 1,
                                             self._epoch_metric_holder.get(self._epoch - 1),
                                             self._model_manager.version,
                                             self._model_manager.save,
                                             self._application_state_getter)
                self._epoch += 1
        else:
            self._total_epochs = None
            first_epoch = True
            for obj in iterable:
                if not first_epoch:
                    self._checkpoint_dumper.dump(self._epoch - 1, False,
                                                 self._epoch_metric_holder.get(self._epoch - 1),
                                                 self._model_manager.version,
                                                 self._model_manager.save,
                                                 self._application_state_getter)
                first_epoch = False
                yield obj
                self._epoch += 1

            self._checkpoint_dumper.dump_step_based(self._epoch, True,
                                                    self._model_manager.version,
                                                    self._model_manager.save)

            self._checkpoint_dumper.dump(self._epoch - 1, True,
                                         self._epoch_metric_holder.get(self._epoch - 1),
                                         self._model_manager.version,
                                         self._model_manager.save,
                                         self._application_state_getter)

    def dump_model_state(self):
        if self._checkpoint_dumper is None:
            return
        is_last = self._total_epochs is not None and self._epoch == self._total_epochs - 1
        self._checkpoint_dumper.dump(self._epoch, is_last, None, self._model_manager.version,
                                     self._model_manager.save,
                                     None)

    def dump_every_iteration(self, iterable: Iterable):
        if self._checkpoint_dumper is None:
            for obj in iterable:
                yield obj
            return
        if hasattr(iterable, '__len__'):
            total_iterations = len(iterable)
            for index, obj in enumerate(iterable):
                yield obj
                is_last = index == total_iterations - 1 and self._total_epochs is not None and self._epoch == self._total_epochs - 1
                self._checkpoint_dumper.dump_step_based(self._global_step, is_last,
                                                        self._model_manager.version,
                                                        self._model_manager.save)
                self._global_step += 1
        else:
            first_iteration = True
            for obj in iterable:
                if not first_iteration:
                    self._checkpoint_dumper.dump_step_based(self._global_step - 1, False,
                                                            self._model_manager.version,
                                                            self._model_manager.save)
                first_iteration = False
                yield obj
                self._global_step += 1
            is_last = self._total_epochs is not None and self._epoch == self._total_epochs - 1
            self._checkpoint_dumper.dump_step_based(self._global_step - 1, is_last,
                                                    self._model_manager.version,
                                                    self._model_manager.save)
