from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.parallel
from timm.scheduler.scheduler import Scheduler as timmScheduler

from trackit.core.runtime.metric_logger import get_current_local_metric_logger, get_current_metric_logger
from trackit.data import HostDataPipeline
from trackit.core.runtime.context.task import get_current_task_context
from trackit.runner import Runner
from trackit.models import ModelInstance
from trackit.data.protocol.train_input import TrainData
from trackit.criteria import CriterionOutput
from trackit.miscellanies.torch.distributed.reduce_dict import reduce_dict_async

from .model_wrapper import build_model_wrapper, build_model_wrapper_eval
from ..common.nan_dump import do_loss_nan_fault_dump
from ..common.distributed_data_parallel import DistributedDataParallelOption
from ..common.optimization import OptimizationModulesAndOptions
from .utils import criterion_has_parameters


class DefaultTrainer(Runner):
    def __init__(self, model: ModelInstance, criterion: nn.Module,
                 optimization_modules: OptimizationModulesAndOptions,
                 distributed_data_parallel_option: Optional[DistributedDataParallelOption],
                 torch_compile_options: Optional[dict],
                 detect_unused_parameters: bool = True):
        self._model_instance = model
        self._raw_model = model.model
        self._criterion = criterion
        self._ema = optimization_modules.ema

        self._model: Optional[nn.Module] = None
        self._wrapped_model_train: Optional[nn.Module] = None
        self._has_in_computational_graph_criterion = criterion_has_parameters(criterion)

        self._init = False
        self._distributed_data_parallel_option = distributed_data_parallel_option
        self._torch_compile_options = torch_compile_options

        self._optimizer = optimization_modules.optimizer
        self._is_apex_optimizer = optimization_modules.is_apex_optimizer
        self._lr_scheduler_per_iteration = optimization_modules.lr_scheduler_per_iteration
        self._lr_scheduler_per_epoch = optimization_modules.lr_scheduler_per_epoch
        self._wd_scheduler_per_iteration = optimization_modules.weight_decay_scheduler_per_iteration
        self._wd_scheduler_per_epoch = optimization_modules.weight_decay_scheduler_per_epoch

        self._parameter_updater = optimization_modules.parameter_updater
        self._amp_auto_cast_fn = optimization_modules.amp_auto_cast_fn
        self._autograd_detect_anomaly_fn = optimization_modules.autograd_detect_anomaly_fn

        self._grad_accumulation_steps = optimization_modules.grad_accumulation_steps
        self._zero_grad_set_to_none = optimization_modules.zero_grad_set_to_none

        self._detect_unused_parameters = detect_unused_parameters

        self.data_pipeline_on_host = {}
        self.task_name = None
        self.is_train = True
        self._iteration = 0

    def _deferred_init(self):
        if self._init:
            return
        model = self._raw_model
        criterion = self._criterion
        distributed_data_parallel_option = self._distributed_data_parallel_option
        torch_compile_options = self._torch_compile_options
        self._wrapped_model_train = build_model_wrapper(model, criterion, distributed_data_parallel_option, torch_compile_options)
        self._init = True

    def register_data_pipeline(self, task_name: str, data_pipeline: HostDataPipeline) -> None:
        if task_name not in self.data_pipeline_on_host:
            self.data_pipeline_on_host[task_name] = []
        self.data_pipeline_on_host[task_name].append(data_pipeline)

    def switch_task(self, task_name, is_train):
        self.task_name = task_name
        self.is_train = is_train

    def epoch_begin(self, epoch: int, _):
        self._deferred_init()
        local_logger = get_current_local_metric_logger()
        if local_logger is not None:
            if self.is_train:
                local_logger.set_metric_format('lr', window_size=1, format='{value:.6f}')
                local_logger.set_metric_format('weight_decay', window_size=1, format='{value:.6f}')
                if self._parameter_updater.is_grad_scaler_enabled():
                    local_logger.set_metric_format('loss_scale', window_size=1, format='{value:.2f}')
                if self._parameter_updater.has_grad_norm():
                    local_logger.set_metric_format('grad_norm', window_size=1, format='{value:.4f}')
            local_logger.set_metric_format('loss')

        self._raw_model.train(self.is_train)
        self._criterion.train(self.is_train)
        if self.is_train:
            self._model = self._wrapped_model_train
        else:
            self._model = build_model_wrapper_eval(self._raw_model, self._criterion)
        if self._ema is not None:
            self._ema.on_epoch_begin()
        self._epoch = epoch

    def epoch_end(self, epoch: int, model_manager):
        assert epoch == self._epoch
        if self.is_train:
            if self._lr_scheduler_per_epoch is not None:
                if isinstance(self._lr_scheduler_per_epoch, timmScheduler):
                    self._lr_scheduler_per_epoch.step(epoch)
                else:
                    self._lr_scheduler_per_epoch.step()
            if self._wd_scheduler_per_epoch is not None:
                self._wd_scheduler_per_epoch.step(epoch)
            if self._ema is None:
                self._model_instance.notify_update()
            else:
                model_manager.load_state_dict(self._ema.get_model().state_dict(), print_missing=False)
        del self._epoch
        self._model = None

    def run(self, data: TrainData):
        with self._autograd_detect_anomaly_fn():
            metrics = {}
            data_pipeline_on_host = self.data_pipeline_on_host.get(self.task_name, None)
            if data_pipeline_on_host is not None:
                for data_pipeline in data_pipeline_on_host:
                    data: TrainData = data_pipeline.pre_process(data)

            model_output = None
            criterion_output: Optional[CriterionOutput] = None
            if data.input is not None:
                with torch.set_grad_enabled(self.is_train), self._amp_auto_cast_fn():
                    criterion_output = self._model(data.input, data.target)

                if criterion_output.metrics is None:
                    metrics['loss'] = criterion_output.loss.item()
                else:
                    metrics['loss'] = sum(criterion_output.metrics.values())
                    metrics.update(criterion_output.metrics)
                if criterion_output.extra_metrics is not None:
                    metrics.update(criterion_output.extra_metrics)

                if not torch.isfinite(criterion_output.loss):
                    output_path = get_current_task_context().get_output_path()
                    if output_path is not None:
                        do_loss_nan_fault_dump(self._model, self._optimizer,
                                               self._lr_scheduler_per_iteration, self._lr_scheduler_per_epoch,
                                               self._parameter_updater,
                                               data, model_output, metrics,
                                               output_path,
                                               self.task_name, self._epoch, self._iteration)
                    raise RuntimeError(f"Loss is {criterion_output.loss.item()}, stopping training\n{metrics}")

            if criterion_output is not None and self.is_train:
                is_second_order = hasattr(self._optimizer, 'is_second_order') and self._optimizer.is_second_order
                update_grad = ((self._iteration + 1) % self._grad_accumulation_steps) == 0
                loss_scale, grad_norm = self._parameter_updater.backward_and_unscale(
                    criterion_output.loss,
                    self._optimizer,
                    create_graph=is_second_order,
                    update_grad=update_grad)

                stat = {'lr': self._optimizer.param_groups[0]["lr"],
                        'weight_decay': self._optimizer.param_groups[0]["weight_decay"]}
                if loss_scale is not None:
                    stat['loss_scale'] = loss_scale
                if grad_norm is not None:
                    stat['grad_norm'] = grad_norm
                metrics.update(stat)
                metrics = reduce_dict_async(metrics)

                self._parameter_updater.step(self._optimizer, update_grad=update_grad)

                if update_grad:
                    if self._detect_unused_parameters:
                        for name, param in self._model.named_parameters():
                            if param.grad is None and param.requires_grad:
                                print(f'unused parameter detected: {name}')
                    if self._is_apex_optimizer:
                        self._optimizer.zero_grad()
                    else:
                        self._optimizer.zero_grad(self._zero_grad_set_to_none)
                    if self._lr_scheduler_per_iteration is not None:
                        if isinstance(self._lr_scheduler_per_iteration, timmScheduler):
                            self._lr_scheduler_per_iteration.step_update(self._iteration)
                        else:
                            self._lr_scheduler_per_iteration.step()
                    if self._wd_scheduler_per_iteration is not None:
                        self._wd_scheduler_per_iteration.step_update(self._iteration)
                    if self._ema is not None:
                        self._ema.update_parameters(self._raw_model)
            else:
                metrics = reduce_dict_async(metrics)

            if data_pipeline_on_host is not None:
                for data_pipeline in reversed(data_pipeline_on_host):
                    model_output = data_pipeline.post_process(model_output)

            if self.is_train:
                self._iteration += 1

            metrics = metrics.get()
            if len(metrics) > 0:
                get_current_metric_logger().log(metrics)

    def get_state(self) -> Any:
        state_dict = {'optimizer': self._optimizer.state_dict(), 'iteration': self._iteration}

        if self._has_in_computational_graph_criterion:
            state_dict['criterion'] = self._criterion.state_dict()

        if self._lr_scheduler_per_iteration is not None:
            state_dict['lr_scheduler_per_iteration'] = self._lr_scheduler_per_iteration.state_dict()
        if self._lr_scheduler_per_epoch is not None:
            state_dict['lr_scheduler'] = self._lr_scheduler_per_epoch.state_dict()
        state_dict['amp_param_updater'] = self._parameter_updater.state_dict()
        if self._ema is not None:
            state_dict['ema'] = self._ema.state_dict()

        return state_dict

    def set_state(self, state_dict: Any) -> None:
        if self._distributed_data_parallel_option is not None:
            assert not self._init, "limitation: state cannot be updated when torch.nn.parallel.DistributedDataParallel is enabled"
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._iteration = state_dict['iteration']
        if self._has_in_computational_graph_criterion:
            self._criterion.load_state_dict(state_dict['criterion'])
        if self._lr_scheduler_per_iteration is not None:
            self._lr_scheduler_per_iteration.load_state_dict(state_dict['lr_scheduler_per_iteration'])
        if self._lr_scheduler_per_epoch is not None:
            self._lr_scheduler_per_epoch.load_state_dict(state_dict['lr_scheduler'])
        if self._ema is not None:
            self._ema.load_state_dict(state_dict['ema'])
        self._parameter_updater.load_state_dict(state_dict['amp_param_updater'])
