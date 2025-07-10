from contextlib import nullcontext
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.parallel
from timm.scheduler.scheduler import Scheduler as timmScheduler

from trackit.core.runtime.metric_logger import get_current_local_metric_logger, get_current_metric_logger
from trackit.data import MainProcessDataPipeline
from trackit.core.runtime.context.task import get_current_task_context
from trackit.miscellanies.torch.distributed import is_rank_0_process
from trackit.runners import Runner
from trackit.models import ModelInstance, ModelAuxiliaryBranchStateSavingMixin
from trackit.data.protocol.train_input import TrainData
from trackit.miscellanies.torch.distributed.reduce_dict import reduce_dict_async, reduce_dict

from .model_wrapper import build_model_wrapper, build_model_wrapper_eval
from ..common.nan_dump import do_loss_nan_fault_dump
from ..common.distributed_data_parallel import DistributedDataParallelOption
from ..common.optimization import OptimizationModulesAndOptions
from .utils import criterion_has_parameters, get_loss_metrics


class DefaultTrainer(Runner):
    def __init__(self, model: ModelInstance, criterion: nn.Module,
                 optimization_modules: OptimizationModulesAndOptions,
                 distributed_data_parallel_option: Optional[DistributedDataParallelOption],
                 torch_compile_options: Optional[dict],
                 detect_unused_parameters: bool = True,
                 print_per_parameter_statistics: bool = False,
                 train_mode_during_validation: bool = False):
        self._model_instance = model
        self._raw_model = model.model
        self._criterion = criterion

        self._model: Optional[nn.Module] = None
        self._wrapped_model_train: Optional[nn.Module] = None
        self._has_in_computational_graph_criterion = criterion_has_parameters(criterion)

        self._init = False
        self._distributed_data_parallel_option = distributed_data_parallel_option
        self._torch_compile_options = torch_compile_options

        self._optimizer = optimization_modules.optimizer
        self._optimizer_logging_helper = optimization_modules.optimizer_logging_helper
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
        self._print_per_parameter_statistics = print_per_parameter_statistics
        self._train_batch_size = optimization_modules.train_batch_size
        self._train_mode_during_validation = train_mode_during_validation

        self.data_pipeline_on_main_process = {}
        self.task_name = None
        self.is_train = True
        self._iteration = 0
        self._ddp_no_sync_fn = None

    def _deferred_init(self):
        if self._init:
            return
        model = self._raw_model
        criterion = self._criterion
        distributed_data_parallel_option = self._distributed_data_parallel_option
        torch_compile_options = self._torch_compile_options
        self._wrapped_model_train, self._ddp_no_sync_fn = build_model_wrapper(model, criterion, distributed_data_parallel_option, torch_compile_options)
        self._init = True

    def register_data_pipeline(self, task_name: str, data_pipeline: MainProcessDataPipeline) -> None:
        if task_name not in self.data_pipeline_on_main_process:
            self.data_pipeline_on_main_process[task_name] = []
        self.data_pipeline_on_main_process[task_name].append(data_pipeline)

    def epoch_begin(self, epoch: int, task_name: Optional[str], is_train: bool,
                    model_manager, data_context):
        if data_context.dtype != self._model_instance.dtype:
            print(f'warning: data context dtype {data_context.dtype} does not match model dtype {self._model_instance.dtype}')
        self.task_name = task_name
        self.is_train = is_train
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

        model_mode = self.is_train or self._train_mode_during_validation
        self._raw_model.train(model_mode)
        self._criterion.train(model_mode)
        if self.is_train:
            self._model = self._wrapped_model_train
        else:
            self._model = build_model_wrapper_eval(self._raw_model, self._criterion)

        self._epoch = epoch
        data_pipeline_on_main_process = self.data_pipeline_on_main_process.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data_pipeline.start(epoch, self._raw_model)
        self._metric_accumulator = MetricAccumulator()
        if self._print_per_parameter_statistics:
            from .utils import PerParameterStatistics
            self._per_parameter_statistics = PerParameterStatistics(print_interval=100)

    def epoch_end(self, epoch: int, *_):
        if self._print_per_parameter_statistics:
            del self._per_parameter_statistics
        del self._metric_accumulator
        data_pipeline_on_main_process = self.data_pipeline_on_main_process.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data_pipeline.stop(epoch)
        assert epoch == self._epoch
        if self.is_train:
            if self._lr_scheduler_per_epoch is not None:
                if isinstance(self._lr_scheduler_per_epoch, timmScheduler):
                    self._lr_scheduler_per_epoch.step(epoch)
                else:
                    self._lr_scheduler_per_epoch.step()
            if self._wd_scheduler_per_epoch is not None:
                self._wd_scheduler_per_epoch.step(epoch)
        del self._epoch
        self._model = None

    def run(self, data: TrainData):
        metrics = None
        update_grad = self.is_train and ((self._iteration + 1) % self._grad_accumulation_steps) == 0
        with self._autograd_detect_anomaly_fn(), nullcontext() if update_grad or self._ddp_no_sync_fn is None else self._ddp_no_sync_fn():
            data_pipeline_on_main_process = self.data_pipeline_on_main_process.get(self.task_name, None)
            if data_pipeline_on_main_process is not None:
                for data_pipeline in data_pipeline_on_main_process:
                    data: TrainData = data_pipeline.pre_process(data)

            with torch.set_grad_enabled(self.is_train), self._amp_auto_cast_fn():
                model_output, criterion_output = self._model(data.input, data.target)

            if not torch.isfinite(criterion_output.loss):
                output_path = get_current_task_context().get_output_path()
                metrics = get_loss_metrics(criterion_output)
                if output_path is not None:
                    do_loss_nan_fault_dump(self._model, self._optimizer,
                                           self._lr_scheduler_per_iteration, self._lr_scheduler_per_epoch,
                                           self._parameter_updater,
                                           data, model_output, metrics,
                                           output_path,
                                           self.task_name, self._epoch, self._iteration)
                raise RuntimeError(f"Loss is {criterion_output.loss.item()}, stopping training\n{metrics}")

            if data_pipeline_on_main_process is not None:
                for data_pipeline in reversed(data_pipeline_on_main_process):
                    model_output = data_pipeline.post_process(model_output)

            del model_output

            if self.is_train:
                is_second_order = hasattr(self._optimizer, 'is_second_order') and self._optimizer.is_second_order
                loss_scale, grad_norm = self._parameter_updater.backward_and_unscale(
                    criterion_output.loss / self._grad_accumulation_steps if self._grad_accumulation_steps > 1 else criterion_output.loss,
                    self._optimizer,
                    create_graph=is_second_order,
                    update_grad=update_grad)

                metrics = get_loss_metrics(criterion_output)
                self._metric_accumulator.update(metrics)
                stat = {'lr': self._optimizer_logging_helper.get_lr(),
                        'weight_decay': self._optimizer_logging_helper.get_weight_decay()}
                if loss_scale is not None:
                    stat['loss_scale'] = loss_scale
                if grad_norm is not None:
                    stat['grad_norm'] = grad_norm
                self._metric_accumulator.update(stat)
                self._metric_accumulator.commit(update_grad)

                self._parameter_updater.step(self._optimizer, update_grad=update_grad)

                if update_grad:
                    if self._print_per_parameter_statistics:
                        self._per_parameter_statistics.collect(self._raw_model)

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
                            self._lr_scheduler_per_iteration.step_update(self._iteration // self._grad_accumulation_steps)
                        else:
                            self._lr_scheduler_per_iteration.step()
                    if self._wd_scheduler_per_iteration is not None:
                        self._wd_scheduler_per_iteration.step_update(self._iteration // self._grad_accumulation_steps)
                    metrics = self._metric_accumulator.get()
            else:
                metrics = get_loss_metrics(criterion_output)
                metrics = reduce_dict(metrics)

            if self.is_train:
                self._iteration += 1
                if update_grad:
                    self._model_instance.notify_update()

            if metrics is not None:
                if len(metrics) > 0:
                    get_current_metric_logger().log(metrics)

    def get_state(self, *_) -> dict:
        if hasattr(self._optimizer, 'consolidate_state_dict'):
            self._optimizer.consolidate_state_dict()
        if not is_rank_0_process():
            return None
        state_dict = {'optimizer': self._optimizer.state_dict(), 'iteration': self._iteration}

        if self._has_in_computational_graph_criterion:
            state_dict['criterion'] = self._criterion.state_dict()

        if self._lr_scheduler_per_iteration is not None:
            state_dict['lr_scheduler_per_iteration'] = self._lr_scheduler_per_iteration.state_dict()
        if self._lr_scheduler_per_epoch is not None:
            state_dict['lr_scheduler'] = self._lr_scheduler_per_epoch.state_dict()
        state_dict['amp_param_updater'] = self._parameter_updater.state_dict()

        if isinstance(self._raw_model, ModelAuxiliaryBranchStateSavingMixin):
            state_dict['model_aux'] = self._raw_model.aux_state_dict()

        return state_dict

    def set_state(self, state_dict: dict, *_) -> None:
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
        if isinstance(self._raw_model, ModelAuxiliaryBranchStateSavingMixin) and 'model_aux' in state_dict:
            self._raw_model.load_state_dict(state_dict['model_aux'], strict=False)
        self._parameter_updater.load_state_dict(state_dict['amp_param_updater'])


class MetricAccumulator:
    def __init__(self):
        self._metrics = []
        self._accumulated = None
        self._step = 0

    def update(self, metrics: dict):
        if len(self._metrics) == self._step:
            self._metrics.append({})
        self._metrics[-1].update(metrics)

    def commit(self, update_grad: bool):
        if update_grad:
            if self._step == 0:
                avg_metrics = self._metrics[0]
            else:
                avg_metrics = {}
                for i_step, step_metric in enumerate(self._metrics):
                    if step_metric is None:
                        continue
                    for name in step_metric:
                        if name in avg_metrics:
                            continue
                        current_metric = [m[name] for m in self._metrics[i_step:] if name in m]
                        current_metric = sum(current_metric) / len(current_metric)
                        avg_metrics[name] = current_metric
            self._accumulated = reduce_dict_async(avg_metrics)
            self._metrics = []
            self._step = 0
        else:
            if len(self._metrics) == self._step:
                self._metrics.append(None)
            self._step += 1

    def get(self):
        return self._accumulated.wait_and_get()
