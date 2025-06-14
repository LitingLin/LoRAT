import os
from typing import Any, Optional

import torch
import torch.nn as nn

from trackit.core.runtime.metric_logger import get_current_local_metric_logger, get_current_metric_logger
from trackit.data import MainProcessDataPipeline
from trackit.data.protocol.train_input import TrainData
from trackit.miscellanies.torch.distributed.reduce_dict import reduce_dict_async
from trackit.models import ModelManager
from trackit.runners import Runner
from trackit.criteria.builder import build_criterion
from ..common.optimization.optimizer.per_parameter_options.apply import parse_optimizer_per_params_config
from ..default.model_wrapper import ModelWithCriterion
from ..default.utils import get_loss_metrics
import deepspeed


class DeepSpeedTrainer(Runner):
    def __init__(self, device: torch.device, criterion_config: dict,
                 deepspeed_config: dict,
                 model_per_parameter_optimization_config: Optional[list[dict]],
                 criterion_per_parameter_optimization_config: Optional[list[dict]],
                 save_torch_state_dict: bool, enable_torch_compile: bool):
        self._init = False
        self._device = device
        self._criterion_config = criterion_config
        self._deepspeed_config = deepspeed_config
        self._model_per_parameter_optimization_config = model_per_parameter_optimization_config
        self._criterion_per_parameter_optimization_config = criterion_per_parameter_optimization_config
        self._save_torch_state_dict = save_torch_state_dict
        self.data_pipeline_on_main_process = {}
        self._state_to_be_loaded = None
        self._enable_torch_compile = enable_torch_compile
        self._deepspeed_fp16_enabled = deepspeed_config.get('fp16', {}).get('enabled', False)
        self._deepspeed_bf16_enabled = deepspeed_config.get('bf16', {}).get('enabled', False)
        self._simulate_o1_mode_for_fp16 = deepspeed_config.get('fp16', {}).get('simulate_o1_mode_for_fp16', True)

    def register_data_pipeline(self, task_name: str, data_pipeline: MainProcessDataPipeline) -> None:
        if task_name not in self.data_pipeline_on_main_process:
            self.data_pipeline_on_main_process[task_name] = []
        self.data_pipeline_on_main_process[task_name].append(data_pipeline)

    def epoch_begin(self, epoch: int, task_name: str, is_train: bool,
                    model_manager: ModelManager, data_context):
        if not self._init:
            dtype = data_context.dtype
            model = model_manager.create_unmanaged(self._device, dtype)
            criterion = build_criterion(self._criterion_config, self._device, dtype)
            model_with_criterion = ModelWithCriterion(model, criterion)
            optimizer_param_groups = None
            if self._model_per_parameter_optimization_config is not None or self._criterion_per_parameter_optimization_config is not None:
                optimizer_param_groups_config = {
                    'lr': self._deepspeed_config['optimizer']['params']['lr'],
                    'weight_decay': self._deepspeed_config['optimizer']['params']['weight_decay']
                }
                if self._model_per_parameter_optimization_config is not None:
                    optimizer_param_groups_config['per_parameter'] = self._model_per_parameter_optimization_config
                if self._criterion_per_parameter_optimization_config is not None:
                    optimizer_param_groups_config['criterion'] = {
                        'per_parameter': self._criterion_per_parameter_optimization_config
                    }
                optimizer_param_groups = parse_optimizer_per_params_config(model, criterion, optimizer_param_groups_config)

            from trackit.miscellanies.torch.distributed import is_dist_initialized
            if is_dist_initialized():
                model_with_criterion = nn.SyncBatchNorm.convert_sync_batchnorm(model_with_criterion)

            deepspeed_fp16_enabled = self._deepspeed_config.get('fp16', {}).get('enabled', False)
            if deepspeed_fp16_enabled and self._simulate_o1_mode_for_fp16:
                from .force_fp32 import mixed_precision_auto_force_fp32_
                mixed_precision_auto_force_fp32_(model_with_criterion)

            engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model_with_criterion, model_parameters=optimizer_param_groups,
                config=self._deepspeed_config,
                dist_init_required=not is_dist_initialized())
            if self._state_to_be_loaded is not None:
                engine.load_checkpoint(self._state_to_be_loaded)
                self._state_to_be_loaded = None
            self.engine = engine
            if self._enable_torch_compile:
                engine.compile()
            self.engine_state_context = DeepSpeedStateContext(engine, self._save_torch_state_dict)
            self.model_state_update_notifier = model_manager.create_external_updater(self.engine_state_context.state_dict, self.engine_state_context.save)
            self._init = True

        self.task_name = task_name
        self.is_train = is_train
        if not is_train:
            print('rebuilding model for evaluation...')
            model = model_manager.create_unmanaged(self._device, dtype)
            criterion = build_criterion(self._criterion_config, self._device)
            model_with_criterion = ModelWithCriterion(model, criterion)
            model_with_criterion.load_state_dict(self.engine_state_context.state_dict(False), strict=False)
            model_with_criterion.eval()
            self.raw_model = model_with_criterion
        local_logger = get_current_local_metric_logger()
        if local_logger is not None:
            if self.is_train:
                local_logger.set_metric_format('lr', window_size=1, format='{value:.6f}')
            local_logger.set_metric_format('loss')

    def epoch_end(self, epoch: int, *_) -> None:
        data_pipeline_on_main_process = self.data_pipeline_on_main_process.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data_pipeline.stop(epoch)
        if self.is_train:
            self.engine_state_context.reset()
            self.model_state_update_notifier.notify_update()
        else:
            del self.raw_model

    def run(self, data: TrainData) -> None:
        data_pipeline_on_main_process = self.data_pipeline_on_main_process.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data: TrainData = data_pipeline.pre_process(data)
        if self.is_train:
            model_output, criterion_output = self.engine(data.input, data.target)
        else:
            if self._deepspeed_fp16_enabled:
                amp_fn = torch.amp.autocast(self._device.type, enabled=True, dtype=torch.float16)
            elif self._deepspeed_bf16_enabled:
                amp_fn = torch.amp.autocast(self._device.type, enabled=True, dtype=torch.bfloat16)
            else:
                amp_fn = torch.amp.autocast(self._device.type, enabled=False)

            with torch.no_grad(), amp_fn:
                model_output, criterion_output = self.raw_model(data.input, data.target)

        metrics = {}
        if self.is_train:
            metrics['lr'] = self.engine.get_lr()[0]
        metrics.update(**get_loss_metrics(criterion_output))
        metrics = reduce_dict_async(metrics)

        if self.is_train:
            self.engine.backward(criterion_output.loss)
            self.engine.step()

        if data_pipeline_on_main_process is not None:
            for data_pipeline in reversed(data_pipeline_on_main_process):
                model_output = data_pipeline.post_process(model_output)

        metrics = metrics.wait_and_get()
        if len(metrics) > 0:
            get_current_metric_logger().log(metrics)

    def get_state(self, checkpoint_path: str) -> Any:
        if not self._init:
            return None
        assert self.engine_state_context.is_dumped(), "model state is not dumped yet."
        return 'deepspeed'

    def set_state(self, state, checkpoint_path: str) -> None:
        if state is not None:
            state_path = os.path.join(checkpoint_path, state)
            if self._init:
                self.engine.load_checkpoint(state_path)
            else:
                self._state_to_be_loaded = state_path


class DeepSpeedStateContext:
    def __init__(self, engine: deepspeed.DeepSpeedEngine, save_torch_state_dict: bool):
        self.engine = engine
        self.save_torch_state_dict = save_torch_state_dict
        self._dumped = False

    def reset(self):
        self._dumped = False

    def is_dumped(self):
        return self._dumped

    def state_dict(self, exclude_frozen_params: bool) -> dict:
        state_dict = self.engine.module_state_dict(exclude_frozen_parameters=exclude_frozen_params)
        state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
        return state_dict

    def save(self, folder_path: str, exclude_frozen_params: bool):
        path = os.path.join(folder_path, 'deepspeed')
        assert not os.path.exists(path), f"Path {path} already exists."
        self.engine.save_checkpoint(path, exclude_frozen_parameters=exclude_frozen_params)
        self._dumped = True
        if self.save_torch_state_dict:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(path, exclude_frozen_parameters=exclude_frozen_params)
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
            torch_model_path = os.path.join(folder_path, 'model.pth')
            torch.save(state_dict, torch_model_path)
            print(f"Saved model state dict to {torch_model_path}")
        return path
