import copy
from typing import Mapping

from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.context.task import TaskContext
from trackit.data.context import DataContext

def build_deepspeed_trainer(runner_config: dict, build_context: BuildContext, config: dict,
                                  associated_task_contexts: Mapping[str, TaskContext],
                                  associated_data_contexts: Mapping[str, DataContext]):
    print('building deepspeed trainer...')
    from .runner import DeepSpeedTrainer
    from ..default.builder import _get_train_task_data_context
    train_task_context, train_data_context = _get_train_task_data_context(associated_task_contexts, associated_data_contexts)
    assert train_task_context is not None, "a train task attached to this runner is required"

    num_epochs = train_task_context.epoch_selector.total_executions()
    train_batch_size = train_data_context.variables['batch_size']
    num_iterations_per_epoch = train_data_context.variables['num_iterations_per_epoch']

    deepspeed_config = copy.deepcopy(runner_config['deepspeed'])
    deepspeed_config['train_micro_batch_size_per_gpu'] = train_batch_size
    assert "train_batch_size" not in deepspeed_config

    if 'scheduler' in deepspeed_config:
        scheduler_config = deepspeed_config['scheduler']
        if scheduler_config['type'] in ('WarmupDecayLR', 'WarmupCosineLR'):
            scheduler_params_config = scheduler_config['params']
            scheduler_params_config['total_num_steps'] = num_epochs * num_iterations_per_epoch
            if 'warmup_num_epochs' in scheduler_params_config:
                scheduler_params_config['warmup_num_steps'] = scheduler_params_config['warmup_num_epochs'] * num_iterations_per_epoch
                del scheduler_params_config['warmup_num_epochs']

    print('deepspeed trainer is lazy initialized')
    return DeepSpeedTrainer(build_context.device,
                            runner_config['criteria'],
                            deepspeed_config,
                            runner_config.get('per_parameter_optimization'),
                            runner_config.get('criterion_per_parameter_optimization'),
                            runner_config.get('save_torch_state_dict', False),
                            runner_config.get('enable_torch_compile', False))