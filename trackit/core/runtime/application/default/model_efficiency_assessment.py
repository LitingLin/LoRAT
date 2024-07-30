import pprint
from typing import Optional
import torch
import gc
from wandb.wandb_run import Run as WandbInstance

from trackit.models import ModelManager, ModelImplSuggestions
from trackit.models.utils.efficiency_assessment.latency import get_model_latency_for_all_paths
from trackit.models.utils.efficiency_assessment.flop_count_analysis import analyze_model_flops_for_all_paths
from trackit.models.utils.efficiency_assessment.number_of_parameters import count_model_parameters


def _get_metric_name(name, exec_path_name):
    if len(exec_path_name) > 1:
        return '_'.join((name, exec_path_name))
    return name


def _run_model_latency_assessment(model_manager: ModelManager, device: torch.device, enable_amp: bool,
                                  amp_dtype: torch.dtype = torch.float16):
    latency_stat = get_model_latency_for_all_paths(model_manager, device,
                                                   auto_mixed_precision=enable_amp, amp_dtype=amp_dtype)
    print('Latency:', end='')
    collected = []
    for path_name, evaluated_model_latency in latency_stat:
        if len(path_name) == 0:
            print(' {:.4f} ms'.format(evaluated_model_latency), end='')
        else:
            print(' {} {:.4f} ms'.format(path_name, evaluated_model_latency), end='')
        collected.append((path_name, evaluated_model_latency))
    print()

    model_metrics = {}
    print('FPS:', end='')
    for path_name, evaluated_model_latency in collected:
        fps = 1000. / evaluated_model_latency
        if len(path_name) == 0:
            print(' {:.4f}'.format(fps), end='')
        else:
            print(' {} {:.4f}'.format(path_name, fps), end='')

        model_metrics[_get_metric_name('model_latency_ms', path_name)] = evaluated_model_latency
        model_metrics[_get_metric_name('model_fps', path_name)] = fps
    print()
    return model_metrics


def _run_model_flop_count_assessment(model_manager: ModelManager, device: torch.device):
    model_metrics = {}
    for path, flop_analysis in analyze_model_flops_for_all_paths(model_manager, device):
        print('Flop count table' + ('' if len(path) == 0 else f' ({path})') + ':\n' + flop_analysis.flop_table)
        # print('Flop count by modules and operator' + ('' if len(path) == 0 else f' ({path})') + ':\n' + pprint.pformat(dict(flop_analysis.flops_by_module_and_operator)))
        print('Flop count unsupported ops' + ('' if len(path) == 0 else f' ({path})') + ':\n' + pprint.pformat(dict(flop_analysis.unsupported_ops)))
        print('Total flop count' + ('' if len(path) == 0 else f' ({path})') + ': ', flop_analysis.total_flops)
        model_metrics[_get_metric_name('model_mac', path)] = flop_analysis.total_flops
    return model_metrics


def _run_model_number_of_parameters_assessment(model_manager: ModelManager, device: torch.device):
    train_model = model_manager.create(device, ModelImplSuggestions(optimize_for_inference=False))
    num_parameters = count_model_parameters(train_model.model)
    print('parameters: {}'.format(num_parameters))

    model_metrics = {'model_num_param': num_parameters}

    trainable_num_parameters = count_model_parameters(train_model.model, trainable_only=True)
    if trainable_num_parameters and trainable_num_parameters != num_parameters:
        print('trainable parameters: {}'.format(trainable_num_parameters))
        print('trainable parameters ratio: {:.4f}'.format(trainable_num_parameters / num_parameters))
        model_metrics['model_num_param_trainable'] = trainable_num_parameters

    if model_manager.get_build_string(ModelImplSuggestions(optimize_for_inference=True)) != train_model.build_string:
        del train_model
        eval_model = model_manager.create(device, ModelImplSuggestions(optimize_for_inference=True))
        num_parameters = count_model_parameters(eval_model.model)
        print('eval parameters: {}'.format(num_parameters))
        model_metrics['model_num_param_eval'] = num_parameters

    return model_metrics


def run_model_efficiency_assessment(model_manager: ModelManager, device: torch.device,
                                    wandb_instance: Optional[WandbInstance] = None,
                                    latency_test_enable_amp: bool = False,
                                    latency_test_amp_dtype: torch.dtype = torch.float16):
    print('Model efficiency assessment report:')
    wandb_report = {}
    wandb_report.update(_run_model_number_of_parameters_assessment(model_manager, device))

    if model_manager.sample_input_data_generator is not None:
        # model latency
        wandb_report.update(_run_model_latency_assessment(model_manager, device, latency_test_enable_amp, latency_test_amp_dtype))

        # model flops
        try:
            wandb_report.update(_run_model_flop_count_assessment(model_manager, device))
        except NotImplementedError:
            print('Model FLOPs assessment is not supported for this model. reason: model cannot be traced by torch.jit.trace')
        except Exception as e:
            print(f'Model FLOPs assessment is not supported for this model. reason: exception occurred.\n{str(e)}')
    else:
        print('Model assessment is not supported for this model. reason: lack of dummy data generator')

    if wandb_instance is not None:
        wandb_instance.summary.update(wandb_report)
    gc.collect()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class ModelEfficiencyAssessment:
    def __init__(self, device: torch.device, wandb_instance: Optional[WandbInstance],
                 latency_test_enable_amp: bool = False, latency_test_amp_dtype: torch.dtype = torch.float16):
        self.device = device
        self.wandb_instance = wandb_instance
        self.latency_test_enable_amp = latency_test_enable_amp
        self.latency_test_amp_dtype = latency_test_amp_dtype

    def __call__(self, model: ModelManager):
        run_model_efficiency_assessment(model, self.device, self.wandb_instance, self.latency_test_enable_amp, self.latency_test_amp_dtype)
