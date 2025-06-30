import pprint
import traceback
from typing import Optional
import torch
import gc
from wandb.wandb_run import Run as WandbInstance

from trackit.miscellanies.debugger import debugger_attached
from trackit.models import ModelManager
from trackit.models.utils.profiling import InferencePrecision
from trackit.models.utils.profiling.latency import get_model_latency_for_all_paths
from trackit.models.utils.profiling.flop_count_analysis import analyze_model_flops_for_all_paths
from trackit.models.utils.profiling.number_of_parameters import count_model_parameters
from trackit.miscellanies.torch.distributed import torch_distributed_disable_temporarily, is_rank_0_process


def _get_metric_name(name, exec_path_name):
    if len(exec_path_name) > 1:
        return '_'.join((name, exec_path_name))
    return name


def _run_model_profiling_latency(model_manager: ModelManager, device: torch.device,
                                 inference_precision: InferencePrecision,
                                 train_inference_precision: Optional[InferencePrecision]):
    loops = 100
    warmup_loops = 10
    if device.type == 'cpu':
        loops = 5
        warmup_loops = 1
    latency_stat = get_model_latency_for_all_paths(model_manager, device,
                                                   inference_precision, train_inference_precision,
                                                   loops=loops, warmup_loops=warmup_loops)
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


def _run_model_profiling_flop_count(model_manager: ModelManager,
                                    device: torch.device, inference_precision, train_inference_precision):
    model_metrics = {}
    for path, flop_analysis in analyze_model_flops_for_all_paths(model_manager, device, inference_precision, train_inference_precision):
        print('Flop count table' + ('' if len(path) == 0 else f' ({path})') + ':\n' + flop_analysis.flop_table)
        # print('Flop count by modules and operator' + ('' if len(path) == 0 else f' ({path})') + ':\n' + pprint.pformat(dict(flop_analysis.flops_by_module_and_operator)))
        print('Flop count unsupported ops' + ('' if len(path) == 0 else f' ({path})') + ':\n' + pprint.pformat(dict(flop_analysis.unsupported_ops)))
        print('Total flop count' + ('' if len(path) == 0 else f' ({path})') + ': ', flop_analysis.total_flops)
        model_metrics[_get_metric_name('model_mac', path)] = flop_analysis.total_flops
    return model_metrics


def _run_model_profiling_get_number_of_parameters(model_manager: ModelManager,
                                                  device: torch.device = torch.device('cpu')):
    train_model = model_manager.create(device, optimize_for_inference=False, load_pretrained=False)
    num_parameters = count_model_parameters(train_model.model)
    print('parameters: {}'.format(num_parameters))

    model_metrics = {'model_num_param': num_parameters}

    trainable_num_parameters = count_model_parameters(train_model.model, trainable_only=True)
    if trainable_num_parameters and trainable_num_parameters != num_parameters:
        print('trainable parameters: {}'.format(trainable_num_parameters))
        print('trainable parameters ratio: {:.4f}'.format(trainable_num_parameters / num_parameters))
        model_metrics['model_num_param_trainable'] = trainable_num_parameters

    if model_manager.get_fingerprint_string(device, optimize_for_inference=True, load_pretrained=False) != train_model.fingerprint_string:
        del train_model
        eval_model = model_manager.create(device, optimize_for_inference=True, load_pretrained=False)
        num_parameters = count_model_parameters(eval_model.model)
        print('eval mode parameters: {}'.format(num_parameters))
        model_metrics['model_num_param_eval'] = num_parameters

    return model_metrics


def run_model_profiling(model_manager: ModelManager, device: torch.device,
                        inference_precision: InferencePrecision,
                        train_inference_precision: Optional[InferencePrecision],
                        wandb_instance: Optional[WandbInstance] = None):
    print('Running model profiling...')
    if train_inference_precision is None:
        print_args = [f'device: {device}', f'dtype: {inference_precision.dtype}']
        if inference_precision.auto_mixed_precision_enabled:
            print_args.append(f'automatic mixed precision dtype: {inference_precision.auto_mixed_precision_dtype}')
        print(',\t'.join(print_args))
    else:
        print(f'device: {device}')
        print_args = [f'dtype: {train_inference_precision.dtype}']
        if train_inference_precision.auto_mixed_precision_enabled:
            print_args.append(f'automatic mixed precision dtype: {train_inference_precision.auto_mixed_precision_dtype}')
        print('train:\t' + ',\t'.join(print_args))
        print_args = [f'dtype: {inference_precision.dtype}']
        if inference_precision.auto_mixed_precision_enabled:
            print_args.append(f'automatic mixed precision dtype: {inference_precision.auto_mixed_precision_dtype}')
        print('eval:\t' + ',\t'.join(print_args))
    print('Model profiling report:')
    wandb_report = {}
    wandb_report.update(_run_model_profiling_get_number_of_parameters(model_manager, device))

    try:
        # model latency
        wandb_report.update(_run_model_profiling_latency(model_manager, device,
                                                         inference_precision, train_inference_precision))
    except Exception as e:
        print(f'Latency profiling is not supported for this model. {type(e).__name__}: {e}.')
        print(traceback.format_exc())
        if debugger_attached():
            raise
    try:
        # model flops
        wandb_report.update(_run_model_profiling_flop_count(model_manager, device,
                                                            inference_precision, train_inference_precision))
    except Exception as e:
        print(f'FLOPs profiling is not supported for this model. {type(e).__name__}: {e}.')
        print(traceback.format_exc())
        if debugger_attached():
            raise

    if wandb_instance is not None:
        wandb_instance.summary.update(wandb_report)
    gc.collect()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class ModelProfiler:
    def __init__(self,
                 device: torch.device,
                 inference_precision: InferencePrecision,
                 train_inference_precision: Optional[InferencePrecision],
                 wandb_instance: Optional[WandbInstance] = None):
        self.device = device
        self.inference_precision = inference_precision
        self.train_inference_precision = train_inference_precision
        self.wandb_instance = wandb_instance

    def __call__(self, model: ModelManager):
        if is_rank_0_process():
            with torch_distributed_disable_temporarily():
                run_model_profiling(model,
                                    self.device,
                                    self.inference_precision,
                                    self.train_inference_precision,
                                    wandb_instance=self.wandb_instance)
