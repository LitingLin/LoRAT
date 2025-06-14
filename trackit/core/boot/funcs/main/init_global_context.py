import traceback
import os
import gc

from typing import Optional

import torch
import wandb

from trackit.miscellanies.torch.distributed import (init_torch_distributed, is_dist_initialized,
                                                    is_rank_0_process, is_local_rank_0_process,
                                                    cleanup_torch_distributed,
                                                    get_num_nodes, get_node_index, get_local_rank)
from trackit.miscellanies.torch.distributed.network_context import (init_distributed_group_network_context,
                                                                    DistributedGroupNetworkContext)
from trackit.miscellanies.versioning import generate_app_version_from_git, GitAppVersion
from trackit.miscellanies.system.machine.cpu_info import get_processor_name
from trackit.core.runtime.global_constant import get_global_constant
from ..utils.output_stream_redirector import OutputStreamRedirector
from trackit.miscellanies.debugger import debugger_attached


def _get_version_string(version: GitAppVersion):
    version_string = f"{version.sha}-{version.branch}"
    if version.has_diff:
        version_string += "-dirty"
    return version_string


def _print_machine_information(device: str):
    device = torch.device(device)
    print('CPU: ', get_processor_name())
    if 'cuda' == device.type and torch.cuda.is_available():
        print('GPU: ', torch.cuda.get_device_name(device))


def _get_logging_file_postfix():
    if not is_dist_initialized():
        return ''
    if get_num_nodes() == 1:
        return f'.{get_local_rank()}'
    else:
        return f'.{get_node_index()}.{get_local_rank()}'


def _get_logging_file_name(args):
    if args.output_dir is None:
        return None

    if args.disable_file_logging:
        return None

    return os.path.join(args.output_dir, f'log{_get_logging_file_postfix()}.txt')


def _update_output_dir_and_redirect_output_stream(args):
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.run_id)
        os.makedirs(args.output_dir, exist_ok=True)

    mute_stdout = not (is_local_rank_0_process() or args.allow_non_master_node_printing)
    output_stream_redirector = OutputStreamRedirector(_get_logging_file_name(args), mute_stdout)
    output_stream_redirector.__enter__()

    if args.output_dir is not None:
        print('output directory: ', args.output_dir)
    return output_stream_redirector


def apply_environment_variables():
    HF_TOKEN = get_global_constant('HF_TOKEN', default=None)
    if HF_TOKEN and 'HF_TOKEN' not in os.environ:
        os.environ['HF_TOKEN'] = HF_TOKEN
    WANDB_API_KEY = get_global_constant('WANDB_API_KEY', default=None)
    if WANDB_API_KEY and 'WANDB_API_KEY' not in os.environ:
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY

    if get_global_constant('TIMM_USE_OLD_CACHE', default=True):
        os.environ['TIMM_USE_OLD_CACHE'] = '1'


class Profiler:
    def __init__(self, output_path: str | None):
        import cProfile
        self._profiler = cProfile.Profile()
        self._output_path = output_path

    def enable(self):
        self._profiler.enable()
        print('cProfile enabled.')

    def disable(self):
        self._profiler.disable()
        print('cProfile disabled.')

    def dump(self):
        if self._output_path is None:
            print('Output path is not specified. Dumping profiling result to stdout...')
            self._profiler.print_stats()
        else:
            print(f'Dumping profiling result to {self._output_path}...', end='')
            self._profiler.dump_stats(self._output_path)
            print('Done.')


class GlobalContext:
    def __init__(self, network_context: DistributedGroupNetworkContext,
                 wandb_instance: Optional[wandb.sdk.wandb_run.Run],
                 output_stream_redirector: OutputStreamRedirector,
                 profiler: Optional[Profiler] = None):
        self.network_context = network_context
        self.wandb_instance = wandb_instance
        self.output_stream_redirector = output_stream_redirector
        self.profiler = profiler
        self._disposed = False

    def __enter__(self):
        if self._disposed:
            raise RuntimeError('GlobalContext is already disposed.')
        if self.profiler is not None:
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disposed = True
        if self.profiler is not None:
            self.profiler.disable()
        exception_raised = exc_type is not None
        if self.wandb_instance is not None:
            self.wandb_instance.__exit__(exc_type, exc_val, exc_tb)
        else:
            if exception_raised:
                traceback.print_exception(exc_type, exc_val, exc_tb)
        self.output_stream_redirector.__exit__(exc_type, exc_val, exc_tb)
        if self.profiler is not None:
            self.profiler.dump()
        del self.profiler
        del self.wandb_instance
        del self.output_stream_redirector
        gc.collect()
        cleanup_torch_distributed()
        return not debugger_attached()

def init_global_context(runtime_vars, config: dict):
    init_torch_distributed(runtime_vars.device)

    apply_environment_variables()

    if not runtime_vars.do_sweep:
        output_stream_redirector = _update_output_dir_and_redirect_output_stream(runtime_vars)

    if not runtime_vars.quiet:
        print('version: ', _get_version_string(generate_app_version_from_git()))
        _print_machine_information(runtime_vars.device)

    network_context = init_distributed_group_network_context()
    print('IP: ', network_context.local.ip)
    print('Hostname: ', network_context.local.hostname)
    if network_context.group is not None:
        print('Distributed Group:')
        host_names = {network_context.ip: network_context.hostname for network_context in network_context.group}
        for context in network_context.group:
            print(f'  {context.ip} ({context.hostname})')
    else:
        host_names = {network_context.local.ip: network_context.local.hostname}

    wandb_instance = None

    if runtime_vars.enable_wandb:
        wandb_tags = []
        if runtime_vars.mixin_config is not None:
            wandb_tags.extend(runtime_vars.mixin_config)

        if runtime_vars.do_sweep:
            wandb_tags.append('sweep')

        if runtime_vars.wandb_distributed_aware or not is_dist_initialized():
            from .setup_wandb import setup_wandb
            wandb_instance = setup_wandb(runtime_vars, config, str(host_names), wandb_tags)
            if runtime_vars.do_sweep:
                runtime_vars.run_id = wandb_instance.id
                output_stream_redirector = _update_output_dir_and_redirect_output_stream(runtime_vars)
        else:
            if is_rank_0_process():
                from .setup_wandb import setup_wandb
                wandb_instance = setup_wandb(runtime_vars, config, str(host_names), wandb_tags)
                if runtime_vars.do_sweep:
                    run_id = wandb_instance.id
                    torch.distributed.broadcast_object_list([run_id])
                    runtime_vars.run_id = run_id
                    output_stream_redirector = _update_output_dir_and_redirect_output_stream(runtime_vars)
            else:
                if runtime_vars.do_sweep:
                    run_id = [None]
                    torch.distributed.broadcast_object_list(run_id)
                    run_id = run_id[0]
                    runtime_vars.run_id = run_id
                    output_stream_redirector = _update_output_dir_and_redirect_output_stream(runtime_vars)

    profiler = None
    if runtime_vars.enable_profiling:
        profiler_output_file_path = None
        if runtime_vars.output_dir is not None:
            profiler_output_file_path = os.path.join(runtime_vars.output_dir,
                                                     f'profiling_result{_get_logging_file_postfix()}.prof')
        profiler = Profiler(profiler_output_file_path)

    return GlobalContext(network_context, wandb_instance, output_stream_redirector, profiler)
