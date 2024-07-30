from typing import Optional
import os

import torch
import wandb

from trackit.miscellanies.torch.distributed import init_torch_distributed, is_dist_initialized, is_main_process
from trackit.miscellanies.torch.distributed.network_context import init_distributed_group_network_context, DistributedGroupNetworkContext
from trackit.miscellanies.versioning import generate_app_version_from_git, GitAppVersion
from trackit.miscellanies.machine.cpu_info import get_processor_name


def _update_output_dir(args):
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.run_id)
        os.makedirs(args.output_dir, exist_ok=True)
        print('output directory: ' + args.output_dir)


def _get_version_string(version: GitAppVersion):
    version_string = f"{version.sha}-{version.branch}"
    if version.has_diff:
        version_string += "-dirty"
    return version_string


def _print_machine_information(device: str):
    device = torch.device(device)
    print(f'CPU: {get_processor_name()}')
    if 'cuda' == device.type and torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(device)}')


class GlobalContext:
    def __init__(self, network_context: DistributedGroupNetworkContext, wandb_instance: Optional[wandb.sdk.wandb_run.Run]):
        self.network_context = network_context
        self.wandb_instance = wandb_instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # cleanup_torch_distributed()
        if self.wandb_instance is not None:
            self.wandb_instance.__exit__(exc_type, exc_val, exc_tb)


def init_global_context(runtime_vars, config: dict):
    init_torch_distributed(runtime_vars.device, not runtime_vars.allow_non_master_node_print)

    if not runtime_vars.quiet:
        print('version: ' + _get_version_string(generate_app_version_from_git()))
        _print_machine_information(runtime_vars.device)

    network_context = init_distributed_group_network_context()
    print(f'IP: {network_context.local.ip}')
    print(f'Hostname: {network_context.local.hostname}')
    if network_context.group is not None:
        print('Distributed Group:')
        host_names = {network_context.ip: network_context.hostname for network_context in network_context.group}
        for context in network_context.group:
            print(f'  {context.ip} ({context.hostname})')
    else:
        host_names = {network_context.local.ip: network_context.local.hostname}

    if not runtime_vars.do_sweep:
        _update_output_dir(runtime_vars)
    wandb_instance = None

    if not runtime_vars.disable_wandb:
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
                _update_output_dir(runtime_vars)
        else:
            if is_main_process():
                from .setup_wandb import setup_wandb
                wandb_instance = setup_wandb(runtime_vars, config, str(host_names), wandb_tags)
                if runtime_vars.do_sweep:
                    run_id = wandb_instance.id
                    torch.distributed.broadcast_object_list([run_id])
                    runtime_vars.run_id = run_id
                    _update_output_dir(runtime_vars)
            else:
                if runtime_vars.do_sweep:
                    run_id = [None]
                    torch.distributed.broadcast_object_list(run_id)
                    run_id = run_id[0]
                    runtime_vars.run_id = run_id
                    _update_output_dir(runtime_vars)
    return GlobalContext(network_context, wandb_instance)
