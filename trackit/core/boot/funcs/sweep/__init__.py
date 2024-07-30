import math
import torch.distributed
import os
from trackit.miscellanies.torch.distributed import is_main_process, is_dist_initialized
from ...funcs.utils.custom_yaml_loader import load_yaml
from ...funcs.mixin import apply_mixin_rules


def get_sweep_config(args):
    if args.sweep_config is not None:
        sweep_config = args.sweep_config + '.yaml'
        if args.sweep_config.startswith('/' or '\\'):
            config_path = os.path.join(args.config_path, sweep_config)
        else:
            config_path = os.path.join(args.config_path, args.method_name, args.config_name, 'sweep', sweep_config)
    else:
        config_path = os.path.join(args.config_path, args.method_name, args.config_name, 'sweep', 'sweep.yaml')
    config = load_yaml(config_path)
    return config


def prepare_sweep(args, wandb_instance, config):
    if is_main_process():
        assert wandb_instance is not None, "wandb must be enabled for hyper-parameter search"
        this_run_config = wandb_instance.config.as_dict()
    else:
        this_run_config = None
    if is_dist_initialized():
        object_list = [this_run_config]
        torch.distributed.broadcast_object_list(object_list, src=0)
        this_run_config, = object_list
    sweep_config = get_sweep_config(args)
    apply_mixin_rules(sweep_config['mixin'], config, this_run_config)
