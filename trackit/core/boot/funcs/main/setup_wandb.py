import copy
import os
import wandb

from trackit.miscellanies.flatten_dict.flattern_dict import flatten
from trackit.miscellanies.versioning import get_app_version_string
from trackit.miscellanies.torch.distributed import is_dist_initialized, get_rank, get_local_rank, get_local_world_size


def setup_wandb(args, network_config: dict, notes: str, extra_tags: list | None = None):
    from packaging.version import Version
    if Version('0.18') <= Version(wandb.__version__) < Version('0.21'):
        wandb.require("legacy-service")
    tags = None
    if extra_tags is not None:
        tags = copy.deepcopy(extra_tags)
    if 'tags' in network_config['logging']:
        if tags is None:
            tags = copy.deepcopy(network_config['logging']['tags'])
        else:
            tags.extend(network_config['logging']['tags'])
    mode = 'online' if not args.wandb_run_offline else 'offline'

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.root_path, 'logging')
    os.makedirs(output_dir, exist_ok=True)

    group = None
    config = None
    project = None
    run_id = None

    if not hasattr(args, 'do_sweep') or not args.do_sweep:
        project = network_config['logging']['category']
        run_id = args.run_id

        network_config = copy.deepcopy(network_config)
        assert 'runtime_vars' not in network_config
        network_config['runtime_vars'] = vars(args)

        config = flatten(network_config, reducer='dot', enumerate_types=(list,))
        config['git_version'] = get_app_version_string()

        if hasattr(args, 'wandb_distributed_aware') and args.wandb_distributed_aware and is_dist_initialized():
            group = run_id
            run_id = run_id + f'-rank{get_rank() // get_local_world_size()}.{get_local_rank()}'

        wandb_id_max_length = 128
        if len(run_id) > wandb_id_max_length:
            from datetime import datetime
            datetime_str_format = "%Y.%m.%d-%H.%M.%S-%f"
            try:
                # assume run_id ends with datetime with the same format
                run_id_split = run_id.split('-')
                run_id_time = datetime.strptime('-'.join(run_id_split[-3:]), datetime_str_format)
                datetime_str = datetime.strftime(run_id_time, datetime_str_format)
                run_id = '-'.join(run_id_split[:-3])
            except ValueError:
                datetime_str = datetime.strftime(datetime.now(), datetime_str_format)
            run_id = run_id[:wandb_id_max_length - len(datetime_str) - 1] + '-' + datetime_str
            print(f'warning(wandb): id is too long, shorten to {run_id}')

    wandb_instance = wandb.init(project=project, tags=tags, config=config, force=True, job_type='train', id=run_id,
                                mode=mode, dir=output_dir, group=group, notes=notes, resume='auto')
    return wandb_instance
