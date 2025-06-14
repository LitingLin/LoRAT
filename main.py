import argparse
import os
from trackit.core.boot.main import main


def setup_arg_parser():
    parser = argparse.ArgumentParser('Set runtime parameters', add_help=False)
    parser.add_argument('method_name', type=str, help='Method name')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('--output_dir', help='path where to save')
    parser.add_argument('--dry_run', action='store_true', help='do not save checkpoints and results')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--instance_id', type=int)
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--quiet', action='store_true', help='do not generate unnecessary messages.')

    parser.add_argument('--pin_memory', action='store_true', help='move tensors to pinned memory before transferring to GPU')

    parser.add_argument('--enable_wandb', action='store_true', help='enable wandb')
    parser.add_argument('--wandb_run_offline', action='store_true', help='run wandb offline')

    parser.add_argument('--disable_file_logging', action='store_true', help='disable file logging')
    parser.add_argument('--enable_rich_logging', action='store_true', help='enable rich logging')
    parser.add_argument('--enable_stack_trace_on_error', action='store_true', help='enable stack trace on error')
    parser.add_argument('--allow_non_master_node_printing', action='store_true', help='enable logging on non-master nodes')

    parser.add_argument('--do_sweep', action='store_true')
    parser.add_argument('--sweep_config', type=str)

    parser.add_argument('--mixin_config', type=str, action='append')

    parser.add_argument('--run_id', type=str)

    # distributed training parameters
    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--distributed_node_rank', type=int, default=0)
    parser.add_argument('--distributed_nnodes', type=int, default=1)
    parser.add_argument('--distributed_nproc_per_node', type=int, default=1)
    parser.add_argument('--distributed_do_spawn_workers', action='store_true')

    parser.add_argument('--wandb_distributed_aware', action='store_true')
    parser.add_argument('--kill_other_python_processes', action='store_true')
    parser.add_argument('--multiprocessing_start_method_spawn', action='store_true')
    parser.add_argument('--weight_path', type=str, action='append')

    parser.add_argument('--enable_profiling', action='store_true')

    return parser


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))
    main(args)
