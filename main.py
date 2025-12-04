import argparse
import os
from trackit.core.boot.main import main


def setup_arg_parser():
    parser = argparse.ArgumentParser('TracKit entry script', add_help=False)
    parser.add_argument('method_name', type=str, help='Name of the method to execute (config/{method_name})')
    parser.add_argument('config_name', type=str, help='Name of the config file to use (config/{method_name}/{config_name})')
    parser.add_argument('--output_dir', help='Directory path where to save outputs, checkpoints, and results, saved in {output_dir}/{run_id}/')
    parser.add_argument('--dry_run', action='store_true', help='Run in dry mode without saving checkpoints and results (debugging purposes)')
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing (default: cuda, options: cuda, cpu, mps, etc.)')
    parser.add_argument('--seed', default=42, type=int, help='Seed for initializing random number generators (default: 42)')
    parser.add_argument('--instance_id', type=int, help='Unique instance identifier for running multi-instance of this program on the same machine')
    parser.add_argument('--weight_path', type=str, action='append',
                       help='Path(s) to model weights (multiple weights can be assigned, load in order)')
    parser.add_argument('--state_path', help='Path to state file (state.tar) for resuming application state')
    parser.add_argument('--resume', help='Path to recovery file (recovery.yaml) containing path to model weight and application state (for resuming training)')
    parser.add_argument('--mixin_config', type=str, action='append',
                        help='Additional configuration files to mix in (multiple mixin configs is allowed, applied in sequential order),'
                             'searching in order of config/{method_name}/{config_name}/mixin/, config/{method_name}/_mixin, '
                             'config/_mixin.')

    parser.add_argument('--pin_memory', action='store_true', help='Move tensors to pinned memory before transferring to GPU (improves performance but uses more CPU memory)')

    parser.add_argument('--enable_wandb', action='store_true', help='Enable Weights & Biases (wandb) experiment tracking and logging')
    parser.add_argument('--wandb_run_offline', action='store_true', help='Run Weights & Biases (wandb) in offline mode)')

    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output and non-essential messages (currently used for fast start up)')
    parser.add_argument('--disable_file_logging', action='store_true', help='Disable logging to file (output only to console)')
    parser.add_argument('--enable_rich_logging', action='store_true', help='Set logging level to INFO')
    parser.add_argument('--enable_stack_trace_on_error', action='store_true', help='Show stack traces when errors occur (for debugging)')
    parser.add_argument('--allow_non_master_node_printing', action='store_true', help='Allow logging output from non-master nodes in distributed training')

    parser.add_argument('--do_sweep', action='store_true', help='Perform hyperparameter sweep using the specified sweep configuration')
    parser.add_argument('--sweep_config', type=str, help='Path to hyperparameter sweep configuration file')

    parser.add_argument('--run_id', type=str, help='Custom run identifier for experiment tracking and output organization (auto-generated if not provided).')

    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                       help='Master node IP address for distributed training (default: 127.0.0.1 for single-node)')
    parser.add_argument('--distributed_node_rank', type=int, default=0,
                       help='Specify current node rank for distributed training (i.e. index of the node) (default: 0)')
    parser.add_argument('--distributed_nnodes', type=int, default=1,
                       help='Total number of nodes participating in distributed training (default: 1)')
    parser.add_argument('--distributed_nproc_per_node', type=int, default=1,
                       help='Number of processes (GPUs) per node for distributed training (default: 1)')
    parser.add_argument('--torchrun_max_restarts', type=int, default=0,
                       help='Maximum number of worker group restarts before failing in distributed training (default: 0)')
    parser.add_argument('--distributed_do_spawn_workers', action='store_true',
                       help='If true, this process is responsible for creating worker group processes in distributed training')

    parser.add_argument('--wandb_distributed_aware', action='store_true',
                       help='Make W&B logging aware of distributed training (enable wandb on all ranks)')
    parser.add_argument('--kill_other_python_processes', action='store_true',
                       help='Kill other Python processes before starting (a rough way to ensure no other instance interfere with this run)')

    parser.add_argument('--multiprocessing_start_method_spawn', action='store_true',
                       help='Use spawn start method for the multiprocessing package (default is fork, which may cause issues with some libraries)')
    parser.add_argument('--enable_profiling', action='store_true',
                       help='Enable performance profiling (cProfile) during execution (may impact performance)')

    parser.add_argument('--use_deterministic_algorithms', action='store_true',
                       help='Use deterministic algorithms for reproducible results (may reduce performance)')

    return parser


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))
    main(args)
