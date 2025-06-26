import os

from .funcs.utils.debugging import enable_stack_trace_on_error, enable_rich_logging, enable_strict_numeric_error_handling
from .funcs.utils.disable_multithreading import disable_multithreading
from .funcs.main.load_config import load_config
from .funcs.main.init_global_context import init_global_context
from .funcs.main.init_application import prepare_application


def _remove_ddp_parameter(args):
    del args.distributed_node_rank
    del args.distributed_nnodes
    del args.distributed_nproc_per_node
    del args.distributed_do_spawn_workers


def main(runtime_vars):
    runtime_vars.config_path = os.path.join(runtime_vars.root_path, 'config')
    if runtime_vars.output_dir is None or len(runtime_vars.output_dir) == 0:
        if not runtime_vars.dry_run:
            print('output_dir is not specified, use --dry_run to run without saving results.')
            return

    if runtime_vars.kill_other_python_processes:
        from .funcs.utils.kill_other_python_processes import kill_other_python_processes
        kill_other_python_processes()

    if runtime_vars.use_deterministic_algorithms:
        if 'cuda' in runtime_vars.device:
            # Set CUBLAS workspace config to a fixed size for reproducibility
            # This is important for deterministic behavior in CUDA operations
            # See: https://docs.nvidia.com/cuda/cublas/#results-reproducibility
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if runtime_vars.run_id is None:
        from .funcs.utils.run_id import generate_run_id
        runtime_vars.run_id = generate_run_id(runtime_vars)

    if runtime_vars.instance_id is None:
        from .funcs.utils.string_to_int import string_to_int_sha256
        runtime_vars.instance_id = string_to_int_sha256(runtime_vars.run_id)

    if runtime_vars.distributed_do_spawn_workers:
        if runtime_vars.device != 'cpu':
            disable_multithreading()
        from .funcs.main.torch_distributed_do_spawn_workers import spawn_workers
        return spawn_workers(runtime_vars)

    if runtime_vars.use_deterministic_algorithms:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("torch: Using deterministic algorithms.")

    enable_strict_numeric_error_handling()

    if runtime_vars.enable_rich_logging:
        enable_rich_logging()
    if runtime_vars.enable_stack_trace_on_error:
        enable_stack_trace_on_error()
    _remove_ddp_parameter(runtime_vars)

    if runtime_vars.multiprocessing_start_method_spawn:
        import multiprocessing
        multiprocessing.set_start_method('spawn')

    config = load_config(runtime_vars)

    context = init_global_context(runtime_vars, config)

    with context:
        if runtime_vars.do_sweep:
            from .funcs.sweep import prepare_sweep
            prepare_sweep(runtime_vars, context.wandb_instance, config)

        app = prepare_application(runtime_vars, config, context.wandb_instance)
        app.run()
