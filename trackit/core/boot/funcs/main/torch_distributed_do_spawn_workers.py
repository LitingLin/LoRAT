import sys
import numpy as np
import torch.distributed.run


def spawn_workers(args):
    command_args = sys.argv

    rdzv_port = np.random.Generator(np.random.PCG64(args.instance_id + 1024)).integers(10000, 50000)

    train_script_path = command_args[0]
    torch_run_args = ['--nnodes', str(args.distributed_nnodes), '--nproc_per_node', str(args.distributed_nproc_per_node),
                      '--rdzv_endpoint', f'{args.master_address}:{rdzv_port}', '--max_restarts', str(args.torchrun_max_restarts),
                      '--rdzv_id', args.run_id, '--node_rank', str(args.distributed_node_rank),
                      '--rdzv_backend', 'static', '--master_addr', args.master_address, '--master_port', str(rdzv_port),
                      train_script_path]

    index_of_arg = 1
    while index_of_arg < len(command_args):
        command_arg = command_args[index_of_arg]
        if command_arg in ('--distributed_nnodes', '--distributed_nproc_per_node', '--run_id',
                           '--distributed_node_rank', "--torchrun_max_restarts"):
            index_of_arg += 2
        elif command_arg in ('--distributed_do_spawn_workers', '--kill_other_python_processes'):
            index_of_arg += 1
        else:
            torch_run_args.append(command_arg)
            index_of_arg += 1
    torch_run_args.extend(
        ['--run_id', args.run_id])
    print(f'Executing torch.distributed.run.main({torch_run_args})', flush=True)
    return torch.distributed.run.main(torch_run_args)
