#!/bin/bash

print_help() {
    echo "Usage: $0 <method_name> <config_name> [options]"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message and exit"
    echo "  --resume <path>            Resume from recovery file"
    echo "  --no_pin_memory            Disable pin memory"
    echo "  --pin_memory               Enable pin memory (default)"
    echo "  --device_ids <ids>         Specify GPU device IDs"
    echo "  --wandb_offline            Run wandb in offline mode"
    echo "  --output_dir <dir>         Specify output directory"
    echo "  --disable_wandb            Disable wandb logging"
    echo "  --enable_wandb             Enable wandb logging (default)"
    echo "  --date <date>              Specify date for run ID"
    echo "  --do_sweep                 Perform hyperparameter sweep"
    echo "  --alt_sweep_config <path>  Specify alternative sweep config"
    echo "  --sweep_run_times <num>    Number of sweep runs"
    echo "  --sweep_id <id>            Specify sweep ID"
    echo "  --run_id <id>              Specify run ID"
    echo "  --branch <branch>          Specify git branch"
    echo "  --mixin <config>           Add mixin config"
    echo "  --max_restarts <num>       Maximum number of restarts for torchrun (only for distributed training)"
    echo "  --nnodes <num>             Number of nodes for distributed training"
    echo "  --instance <id>            Instance ID"
    echo "  --node_rank <rank>         Node rank for distributed training"
    echo "  --master_address <address> Master node address"
    echo "  --seed <seed>              Set random seed"
    echo "  --disable_ib               Disable InfiniBand"
    echo "  --enable_ib                Enable InfiniBand (default)"
    echo "  --weight_path <path>       Specify weight path"
    echo "  --checkout <hash>          Checkout specific git commit"
    echo "  --enable_file_logging      Enable file logging (default)"
    echo "  --disable_file_logging     Disable file logging"
    echo "  --enable_profiling         Enable profiling"
    echo "  --use_deterministic_algorithms  Use deterministic algorithms (default: false)"
}

if [[ $# -eq 0 ]]; then
    print_help
    exit 1
fi

output_dir=""

use_git=false
pin_memory=true
wandb_offline=false
do_sweep=false
max_restarts=0
use_deterministic_algorithms=false
enable_wandb=true
disable_ib=false
disable_file_logging=false
enable_profiling=false
debug=false
multiprocessing_start_method_spawn=false

method_name=$1
config_name=$2
shift
shift

declare -a mixin_config
declare -a weight_paths

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) print_help; exit 0 ;;
        --resume) resume_file_path="${2}"; shift ;;
        --no_pin_memory) pin_memory=false ;;
        --pin_memory) pin_memory=true ;;
        --device_ids) device_ids="$2"; shift ;;
        --wandb_offline) wandb_offline=true ;;
        --output_dir) output_dir="$2"; shift ;;
        --disable_wandb) enable_wandb=false ;;
        --enable_wandb) enable_wandb=true ;;
        --date) DATE_WITH_TIME="$2"; shift ;;
        --do_sweep) do_sweep=true ;;
        --alt_sweep_config) sweep_config_path="${2}"; shift ;;
        --sweep_run_times) sweep_run_times=$2; shift ;;
        --sweep_id) sweep_id="$2"; shift ;;
        --run_id) run_id="$2"; shift ;;
        --branch) git_branch="$2"; shift ;;
        --mixin|--mixin_config) mixin_config+=("$2"); shift ;;
        --nnodes) NUM_NODES=$2; shift ;;
        --instance) instance_id=$2; shift ;;
        --node_rank) NODE_RANK=$2; shift ;;
        --master_address) MASTER_ADDRESS="$2"; shift ;;
        --seed) seed=$2; shift ;;
        --max_restarts) max_restarts=$2; shift ;;
        --disable_ib) disable_ib=true ;;
        --enable_ib) disable_ib=false ;;
        --weight_path) weight_paths+=("$2"); shift ;;
        --checkout) git_commit_hash="$2"; shift ;;
        --disable_file_logging) disable_file_logging=true ;;
        --enable_file_logging) disable_file_logging=false ;;
        --enable_profiling) enable_profiling=true ;;
        --multiprocessing_start_method_spawn) multiprocessing_start_method_spawn=true ;;
        --use_deterministic_algorithms) use_deterministic_algorithms=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$output_dir" ]]; then
    echo "Please specify the output directory"
    exit 1
fi

if [[ -z "$device_ids" ]]
then
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    nvidia-smi
else
    num_gpus=$(nvidia-smi -i "$device_ids" --query-gpu=name --format=csv,noheader | wc -l)
    nvidia-smi -i "$device_ids"
    export CUDA_VISIBLE_DEVICES="$device_ids"
fi

set -o pipefail

repo_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$repo_path"

if [[ $use_git == true ]]
then
    git pull

    if [[ -n "$git_branch" ]]
    then
        git checkout "$git_branch"
        git pull
    fi

    if [[ -n "$git_commit_hash" ]]
    then
        git checkout "$git_commit_hash"
    fi
fi

if [[ -z "$run_id" ]]; then
    if [[ -z "$DATE_WITH_TIME" ]]; then
        DATE_WITH_TIME=$(date "+%Y.%m.%d-%H.%M.%S-%6N")
    fi
    run_id="$method_name-$config_name"
    for i in "${mixin_config[@]}"
    do
        i=$(basename -- "$i")
        run_id="$run_id-mixin-${i}"
    done
    run_id="$run_id-$DATE_WITH_TIME"
fi

mkdir -p "$output_dir"

target_options=("$method_name" "$config_name")

common_options=("--run_id" "$run_id" "--output_dir" "$output_dir")

if [[ -n "$instance_id" ]]; then
    common_options+=("--instance_id" "$instance_id")
fi
if [[ "$pin_memory" == true ]]; then
    common_options+=("--pin_memory")
fi
if [[ "$disable_ib" == true ]]; then
    export NCCL_IB_DISABLE=1
fi
if [[ "$wandb_offline" == true ]]; then
    common_options+=("--wandb_run_offline")
fi
if [[ "$enable_wandb" == true ]]; then
    common_options+=("--enable_wandb")
fi
if [[ -n "$NUM_NODES" ]]; then
    common_options+=("--distributed_nnodes" "$NUM_NODES")
fi
if [[ -n "$MASTER_ADDRESS" ]]; then
    common_options+=("--master_address" "$MASTER_ADDRESS")
fi
if [[ -n "$seed" ]]; then
    common_options+=("--seed" "$seed")
fi
if [[ "$num_gpus" -gt 1 ]]; then
    common_options+=("--distributed_nproc_per_node" "$num_gpus")
fi
if [[ -n "$NODE_RANK" ]]; then
    common_options+=("--distributed_node_rank" "$NODE_RANK")
fi
if [[ -n "$NUM_NODES" || "$num_gpus" -gt 1 ]]; then
    common_options+=("--distributed_do_spawn_workers")
fi
for weight_path in "${weight_paths[@]}"
do
    common_options+=("--weight_path" "$weight_path")
done
for i in "${mixin_config[@]}"
do
    common_options+=("--mixin_config" "$i")
done
if [[ -n "$resume_file_path" ]]; then
    common_options+=("--resume" "$resume_file_path")
fi
if [[ "$disable_file_logging" == true ]]; then
    common_options+=("--disable_file_logging")
fi
if [[ "$enable_profiling" == true ]]; then
    common_options+=("--enable_profiling")
fi
if [[ "$multiprocessing_start_method_spawn" == true ]]; then
    common_options+=("--multiprocessing_start_method_spawn")
fi
if [[ "$use_deterministic_algorithms" == true ]]; then
    common_options+=("--use_deterministic_algorithms")
fi
if [[ "$max_restarts" -gt 0 ]]; then
    common_options+=("--torchrun_max_restarts" "$max_restarts")
fi
if [[ "$debug" == true ]]; then
    common_options+=("--enable_stack_trace_on_error" "--allow_non_master_node_printing")
fi
if [[ "$do_sweep" == false ]]; then
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 python main.py "${target_options[@]}" "${common_options[@]}"
else
    if [[ -n "$NUM_NODES" && "$NUM_NODES" -gt 1 ]]; then
        echo "Multi-nodes distributed training currently not support for hyper-parameter tunning"
        exit 1
    fi
    sweep_options=()
    if [[ -n "$sweep_config_path" ]]; then
        sweep_options+=("--sweep_config" "$sweep_config_path")
    fi
    if [[ -n "$sweep_id" ]]; then
        sweep_options+=("--sweep_id" "$sweep_id")
    fi
    if [[ -n "$sweep_run_times" ]]; then
        sweep_options+=("--agents_run_limit" "$sweep_run_times")
    fi

    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 python sweep.py "${target_options[@]}" "${sweep_options[@]}" "${common_options[@]}"
fi
