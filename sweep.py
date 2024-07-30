import os
from trackit.core.boot.sweep import sweep_main


def setup_sweep_argparse():
    import argparse
    arg_parser = argparse.ArgumentParser(description='Hyperparameter tuning script')
    arg_parser.add_argument('method_name', type=str)
    arg_parser.add_argument('config_name', type=str)
    arg_parser.add_argument('--sweep_config', type=str)
    arg_parser.add_argument('--sweep_id', type=str)
    arg_parser.add_argument('--agents_run_limit', type=int)
    arg_parser.add_argument('--run_id', type=str)
    arg_parser.add_argument('--output_dir', type=str)
    arg_parser.add_argument('--mixin_config', type=str, action='append')
    return arg_parser


if __name__ == '__main__':
    args, unknown_args = setup_sweep_argparse().parse_known_args()
    args.root_path = os.path.dirname(__file__)
    sweep_main(args, unknown_args)
