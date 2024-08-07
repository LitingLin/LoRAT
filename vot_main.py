import os
from trackit.core.boot.vot import vot_main
from trackit.core.third_party.vot.supported_stacks import vot_stacks
import argparse


def setup_arg_parser():
    arg_parser = argparse.ArgumentParser('Set runtime parameters')
    arg_parser.add_argument('vot_stack', type=str, choices=list(vot_stacks.keys()), help='VOT stack')
    arg_parser.add_argument('method_name', type=str)
    arg_parser.add_argument('config_name', type=str)
    arg_parser.add_argument('output_path', type=str, help='VOT workspace')
    arg_parser.add_argument('--tracker_name', type=str)
    arg_parser.add_argument('--mixin_config', type=str, action='append')
    arg_parser.add_argument('--device', type=str, default='cuda:0', help="pytorch device string.")
    arg_parser.add_argument('--run_id', type=str)
    arg_parser.add_argument('--weight_path', type=str)
    return arg_parser


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))
    vot_main(args)
