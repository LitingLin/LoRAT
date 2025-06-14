import os.path
import sys
from subprocess import list2cmdline
from trackit.core.third_party.vot.prepare_workspace import prepare_vot_workspace
from trackit.core.third_party.vot.vot_launcher import launch_vot_evaluation, launch_vot_analysis, launch_vot_pack


def vot_main(args):
    if args.run_id is None:
        from trackit.core.boot.funcs.utils.run_id import generate_run_id
        args.run_id = generate_run_id(args, ('vot',))
    args.output_path = os.path.join(args.output_path, args.run_id)
    vot_workspace = os.path.join(args.output_path, 'vot_workspace')

    if args.tracker_name is None:
        args.tracker_name = '-'.join((args.method_name, args.config_name))

    vot_run_command = [sys.executable, os.path.join(args.root_path, 'main.py'), args.method_name, args.config_name,
                       '--device', args.device, '--disable_wandb', '--quiet',
                       '--run_id', args.run_id,
                       '--output_dir', os.path.join(args.output_path, 'output')]
    if not args.enable_file_logging:
        vot_run_command.append('--disable_file_logging')
    if args.mixin_config is not None:
        for mixin_config in args.mixin_config:
            vot_run_command.extend(['--mixin_config', mixin_config])
    if args.weight_path is not None:
        vot_run_command.extend(['--weight_path', args.weight_path])

    trax_timeout = 18000

    print('Preparing VOT workspace...', end='')
    prepare_vot_workspace(vot_workspace, args.tracker_name, list2cmdline(vot_run_command),
                          args.vot_stack, args.root_path, trax_timeout)
    print('done.')

    print('===================== Begin VOT evaluation =====================', flush=True)
    launch_vot_evaluation(vot_workspace, args.tracker_name)
    print('===================== End VOT evaluation =====================', flush=True)

    print('===================== Begin VOT analysis =====================', flush=True)
    launch_vot_analysis(vot_workspace)
    print('===================== End VOT analysis =====================', flush=True)

    print('===================== Begin VOT packing =====================', flush=True)
    launch_vot_pack(vot_workspace, args.tracker_name)
    print('===================== End VOT packing =====================', flush=True)
