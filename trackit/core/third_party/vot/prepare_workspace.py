from .supported_stacks import vot_stacks
from .vot_launcher import launch_vot_initialize
from trackit.miscellanies.dump_yaml import dump_yaml
from trackit.core.runtime.global_constant import get_global_constant
import os


def prepare_vot_workspace(workspace_path: str, tracker_name: str, tracker_launch_command: str, vot_stack_name: str,
                          root_path: str, trax_timeout: int):
    dataset_path = get_global_constant(vot_stacks[vot_stack_name].path_name)
    os.makedirs(workspace_path, exist_ok=True)
    if len(dataset_path) > 0 and os.path.exists(dataset_path):
        os.mkdir(os.path.join(workspace_path, 'results'))
        try:
            os.symlink(dataset_path, os.path.join(workspace_path, 'sequences'), target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(dataset_path, os.path.join(workspace_path, 'sequences'))
    else:
        print(f"{vot_stacks[vot_stack_name].path_name}: {dataset_path} does not exist. calling vot initialize {vot_stack_name}.")
        launch_vot_initialize(workspace_path, vot_stack_name)

    dump_yaml({'registry': ['./trackers.ini'], 'stack': vot_stack_name}, os.path.join(workspace_path, 'config.yaml'))

    with open(os.path.join(workspace_path, 'trackers.ini'), 'w', newline='') as f:
        f.write(f'[{tracker_name}]\n')
        f.write(f'label = {tracker_name}\n')
        f.write('protocol = trax\n')
        f.write(f'command = {tracker_launch_command}\n')
        f.write(f'env_PYTHONPATH = {root_path}\n')
        f.write(f'env_PYTHONUNBUFFERED = 1\n')
        f.write(f'timeout = {trax_timeout}\n')
        f.write('socket = 1\n')
