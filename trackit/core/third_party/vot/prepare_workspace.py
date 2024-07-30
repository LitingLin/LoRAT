from .supported_stacks import vot_stacks
from trackit.miscellanies.dump_yaml import dump_yaml
from trackit.core.runtime.global_constant import get_global_constant
import os


def prepare_vot_workspace(workspace_path: str, tracker_name: str, tracker_launch_command: str, vot_stack_name: str,
                          root_path: str, trax_timeout: int):
    dataset_path = get_global_constant(vot_stacks[vot_stack_name].path_name)
    assert os.path.exists(dataset_path), f'Dataset path {dataset_path} does not exist.'

    os.makedirs(workspace_path, exist_ok=True)
    os.mkdir(os.path.join(workspace_path, 'results'))
    try:
        os.symlink(dataset_path, os.path.join(workspace_path, 'sequences'), target_is_directory=True)
    except OSError:
        import shutil
        shutil.copytree(dataset_path, os.path.join(workspace_path, 'sequences'))

    dump_yaml({'registry': ['./trackers.ini'], 'stack': vot_stack_name}, os.path.join(workspace_path, 'config.yaml'))

    with open(os.path.join(workspace_path, 'trackers.ini'), 'w', newline='\n') as f:
        f.write(f'[{tracker_name}]\n')
        f.write(f'label = {tracker_name}\n')
        f.write('protocol = trax\n')
        f.write(f'command = {tracker_launch_command}\n')
        f.write(f'env_PYTHONPATH = {root_path}\n')
        f.write(f'env_PYTHONUNBUFFERED = 1\n')
        f.write(f'timeout = {trax_timeout}')
