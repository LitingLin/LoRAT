import os
import yaml


def has_recovery_file(checkpoint_path: str) -> bool:
    return os.path.exists(get_recovery_file_path(checkpoint_path))


def get_recovery_file_path(checkpoint_path: str) -> str:
    return os.path.join(checkpoint_path, 'recovery.yaml')


def read_recovery_file(recovery_file_path: str):
    """
    Read the recovery file, resolving relative paths to absolute paths.

    Args:
        recovery_file_path (str): The path to the recovery file (typically recovery.yaml).

    Returns:
        Tuple(str, str): A tuple containing the absolute model weight file
                         path and the absolute application state file path.
    """
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader

    with open(recovery_file_path, 'rb') as f:
        object_ = yaml.load(f, Loader=Loader)
    parent_dir = os.path.dirname(recovery_file_path)

    model_path_in_yaml = object_['model']
    state_path_in_yaml = object_['state']

    # Resolve model path
    if os.path.isabs(model_path_in_yaml):
        model_path = model_path_in_yaml
    else:
        model_path = os.path.abspath(os.path.join(parent_dir, model_path_in_yaml))

    # Resolve state path
    if os.path.isabs(state_path_in_yaml):
        state_path = state_path_in_yaml
    else:
        state_path = os.path.abspath(os.path.join(parent_dir, state_path_in_yaml))

    return model_path, state_path


def write_recovery_file(checkpoint_path: str, model_weight_file_path: str, application_state_file_path: str):
    """
    Write the model and state file paths to the recovery file.

    Stores a relative path if the file is a sub-path of checkpoint_path,
    otherwise stores the absolute path.

    Args:
        checkpoint_path (str): The path to the checkpoint directory.
        model_weight_file_path (str): The path to the model weight file.
        application_state_file_path (str): The path to the application state file.
    """
    recovery_file_path = get_recovery_file_path(checkpoint_path)

    # Determine whether to store relative or absolute paths
    model_path_to_store = _get_path_to_store(model_weight_file_path, checkpoint_path)
    state_path_to_store = _get_path_to_store(application_state_file_path, checkpoint_path)

    with open(recovery_file_path, 'w') as f:
        yaml.dump({'model': model_path_to_store, 'state': state_path_to_store}, f)
    return recovery_file_path


def _is_subpath(path: str, base_path: str) -> bool:
    """
    Check if 'path' is a subpath of 'base_path' in a platform-independent way.
    """
    try:
        # Get absolute paths to handle symbolic links and relative paths
        abs_path = os.path.abspath(path)
        abs_base_path = os.path.abspath(base_path)

        # os.path.commonpath works on a sequence of paths
        common_path = os.path.commonpath([abs_path, abs_base_path])
        return common_path == abs_base_path
    except ValueError:
        # This can happen on Windows if paths are on different drives
        return False


def _get_path_to_store(file_path: str, base_path: str) -> str:
    """
    Decide whether to return a relative or absolute path.
    """
    if _is_subpath(file_path, base_path):
        return os.path.relpath(file_path, base_path)
    return os.path.abspath(file_path)
