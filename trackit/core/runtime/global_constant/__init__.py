import shutil
import os
from typing import Sequence, Mapping, Optional, Any
from types import MappingProxyType
from ..utils.custom_yaml_loader import load_yaml

_global_constants: Optional[Mapping] = None
_sentinel = object()  # A unique object to detect if a default was not provided.
__root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def _initialize_global_constants():
    global _global_constants
    constants_config_file_path = os.path.join(__root_path, 'consts.yaml')
    if not os.path.exists(constants_config_file_path):
        shutil.copy(os.path.join(__root_path, 'consts.yaml.template'), constants_config_file_path)
        print('consts.yaml not found, copied from template', flush=True)

    _global_constants = MappingProxyType(load_yaml(constants_config_file_path))


def _get_value(constants: Mapping, paths: Sequence[str], *, default: Any):
    """
    Traverses a nested mapping to retrieve a value.

    Args:
        constants: The mapping to traverse.
        paths: A sequence of keys representing the path to the value.
        default: The value to return if the key path is not found.

    Returns:
        The retrieved value or the default.

    Raises:
        KeyError: If the key path is not found and no default was provided.
        TypeError: If an intermediate value in the path is not a mapping.
    """
    try:
        current_level = constants
        for path in paths:
            current_level = current_level[path]
        return current_level
    except (KeyError, TypeError):
        if default is _sentinel:
            # If no default was provided, re-raise with a more helpful message.
            raise KeyError(f'Key path "{".".join(paths)}" not found in global constants.')
        # A default was provided, so return it.
        return default


def get_global_constant(*paths: str, default: Any = _sentinel) -> Any:
    """
    Retrieves a configuration value from the global constants.

    This function allows accessing nested values by providing keys as separate
    arguments. For example, `get_global_constant('num_train_workers')`.

    Args:
        *paths: A sequence of strings representing the nested keys.
        default: An optional value to return if the key path is not found.
                 If not provided, a KeyError will be raised for missing keys.

    Returns:
        The requested configuration value, or the default value if provided.

    Raises:
        KeyError: If the key path is not found and no default value is specified.
    """
    if _global_constants is None:
        _initialize_global_constants()

    # The `_global_constants` is guaranteed to be initialized here,
    # but a type checker might not know that.
    return _get_value(_global_constants, paths, default=default)
