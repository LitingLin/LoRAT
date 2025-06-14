import os
from .interface import get_current_os_interface, OSInterface
from typing import Iterable


def convert_win32_path_to_posix_path(path: str):
    return path.replace('\\', '/')


def convert_posix_path_to_win32_path(path: str):
    return path.replace('/', '\\')


def join_paths(*args: Iterable[str], do_normalization: bool = True):
    path = os.path.join(*args)
    if do_normalization:
        if get_current_os_interface() == OSInterface.Win32:
            path = convert_posix_path_to_win32_path(path)
        path = os.path.abspath(path)
    return path
