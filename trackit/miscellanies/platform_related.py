import os
import enum
import platform
from functools import lru_cache


class OSInterface(enum.Enum):
    Win32 = enum.auto()
    Posix = enum.auto()


class OS(enum.Enum):
    Windows = enum.auto()
    macOS = enum.auto()
    Linux = enum.auto()


@lru_cache(maxsize=None)
def get_os() -> OS:
    platform_string = platform.platform()
    if 'Windows' in platform_string or 'NT' in platform_string:
        return OS.Windows
    elif 'Darwin' in platform_string:
        return OS.macOS
    elif 'Linux' in platform_string:
        return OS.Linux
    else:
        raise NotImplementedError()


@lru_cache(maxsize=None)
def get_os_interface():
    if os.name == 'nt':
        return OSInterface.Win32
    elif os.name == 'posix':
        return OSInterface.Posix
    raise NotImplementedError()


def get_platform_style_path(path):
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path


def join_path(*args):
    path = os.path.abspath(os.path.join(*args))
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path
