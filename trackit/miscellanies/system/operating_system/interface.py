import enum
from functools import lru_cache
import os


class OSInterface(enum.Enum):
    Win32 = enum.auto()
    Posix = enum.auto()


@lru_cache(maxsize=None)
def get_current_os_interface():
    if os.name == 'nt':
        return OSInterface.Win32
    elif os.name == 'posix':
        return OSInterface.Posix
    raise NotImplementedError()


IS_WIN32 = get_current_os_interface() == OSInterface.Win32