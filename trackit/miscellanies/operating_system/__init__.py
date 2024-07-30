import enum
import platform


class OperatingSystem(enum.Enum):
    Windows = enum.auto()
    macOS = enum.auto()
    Linux = enum.auto()


def get_os_running_on() -> OperatingSystem:
    platform_string = platform.platform()
    if 'Windows' in platform_string or 'NT' in platform_string:
        return OperatingSystem.Windows
    elif 'Darwin' in platform_string or 'macOS' in platform_string:
        return OperatingSystem.macOS
    elif 'Linux' in platform_string:
        return OperatingSystem.Linux
    else:
        raise NotImplementedError()


IS_WINDOWS = get_os_running_on() == OperatingSystem.Windows
