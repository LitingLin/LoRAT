import os

if os.name == 'nt':
    from ._win32_named_mutex import NamedMutex
    NamedInterprocessLock = NamedMutex
elif os.name == 'posix':
    from ._posix_file_lock import FileLock
    NamedInterprocessLock = FileLock
else:
    raise RuntimeError('Unsupported platform')
