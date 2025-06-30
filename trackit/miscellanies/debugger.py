import sys


def debugger_attached():
    if 'debugpy' in sys.modules:
        import debugpy
        return debugpy.is_client_connected()
    if 'pydevd' in sys.modules:
        import pydevd
        return pydevd.connected
    return False