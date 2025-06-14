import sys


def debugger_attached():
    if 'debugpy' in sys.modules:
        import debugpy
        return debugpy.is_client_connected()
    return sys.gettrace() is not None
