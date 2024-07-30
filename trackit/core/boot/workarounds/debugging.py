import torch
import logging
import faulthandler
import os
import code, traceback, signal


# https://stackoverflow.com/questions/132058/showing-the-stack-trace-from-a-running-python-application
def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)


def register_sig_handler():
    signal.signal(signal.SIGUSR1, debug)  # Register handler
    # print stack trace on SIGTERM
    signal.signal(signal.SIGTERM, lambda sig, frame: traceback.print_stack(frame))


def enable_stack_trace_on_error():
    logging.basicConfig(level=logging.INFO)
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True, exit=False)
    if os.name == 'posix':
        torch._C._set_print_stack_traces_on_fatal_signal(True)
        register_sig_handler()
