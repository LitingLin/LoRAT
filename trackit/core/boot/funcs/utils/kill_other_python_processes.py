import os


# This function kills all other python processes except the current process and its parent process.
# Used for auto hyper-parameter tuning.
def kill_other_python_processes():
    self_pid = os.getpid()
    parent_pid = os.getppid()
    import psutil
    for proc in psutil.process_iter():
        try:
            process_name = proc.name()
            if process_name == 'python':
                pid = proc.pid
                if pid == self_pid or pid == parent_pid:
                    continue
                else:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
