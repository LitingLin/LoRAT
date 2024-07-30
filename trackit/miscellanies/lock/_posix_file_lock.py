import os
import tempfile
import fcntl
from contextlib import contextmanager, nullcontext
import signal, errno


# https://stackoverflow.com/questions/5255220/fcntl-flock-how-to-implement-a-timeout
@contextmanager
def set_syscall_timeout(seconds):
    def timeout_handler(signum, frame):
        pass

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


class FileLock:
    def __init__(self, name: str, lock_directory=None):
        if lock_directory is None:
            lock_directory = tempfile.gettempdir()
        self._file_path = os.path.join(lock_directory, name + '.lock')
        self._fid = None

    def acquire(self, timeout=None):
        assert self._fid is None, "lock has been acquired"
        self._fid = os.open(self._file_path, os.O_WRONLY | os.O_TRUNC, 0o644)
        if timeout is not None:
            with set_syscall_timeout(timeout):
                try:
                    fcntl.flock(self._fid, fcntl.LOCK_EX)
                except IOError as e:
                    if e.errno != errno.EINTR:
                        raise e
                    return False
        else:
            fcntl.flock(self._fid, fcntl.LOCK_EX)
        return True

    def release(self):
        os.unlink(self._file_path)
        os.close(self._fid)
        self._fid = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

