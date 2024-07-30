import os
import errno


class _AdvisoryFileLockContext:
    def __init__(self, path, file_handle, is_locked):
        self.path = path
        self.file_handle = file_handle
        self._is_locked = is_locked

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()

    def is_locked(self):
        return self._is_locked

    def unlock(self):
        if self._is_locked:
            os.close(self.file_handle)
            os.remove(self.path)
            self._is_locked = False


def try_lock_file(path):
    path = path + '.lock'
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    try:
        file_handle = os.open(path, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            return _AdvisoryFileLockContext(path, 0, False)
        else:  # Something unexpected went wrong so reraise the exception.
            raise
    else:
        return _AdvisoryFileLockContext(path, file_handle, True)
