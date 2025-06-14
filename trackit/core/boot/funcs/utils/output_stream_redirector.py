import sys
import os


class OutputStreamRedirector:
    class StreamWriter:
        def __init__(self, original_stream, parent):
            self.original_stream = original_stream
            self.parent = parent

        def write(self, data):
            self.original_stream.write(data)
            self.parent.log_file.write(data)

        def flush(self):
            self.original_stream.flush()
            self.parent.log_file.flush()

        def close(self):
            self.original_stream.close()
            self.parent.active_streams -= 1
            if self.parent.active_streams <= 0:
                self.parent.log_file.close()

        def __getattr__(self, name):
            # Dynamically forward any other method calls to the original stream
            return getattr(self.original_stream, name)

    class ErrorStreamWriter(StreamWriter):
        def write(self, data):
            self.original_stream.write(data)
            self.parent.log_file.write(data)
            # Error output is typically flushed immediately
            self.parent.log_file.flush()

    def __init__(self, log_path: str | None, mute_stdout: bool = False, mute_stderr: bool = False):
        self.log_path = log_path
        self.mute_stdout = mute_stdout
        self.mute_stderr = mute_stderr
        self._devnull = None

    def __enter__(self):
        self.log_file = open(self.log_path, 'a', encoding='utf-8', newline='') if self.log_path is not None else open(os.devnull, 'w')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active_streams = 2  # Counting both stdout and stderr

        if self.mute_stdout or self.mute_stderr:
            self._devnull = open(os.devnull, 'w')

        stdout_target = self._devnull if self.mute_stdout else sys.stdout
        stderr_target = self._devnull if self.mute_stderr else sys.stderr

        sys.stdout = OutputStreamRedirector.StreamWriter(stdout_target, self)
        sys.stderr = OutputStreamRedirector.ErrorStreamWriter(stderr_target, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Ensure log file is closed if still open
        if self.active_streams > 0:
            self.log_file.close()

        # Close devnull if it was opened
        if self._devnull is not None:
            self._devnull.close()
            self._devnull = None
