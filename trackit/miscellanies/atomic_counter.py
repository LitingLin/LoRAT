import platform


if platform.python_implementation() == 'CPython':
    import itertools

    class AtomicCounter:
        def __init__(self, initial=-1):
            self.value = itertools.count(initial + 1)

        def increment(self):
            return next(self.value)

else:
    import threading

    class AtomicCounter:
        def __init__(self, initial=-1):
            self.value = initial
            self.lock = threading.Lock()

        def increment(self):
            with self.lock:
                self.value += 1
                return self.value
