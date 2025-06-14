from typing import Optional
import gc


class GarbageCollection:
    def __init__(self, type: str = 'auto', frequency: Optional[int] = None):
        assert type in ("auto", "step", "disabled"), "Garbage Collection type must be in ('auto', 'step', 'disabled')"
        self._type = type
        if type == 'step' and frequency is None:
            frequency = 1
        self._frequency = frequency

    def begin(self):
        if self._type == 'auto':
            gc.enable()
        elif self._type == 'disabled':
            gc.collect()
            gc.disable()
        else:
            gc.collect()
            gc.disable()
            self._step = 0

    def run(self, ):
        if self._type == 'step':
            self._step += 1
            if self._step % self._frequency == 0:
                gc.collect()

    def end(self):
        if self._type == 'step':
            gc.enable()
            del self._step
        elif self._type == 'disabled':
            gc.enable()
