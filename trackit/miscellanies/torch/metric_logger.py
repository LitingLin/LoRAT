# modify from https://github.com/facebookresearch/detr/blob/main/util/misc.py
import math
from collections import defaultdict
import numpy as np
import torch
import time
import datetime
from collections import deque
from typing import Optional
from trackit.miscellanies.ema import EMA
from trackit.miscellanies.machine.mem_info import get_mem_rss
from trackit.miscellanies.machine.utils import sizeof_fmt


class ProgressTrackerInterface:
    def reset(self) -> None:
        raise NotImplementedError()

    def update(self, elapsed_time: float) -> None:
        raise NotImplementedError()

    def is_last(self) -> bool:
        raise NotImplementedError()

    def pos(self) -> int:
        raise NotImplementedError()

    def total(self) -> Optional[int]:
        raise NotImplementedError()

    def eta(self) -> Optional[float]:
        raise NotImplementedError()

    def rate(self) -> Optional[float]:
        raise NotImplementedError()


class DefaultProgressTracker(ProgressTrackerInterface):
    def __init__(self, total: Optional[int] = None):
        assert total is None or total > 0
        self._total = total
        self.reset()

    def reset(self):
        self._index = -1
        if self._total is not None:
            self._rate_ema = EMA()
            self._rate = None

    def update(self, elapsed_time: float):
        self._index += 1
        if self._total is not None:
            self._rate = self._rate_ema(elapsed_time)

    def is_last(self):
        if self._total is None:
            return False
        return self._index >= self._total - 1

    def pos(self):
        return self._index

    def total(self):
        return self._total

    def eta(self):
        if self._total is None:
            raise RuntimeError("eta() is not available when total is not specified")
        if self._rate is None:
            return None
        return self._rate * (self._total - self._index - 1)

    def rate(self):  # we have no way to know the batch size by default
        return None


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    @property
    def median(self):
        if self.count == 0:
            return float('nan')
        return np.median(self.deque)

    @property
    def avg(self):
        if self.count == 0:
            return float('nan')
        return np.mean(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return float('nan')
        return self.total / self.count

    @property
    def max(self):
        if self.count == 0:
            return float('nan')
        return max(self.deque)

    @property
    def value(self):
        if self.count == 0:
            return float('nan')
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class LocalMetricLogger(object):
    def __init__(self, delimiter="\t", print_freq=10):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print_freq = print_freq
        self.progress_indicator: Optional[ProgressTrackerInterface] = None
        self.extra_monitoring_items = {}

    def reset(self):
        for meter in self.meters.values():
            meter.reset()
        if self.progress_indicator is not None:
            self.progress_indicator.reset()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        return self.as_string()

    def as_string(self, global_avg=False):
        loss_str = []
        if global_avg:
            for name, meter in self.meters.items():
                loss_str.append(f"{name}: {meter.global_avg:.4f}")
        else:
            for name, meter in self.meters.items():
                loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name: str, window_size: int = 20, format: str = "{median:.4f} ({global_avg:.4f})"):
        self.meters[name] = SmoothedValue(window_size, format)

    def remove_meter(self, name: str):
        del self.meters[name]

    def set_custom_progress_tracker(self, progress_indicator: Optional[ProgressTrackerInterface]):
        self.progress_indicator = progress_indicator

    def enable_monitoring_system_total_resident_set_size(self, name='mem'):
        self.extra_monitoring_items[name] = lambda: sizeof_fmt(get_mem_rss())

    def enable_monitoring_cuda_device_max_memory_allocated(self, name='dev_mem_max', device: Optional[torch.device] = None):
        import torch.cuda
        self.extra_monitoring_items[name] = lambda: sizeof_fmt(torch.cuda.max_memory_allocated(device))

    def enable_monitoring_cuda_device_memory_allocated(self, name='dev_mem', device: Optional[torch.device] = None):
        import torch.cuda
        self.extra_monitoring_items[name] = lambda: sizeof_fmt(torch.cuda.max_memory_allocated(device))

    def enable_monitoring_cpu_percent(self, name='cpu'):
        import psutil
        self.extra_monitoring_items[name] = lambda: f"{psutil.cpu_percent():.2f}%"

    def log_every(self, iterable, header=None):
        if self.progress_indicator is not None:
            progress_tracker = self.progress_indicator
        else:
            if hasattr(iterable, '__len__'):
                total = len(iterable)
            else:
                total = None
            progress_tracker = DefaultProgressTracker(total)
        i = 0
        if not header:
            header = ''
        start_time = time.perf_counter()
        end = time.perf_counter()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        for obj in iterable:
            data_elapsed = time.perf_counter() - end
            data_time.update(data_elapsed)
            yield obj
            iter_elapsed = time.perf_counter() - end
            iter_time.update(iter_elapsed)
            progress_tracker.update(iter_elapsed)
            if i % self.print_freq == 0 or progress_tracker.is_last():
                msg_items = [header]
                log_values = {}

                if progress_tracker.total() is not None:
                    space_fmt = ':' + str(len(str(progress_tracker.total()))) + 'd'
                    msg_items.append('[{index' + space_fmt + '}/{total}]')

                    log_values['total'] = progress_tracker.total()

                    msg_items.append('eta: {eta}')

                    eta_seconds = progress_tracker.eta()
                    if eta_seconds is None:
                        eta_string = 'N/A'
                    else:
                        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                    log_values['eta'] = eta_string
                else:
                    msg_items.append('[{index}]')

                msg_items.extend([
                    '{localtime}',
                    '{meters}',
                    'time: {time}',
                    'data: {data}',
                ])

                localtime = time.asctime(time.localtime())

                log_values.update({
                    'index': progress_tracker.pos(),
                    'localtime': localtime,
                    'meters': str(self),
                    'time': str(iter_time),
                    'data': str(data_time),
                })

                for name, item in self.extra_monitoring_items.items():
                    msg_items.append(f'{name}: {{{name}}}')
                    log_values[name] = item()

                samples_per_second = progress_tracker.rate()
                if samples_per_second is not None:
                    if math.isfinite(samples_per_second):
                        msg_items.append('({rate:.3f} samples/sec)')
                        log_values['rate'] = samples_per_second
                    else:
                        msg_items.append('(N/A samples/sec)')

                log_msg = self.delimiter.join(msg_items)

                print(log_msg.format(**log_values) + '\n', flush=True, end='')
            i += 1
            end = time.perf_counter()
        total_time = time.perf_counter() - start_time
        total_time_string = f'{header} Total time: {datetime.timedelta(seconds=int(total_time))}'
        if progress_tracker.total() is not None:
            total_time_string += f' ({total_time / progress_tracker.total():.4f} s / it)'

        print(total_time_string + '\n', flush=True, end='')
