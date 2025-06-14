import psutil
from ..machine.cpu_info import get_num_cpu_cores


def get_cpu_percent():
    all_core_cpu_percent = psutil.cpu_percent(percpu=True)
    return sum(all_core_cpu_percent) / get_num_cpu_cores(logical=True)
