import platform
import subprocess
import sys
import typing
from functools import lru_cache
import psutil

from trackit.miscellanies.operating_system import get_os_running_on, OperatingSystem


is_64bits = sys.maxsize > 2**32
is_x86 = platform.machine() in ("i386", "AMD64", "x86_64")
is_arm = platform.machine().startswith('arm')


@lru_cache(maxsize=None)
def get_processor_name() -> str:
    if get_os_running_on() == OperatingSystem.Windows:
        name = subprocess.check_output(["wmic", "cpu", "get", "name"], universal_newlines=True).strip().split("\n")[-1]
        return name
    elif get_os_running_on() == OperatingSystem.macOS:
        command = ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.check_output(command, universal_newlines=True).strip()
    elif get_os_running_on() == OperatingSystem.Linux:
        with open('/proc/cpuinfo', 'rb') as f:
            for line in f:
                line = line.strip()
                if line.startswith(b'model name'):
                    return line.split(b':', 1)[1].strip().decode('utf-8')
    return ""


class CPUFeatures(typing.NamedTuple):
    has_avx: bool
    has_avx2: bool
    has_avx512f: bool
    has_avx512cd: bool
    has_avx512bw: bool
    has_avx512dq: bool
    has_avx512vl: bool


@lru_cache(maxsize=None)
def get_cpu_features() -> CPUFeatures:
    has_avx = False
    has_avx2 = False
    has_avx512f = False
    has_avx512cd = False
    has_avx512bw = False
    has_avx512dq = False
    has_avx512vl = False

    if is_x86:
        if get_os_running_on() == OperatingSystem.Windows:
            from trackit.miscellanies.machine.cpuid import CPUID

            cpuid = CPUID()
            eax, ebx, ecx, edx = cpuid(1)
            has_avx = (ecx & (1 << 28)) != 0
            eax, ebx, ecx, edx = cpuid(7)
            has_avx2 = (ebx & (1 << 5)) != 0
            has_avx512f = (ebx & (1 << 16)) != 0
            has_avx512cd = (ebx & (1 << 28)) != 0
            has_avx512bw = (ebx & (1 << 30)) != 0
            has_avx512dq = (ebx & (1 << 17)) != 0
            has_avx512vl = (ebx & (1 << 31)) != 0
        elif get_os_running_on() == OperatingSystem.macOS:
            command = ["/usr/sbin/sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"]
            flags = subprocess.check_output(command).strip()

            has_avx = b'AVX1.0' in flags
            has_avx2 = b'AVX2' in flags
            has_avx512f = b'AVX512F' in flags
            has_avx512cd = b'AVX512CD' in flags
            has_avx512bw = b'AVX512BW' in flags
            has_avx512dq = b'AVX512DQ' in flags
            has_avx512vl = b'AVX512VL' in flags
        elif get_os_running_on() == OperatingSystem.Linux:
            def get_cpu_flags():
                with open('/proc/cpuinfo', 'rb') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(b'flags'):
                            return line.split(b':', 1)[1].strip().split(b' ')
                raise RuntimeError()
            flags = get_cpu_flags()

            has_avx = b'avx' in flags
            has_avx2 = b'avx2' in flags
            has_avx512f = b'avx512f' in flags
            has_avx512cd = b'avx512cd' in flags
            has_avx512bw = b'avx512bw' in flags
            has_avx512dq = b'avx512dq' in flags
            has_avx512vl = b'avx512vl' in flags
    return CPUFeatures(has_avx, has_avx2, has_avx512f, has_avx512cd, has_avx512bw, has_avx512dq, has_avx512vl)


# http://donghao.org/2022/01/20/how-to-get-the-number-of-cpu-cores-inside-a-container/
# For physical machine, the `cfs_quota_us` could be '-1'
def _get_cgroup_cpu_limit():
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())
    except RuntimeError:
        return 0
    container_cpus = cfs_quota_us // cfs_period_us
    return container_cpus


def get_num_cpu_cores(logical=False):
    if get_os_running_on() == OperatingSystem.Linux:
        num_cpu_cores = _get_cgroup_cpu_limit()
        if num_cpu_cores <= 0:
            num_cpu_cores = psutil.cpu_count(logical=logical)
    else:
        num_cpu_cores = psutil.cpu_count(logical=logical)
    return num_cpu_cores
