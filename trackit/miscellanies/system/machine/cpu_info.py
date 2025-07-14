import platform
import subprocess
import sys
import typing
from functools import lru_cache
import psutil

from trackit.miscellanies.system.operating_system import get_os_running_on, OperatingSystem


is_64bits = sys.maxsize > 2**32
is_x86 = platform.machine() in ("i386", "AMD64", "x86_64")
is_arm = platform.machine().startswith('arm')


def get_cpu_name_from_registry():
    try:
        import winreg
        reg_key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_key_path)

        cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")

        winreg.CloseKey(key)

        return cpu_name.strip()
    except FileNotFoundError:
        return "CPU name not found in registry."
    except Exception as e:
        return f"Error accessing registry: {e}"

@lru_cache(maxsize=None)
def get_processor_name() -> str:
    if get_os_running_on() == OperatingSystem.Windows:
        return get_cpu_name_from_registry()
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


class x86CPUFeatures(typing.NamedTuple):
    has_avx: bool
    has_avx2: bool
    has_avx512f: bool
    has_avx512cd: bool
    has_avx512bw: bool
    has_avx512dq: bool
    has_avx512vl: bool
    has_avx512_fp16: bool
    has_avx512_bf16: bool
    has_amx_fp16: bool
    has_amx_bf16: bool


class ARMCPUFeatures(typing.NamedTuple):
    has_fp16: bool
    has_bf16: bool

@lru_cache(maxsize=None)
def get_arm_cpu_features():
    if get_os_running_on() == OperatingSystem.Linux:
        flags = _get_linux_cpu_flags()
        has_fphp = b'fphp' in flags
        has_bf16 = b'bf16' in flags
        return ARMCPUFeatures(has_fphp, has_bf16)
    elif get_os_running_on() == OperatingSystem.macOS:
        command = ["/usr/sbin/sysctl", "-a", "hw.optional"]
        flags = subprocess.check_output(command).strip().split(b'\n')
        has_fp16 = False
        has_bf16 = False
        for flag in flags:
            if flag.startswith(b'hw.optional.arm.FEAT_FP16'):
                has_fp16 = flag.endswith(b'1')
            if flag.startswith(b'hw.optional.arm.FEAT_BF16'):
                has_bf16 = flag.endswith(b'1')
        return ARMCPUFeatures(has_fp16, has_bf16)
    elif get_os_running_on() == OperatingSystem.Windows:
        return ARMCPUFeatures(True, False)
    else:
        raise NotImplementedError("ARM CPU features detection is not implemented for this OS.")


@lru_cache(maxsize=None)
def get_x86_cpu_features() -> x86CPUFeatures:
    has_avx = False
    has_avx2 = False
    has_avx512f = False
    has_avx512cd = False
    has_avx512bw = False
    has_avx512dq = False
    has_avx512vl = False
    has_avx512_fp16 = False
    has_avx512_bf16 = False
    has_amx_fp16 = False
    has_amx_bf16 = False

    if get_os_running_on() == OperatingSystem.Windows:
        from .cpuid import CPUID

        cpuid = CPUID()
        eax, ebx, ecx, edx = cpuid(eax=1, ecx=0)
        has_avx = (ecx & (1 << 28)) != 0
        eax, ebx, ecx, edx = cpuid(eax=7, ecx=0)
        has_avx2 = (ebx & (1 << 5)) != 0
        has_avx512f = (ebx & (1 << 16)) != 0
        has_avx512cd = (ebx & (1 << 28)) != 0
        has_avx512bw = (ebx & (1 << 30)) != 0
        has_avx512dq = (ebx & (1 << 17)) != 0
        has_avx512vl = (ebx & (1 << 31)) != 0
        has_amx_bf16 = (edx & (1 << 22)) != 0
        has_avx512_fp16 = (edx & (1 << 23)) != 0
        eax, ebx, ecx, edx = cpuid(eax=7, ecx=1)
        has_avx512_bf16 = (eax & (1 << 5)) != 0
        has_amx_fp16 = (eax & (1 << 21)) != 0
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
        flags = _get_linux_cpu_flags()

        has_avx = b'avx' in flags
        has_avx2 = b'avx2' in flags
        has_avx512f = b'avx512f' in flags
        has_avx512cd = b'avx512cd' in flags
        has_avx512bw = b'avx512bw' in flags
        has_avx512dq = b'avx512dq' in flags
        has_avx512vl = b'avx512vl' in flags
        has_avx512_fp16 = b'avx512_fp16' in flags
        has_avx512_bf16 = b'avx512_bf16' in flags
        has_amx_fp16 = b'amx_fp16' in flags
        has_amx_bf16 = b'amx_bf16' in flags
    return x86CPUFeatures(has_avx, has_avx2, has_avx512f, has_avx512cd, has_avx512bw, has_avx512dq, has_avx512vl,
                          has_avx512_fp16, has_avx512_bf16, has_amx_fp16, has_amx_bf16)


def _get_linux_cpu_flags():
    with open('/proc/cpuinfo', 'rb') as f:
        for line in f:
            line = line.strip()
            if line.startswith(b'Features'):
                return line.split(b':', 1)[1].strip().split(b' ')
    raise RuntimeError()


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
    except FileNotFoundError:
        return 0
    cgroup_num_cpus = cfs_quota_us // cfs_period_us
    return cgroup_num_cpus


def get_num_cpu_cores(logical=False):
    if get_os_running_on() == OperatingSystem.Linux:
        num_cpu_cores = _get_cgroup_cpu_limit()
        if num_cpu_cores <= 0:
            num_cpu_cores = psutil.cpu_count(logical=logical)
    else:
        num_cpu_cores = psutil.cpu_count(logical=logical)
    return num_cpu_cores
