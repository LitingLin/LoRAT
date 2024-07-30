import psutil
import os

_run_in_container_and_has_cgroup_mem = False

if os.name == 'posix' and os.path.exists('/sys/fs/cgroup/memory/'):
    if os.path.exists('/run/.containerenv') or os.path.exists('/.dockerenv'):
        _run_in_container_and_has_cgroup_mem = True


def _get_cgroup_mem_stat():
    mem_stat = {}
    with open('/sys/fs/cgroup/memory/memory.stat', 'rb') as f:
        for line in f:
            fields = line.split()
            mem_stat[fields[0]] = int(fields[1])
    return mem_stat


def _get_cgroup_mem_rss():
    stat = _get_cgroup_mem_stat()
    return stat[b'rss'] + stat[b'mapped_file']


def _get_cgroup_mem_limit():
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'rb') as f:
        return int(f.read())


def get_mem_rss():
    if _run_in_container_and_has_cgroup_mem:
        return _get_cgroup_mem_rss()
    else:
        virtual_memory = psutil.virtual_memory()
        return virtual_memory.total - virtual_memory.available


def get_mem_total():
    if _run_in_container_and_has_cgroup_mem:
        return _get_cgroup_mem_limit()
    else:
        return psutil.virtual_memory().total


def get_mem_usage_and_total():
    if _run_in_container_and_has_cgroup_mem:
        return _get_cgroup_mem_rss(), _get_cgroup_mem_limit()
    else:
        virtual_memory = psutil.virtual_memory()
        return virtual_memory.total - virtual_memory.available, virtual_memory.total
