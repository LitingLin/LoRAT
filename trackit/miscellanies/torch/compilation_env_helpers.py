import os
import sys
from trackit.miscellanies.machine.cpu_info import is_x86, is_arm, is_64bits, get_cpu_features


_is_env_setup = False
is_windows = sys.platform == 'win32'  # assuming using msvc on windows


def is_anaconda_dist():
    return os.path.exists(os.path.join(sys.prefix, 'conda-meta'))


def _apply_envs(envs):
    for key, value in envs.items():
        os.environ[key] = value


def _get_anaconda_vc_version():
    conda_meta_path = os.path.join(sys.prefix, 'conda-meta')
    conda_packages = os.listdir(conda_meta_path)
    conda_packages = [conda_package for conda_package in conda_packages if
                      conda_package.startswith('vc') and conda_package.endswith('.json')]
    assert len(conda_packages) == 1
    vc_package_path = os.path.join(conda_meta_path, conda_packages[0])
    import json
    with open(vc_package_path) as fid:
        vc_package_meta = json.load(fid)
        return vc_package_meta['version']


def setup_compilation_env():
    global _is_env_setup
    if _is_env_setup:
        return

    if is_windows:
        if is_x86 and not is_64bits:
            vc_bat_args = 'x86'
        elif is_x86 and is_64bits:
            vc_bat_args = 'amd64'
        elif is_arm and not is_64bits:
            vc_bat_args = 'x86_arm'
        elif is_arm and is_64bits:
            vc_bat_args = 'arm64'
        else:
            raise NotImplementedError()
        consist_with_anaconda_vc_version = False
        if consist_with_anaconda_vc_version and is_anaconda_dist():
            vc_version = _get_anaconda_vc_version()
            vc_bat_args += ' -vcvars_ver=' + vc_version

        import distutils._msvccompiler
        print('Calling vcvarsall.bat with args: ' + vc_bat_args + '...', end='')
        _apply_envs(distutils._msvccompiler._get_vc_env(vc_bat_args))
        print('done')
    _is_env_setup = True


_extra_cflags = None
_extra_ldflags = None
_extra_nvcc_flags = None


def _get_msvc_arch_flag():
    cpu_features = get_cpu_features()
    if cpu_features.has_avx512f and cpu_features.has_avx512cd and cpu_features.has_avx512bw and cpu_features.has_avx512dq and cpu_features.has_avx512vl:
        return '/arch:AVX512'
    elif cpu_features.has_avx2:
        return '/arch:AVX2'
    elif cpu_features.has_avx:
        return '/arch:AVX'
    else:
        return None


_local_machine_cuda_arch_list = None


def get_local_machine_cuda_arch_list():
    global _local_machine_cuda_arch_list
    if _local_machine_cuda_arch_list is not None:
        return _local_machine_cuda_arch_list
    import torch.cuda
    if not torch.cuda.is_available():
        return None
    else:
        arch_list = set()
        for device_index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(device_index)
            arch_list.add((device.major, device.minor))
        _local_machine_cuda_arch_list = tuple(arch_list)
        return _local_machine_cuda_arch_list


def get_local_machine_cuda_arch_flags():
    arch_list = get_local_machine_cuda_arch_list()
    if arch_list is None:
        return ()
    else:
        return tuple('-gencode=arch=compute_{},code=sm_{}'.format(str(major) + str(minor), str(major) + str(minor)) for major, minor in arch_list)


def set_env_TORCH_CUDA_ARCH_LIST_with_local_machine_cuda_arch_list():
    arch_list = get_local_machine_cuda_arch_list()
    if arch_list is None:
        return
    os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join('{}.{}'.format(major, minor) for major, minor in arch_list)


def unset_env_TORCH_CUDA_ARCH_LIST():
    if 'TORCH_CUDA_ARCH_LIST' in os.environ:
        del os.environ['TORCH_CUDA_ARCH_LIST']


def init_extra_build_flags(fast_math=False, lto=True, with_symbols=False, with_machine_cuda_arch_flags=True):
    global _extra_cflags, _extra_ldflags, _extra_nvcc_flags
    if _extra_cflags is not None:
        return
    if is_windows:
        cflags = ['/O2', '/DNDEBUG']
        if fast_math:
            cflags.append('/fp:fast')
        arch_flag = _get_msvc_arch_flag()
        if arch_flag is not None:
            cflags.append(arch_flag)
        if with_symbols:
            cflags.append('/Zi')

        nvcc_flags = ['-O3', '-DNDEBUG']
        for cflag in cflags:
            nvcc_flags.extend(['-Xcompiler', cflag])

        if lto:
            cflags.append('/GL')  # nvcc with LTCG will cause unexpected runtime behavior
        _extra_cflags = tuple(cflags)

        if with_machine_cuda_arch_flags:
            nvcc_flags.extend(get_local_machine_cuda_arch_flags())
        _extra_nvcc_flags = tuple(nvcc_flags)

        ldflags = ['/OPT:REF', '/OPT:ICF']
        if lto:
            ldflags.append('/LTCG')
        if with_symbols:
            ldflags.append('/DEBUG')
        _extra_ldflags = tuple(ldflags)
    else:
        cflags = ['-O3', '-DNDEBUG', '-march=native', '-ffunction-sections', '-fdata-sections']
        if fast_math:
            cflags.append('-ffast-math')
        if lto:
            cflags.append('-flto')
        if with_symbols:
            cflags.append('-g')
        _extra_cflags = tuple(cflags)

        ldflags = ['-Wl,--gc-sections']
        if lto:
            ldflags.append('-flto')
            ldflags.extend(cflags)
        if with_symbols:
            ldflags.append('-g')
        _extra_ldflags = tuple(ldflags)

        nvcc_flags = ['-O3', '-DNDEBUG']
        for cflag in cflags:
            nvcc_flags += ['-Xcompiler', cflag]
        if with_machine_cuda_arch_flags:
            nvcc_flags.extend(get_local_machine_cuda_arch_flags())
        _extra_nvcc_flags = tuple(nvcc_flags)


def get_extra_cflags():
    init_extra_build_flags()
    return _extra_cflags


def get_extra_ldflags():
    init_extra_build_flags()
    return _extra_ldflags


def get_extra_cuda_cflags():
    init_extra_build_flags()
    return _extra_nvcc_flags
