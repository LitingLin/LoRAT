from typing import Union, List
import torch.utils.cpp_extension
from .compilation_env_helpers import setup_compilation_env, get_extra_cflags, get_extra_ldflags, get_extra_cuda_cflags, set_env_TORCH_CUDA_ARCH_LIST_with_local_machine_cuda_arch_list, unset_env_TORCH_CUDA_ARCH_LIST


_cxx_compiler_setup = False


def load_cpp_extension(name: str, sources: Union[str, List[str]], verbose: bool = False):
    global _cxx_compiler_setup
    if not _cxx_compiler_setup:
        setup_compilation_env()
        _cxx_compiler_setup = True
    return torch.utils.cpp_extension.load(name, sources, extra_cflags=list(get_extra_cflags()), extra_ldflags=list(get_extra_ldflags()), verbose=verbose)


def load_cuda_extension(name: str, sources: Union[str, List[str]], arch_match_local_machine: bool = True, verbose: bool = False):
    global _cxx_compiler_setup
    if not _cxx_compiler_setup:
        setup_compilation_env()
        _cxx_compiler_setup = True
    if arch_match_local_machine:
        set_env_TORCH_CUDA_ARCH_LIST_with_local_machine_cuda_arch_list()

    module = torch.utils.cpp_extension.load(name, sources, extra_cflags=list(get_extra_cflags()), extra_ldflags=list(get_extra_ldflags()), extra_cuda_cflags=list(get_extra_cuda_cflags()), verbose=verbose)

    if arch_match_local_machine:
        unset_env_TORCH_CUDA_ARCH_LIST()

    return module
