import copy
import torch.nn as nn


def clone_module_with_shared_parameters_and_buffers(module_to_clone: nn.Module) -> nn.Module:
    """
    Creates a new instance of a module, sharing its parameters and buffers.

    This function performs a deep copy of the module's architecture (layers,
    submodules, etc.) but ensures that the actual torch.Tensor objects for
    parameters and buffers are shared between the original module and the
    cloned module.

    This means:
    - The cloned module will have the same structure as the original.
    - The `torch.Tensor` objects for `parameters()` and `buffers()` in the
      cloned module will be the *exact same objects* as in the original module.
    - Consequently, if a parameter or buffer is modified in-place in one
      module, the change will be reflected in the other.
    - This is useful for scenarios where you need multiple module instances
      operating on the exact same underlying numerical data.

    The sharing is achieved by populating the `memo` argument of
    `copy.deepcopy` with the original module's parameters and buffers. When
    `deepcopy` encounters an object already in `memo`, it uses the memoized
    object instead of creating a new copy.

    Args:
        module_to_clone (nn.Module): The PyTorch module to clone.
                                     The original name in the prompt was `model`.

    Returns:
        nn.Module: A new module instance whose parameters and buffers are
                   the same tensor objects as those in `module_to_clone`.
    """
    memo = {}
    # Add all parameters to the memo.
    # When deepcopy encounters these, it will use the existing tensor
    # from the memo instead of creating a new one.
    for param in module_to_clone.parameters():
        memo[id(param)] = param

    # Add all buffers to the memo for the same reason.
    for buffer in module_to_clone.buffers():
        memo[id(buffer)] = buffer

    # Perform deepcopy. The module structure will be copied, but parameters
    # and buffers (whose IDs are in memo) will be reused (shared).
    cloned_module = copy.deepcopy(module_to_clone, memo=memo)

    return cloned_module
