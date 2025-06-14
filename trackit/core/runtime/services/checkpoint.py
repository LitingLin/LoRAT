import copy
from typing import Sequence, Tuple
from trackit.miscellanies.torch.distributed import is_rank_0_process
from . import CheckpointStatefulObjectRegistry


def get_state_from_registries(checkpoint_stateful_object_registry_list: Sequence[Tuple[str, CheckpointStatefulObjectRegistry]]):
    state_dict = {}
    for name_prefix, registry in checkpoint_stateful_object_registry_list:
        for parameters in registry.list():
            name = name_prefix + parameters.name
            assert name not in state_dict, f"Duplicate name {name} in checkpoint state dict"
            state_dict[name] = (parameters.get_state_fn(), parameters.main_process_only)
    return state_dict


def load_state_to_registries(state_dict: dict, checkpoint_stateful_object_registry_list: Sequence[Tuple[str, CheckpointStatefulObjectRegistry]], strict=False):
    state_dict = copy.copy(state_dict)
    for name_prefix, registry in checkpoint_stateful_object_registry_list:
        for parameters in registry.list():
            name = name_prefix + parameters.name
            if name in state_dict:
                state, is_main_process_only = state_dict[name]
                if is_main_process_only and not is_rank_0_process():
                    raise RuntimeError(f"{name} is main process only, but current process is not main process")
                parameters.set_state_fn(state)
                del state_dict[name]
            elif strict:
                raise RuntimeError(f"Missing key {name} in checkpoint state dict")
    if len(state_dict) > 0:
        unused_keys = []
        for key, (state, is_main_process_only) in state_dict.items():
            if is_main_process_only and not is_rank_0_process():
                continue
            unused_keys.append(key)
        raise RuntimeError(f"Extra keys {unused_keys} in checkpoint state dict")
