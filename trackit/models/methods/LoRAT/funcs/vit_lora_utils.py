from collections import OrderedDict
from functools import partial
from typing import Mapping, Any, List

import torch
import torch.nn as nn

from ..modules.lora import LinearWithLoRA, LinearWithLoRA_QKVFused, LinearWithLoRA_KVFused
from ..modules.lora.merge import lora_merge


# Modified from https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_frozen_nn_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and not module.weight.requires_grad:
            lora_module_names.add(name)

    return sorted(list(lora_module_names))


def apply_lora(model: nn.Module, lora_module_names: List[str],
               lora_r: int, lora_alpha: float,
               lora_dropout: float = 0.0, use_rslora: bool = False):
    for lora_module_name in lora_module_names:
        *parent_path, child_module_name = lora_module_name.split(".")
        parent_module = model.get_submodule(".".join(parent_path))

        if child_module_name == 'qkv':
            wrapper_cls = LinearWithLoRA_QKVFused
        elif child_module_name == 'kv':
            wrapper_cls = LinearWithLoRA_KVFused
        else:
            wrapper_cls = LinearWithLoRA
        setattr(parent_module, child_module_name,
                wrapper_cls(getattr(parent_module, child_module_name), lora_r,
                            lora_alpha,
                            lora_dropout, use_rslora))


def enable_lora_(self: nn.Module, lora_r: int, lora_alpha: float, lora_dropout: float, use_rslora: bool):
    """
    Patch LoRA for the given model.

    Args:
        self (nn.Module): The model to apply LoRA to.
        lora_r (int): The rank of the LoRA.
        lora_alpha (float): The scaling factor for LoRA.
        lora_dropout (float): The dropout rate for LoRA.
        use_rslora (bool): Whether to use RSLORA or not.
    """
    for i_layer, block in enumerate(self.blocks):
        linear_names = find_all_frozen_nn_linear_names(block)
        apply_lora(block, linear_names, lora_r, lora_alpha, lora_dropout, use_rslora)

    self.lora_alpha = lora_alpha
    self.use_rslora = use_rslora

    self.state_dict = state_dict_with_lora_meta_attributes.__get__(self, self.__class__)
    self.load_state_dict = load_state_dict_with_lora_meta_attributes.__get__(self, self.__class__)

    return self

def state_dict_with_lora_meta_attributes(self, **kwargs):
    state_dict = super(self.__class__, self).state_dict(**kwargs)
    prefix = kwargs.get('prefix', '')
    if self.lora_alpha != 1. or self.use_rslora:
        state_dict[prefix + 'lora_alpha'] = torch.as_tensor(self.lora_alpha)
        state_dict[prefix + 'use_rslora'] = torch.as_tensor(self.use_rslora)
    return state_dict

def load_state_dict_with_lora_meta_attributes(self, state_dict: Mapping[str, Any], **kwargs):
    if 'lora_alpha' in state_dict:
        state_dict = OrderedDict(**state_dict)
        self.lora_alpha = state_dict['lora_alpha'].item()
        self.use_rslora = state_dict['use_rslora'].item()
        del state_dict['lora_alpha']
        del state_dict['use_rslora']
    return super(self.__class__, self).load_state_dict(state_dict, **kwargs)


def attach_lora_state_dict_hooks_(self: nn.Module, lora_alpha: None | float = None, use_rslora: bool = False):
    hooker = Hooker(lora_alpha, use_rslora)
    self._register_load_state_dict_pre_hook(partial(hooker.root_hook, self))
    for block in self.blocks:
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('.qkv'):
                    module._register_load_state_dict_pre_hook(partial(hooker.qkv_linear_hook, module))
                elif name.endswith('.kv'):
                    module._register_load_state_dict_pre_hook(partial(hooker.kv_linear_hook, module))
                else:
                    module._register_load_state_dict_pre_hook(partial(hooker.linear_hook, module))

    return self


class Hooker:
    def __init__(self, lora_alpha: None | float = None, use_rslora: bool = False):
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora

    def root_hook(self, module, state_dict, prefix, *_):
        if 'lora_alpha' in state_dict:
            self.lora_alpha = state_dict[prefix + 'lora_alpha']
            if isinstance(self.lora_alpha, torch.Tensor):
                self.lora_alpha = self.lora_alpha.item()
            del state_dict[prefix + 'lora_alpha']
        if 'use_rslora' in state_dict:
            self.use_rslora = state_dict[prefix + 'use_rslora']
            if isinstance(self.use_rslora, torch.Tensor):
                self.use_rslora = self.use_rslora.item()
            del state_dict[prefix + 'use_rslora']

    def qkv_linear_hook(self, module: nn.Linear, state_dict: dict, prefix: str, *_):
        merge_required = False
        for name in state_dict.keys():
            if name.startswith(prefix) and name not in (prefix + 'weight', prefix + 'bias'):
                merge_required = True
                break
        if not merge_required:
            return
        dim = module.in_features
        merged_weights, merged_biases = [], []

        handle_bias = module.bias is not None
        if handle_bias:
            handle_bias = any(f"{prefix}{t}.{k}" in state_dict
                              for t in ("q", "k", "v")
                              for k in ("linear.bias", "bias"))

        for idx, tag in enumerate(("q", "k", "v")):
            base_w = _pop_first(state_dict,
                                [f"{prefix}{tag}.linear.weight",
                                 f"{prefix}{tag}.weight"])
            if base_w is None:
                base_w = _get_weight_chunk(module, idx, dim)

            A = _pop_first(state_dict, [f"{prefix}{tag}.lora.A"])
            B = _pop_first(state_dict, [f"{prefix}{tag}.lora.B"])
            if A is not None and B is not None:
                l_alpha  = _pop_first(state_dict, [f"{prefix}{tag}.lora.alpha"])
                l_rslora = _pop_first(state_dict, [f"{prefix}{tag}.lora.use_rslora"])
                l_alpha  = l_alpha.item()  if l_alpha  is not None else self.lora_alpha
                l_rslora = l_rslora.item() if l_rslora is not None else self.use_rslora
                merged_w = lora_merge(base_w, A.to(base_w.device), B.to(base_w.device), l_alpha, l_rslora)
            else:
                merged_w = base_w
            merged_weights.append(merged_w)

            if handle_bias:
                bias = _pop_first(state_dict,
                                  [f"{prefix}{tag}.linear.bias",
                                   f"{prefix}{tag}.bias"])
                if bias is None:
                    bias = module.bias.data[idx * dim:(idx + 1) * dim]
                merged_biases.append(bias)

        state_dict[prefix + "weight"] = torch.cat(merged_weights, dim=0)
        if merged_biases:
            state_dict[prefix + "bias"] = torch.cat(merged_biases, dim=0)

    def kv_linear_hook(self, module, state_dict, prefix, *_):
        merge_required = False
        for name in state_dict.keys():
            if name.startswith(prefix) and name not in (prefix + 'weight', prefix + 'bias'):
                merge_required = True
                break
        if not merge_required:
            return
        dim = module.in_features
        merged_weights, merged_biases = [], []

        handle_bias = module.bias is not None
        if handle_bias:
            handle_bias = any(f"{prefix}{t}.{k}" in state_dict
                              for t in ("k", "v")
                              for k in ("linear.bias", "bias"))

        for idx, tag in enumerate(("k", "v")):
            base_w = _pop_first(state_dict,
                                [f"{prefix}{tag}.linear.weight",
                                 f"{prefix}{tag}.weight"])
            if base_w is None:
                base_w = _get_weight_chunk(module, idx, dim)

            A = _pop_first(state_dict, [f"{prefix}{tag}.lora.A"])
            B = _pop_first(state_dict, [f"{prefix}{tag}.lora.B"])
            if A is not None and B is not None:
                l_alpha  = _pop_first(state_dict, [f"{prefix}{tag}.lora.alpha"])
                l_rslora = _pop_first(state_dict, [f"{prefix}{tag}.lora.use_rslora"])
                l_alpha  = l_alpha.item()  if l_alpha  is not None else self.lora_alpha
                l_rslora = l_rslora.item() if l_rslora is not None else self.use_rslora
                merged_w = lora_merge(base_w, A.to(base_w.device), B.to(base_w.device), l_alpha, l_rslora)
            else:
                merged_w = base_w
            merged_weights.append(merged_w)

            if handle_bias:
                bias = _pop_first(state_dict,
                                  [f"{prefix}{tag}.linear.bias",
                                   f"{prefix}{tag}.bias"])
                if bias is None:
                    bias = module.bias.data[idx * dim:(idx + 1) * dim]
                merged_biases.append(bias)

        state_dict[prefix + "weight"] = torch.cat(merged_weights, dim=0)
        if merged_biases:
            state_dict[prefix + "bias"] = torch.cat(merged_biases, dim=0)

    def linear_hook(self, module, state_dict, prefix, *_):
        if prefix + 'lora.A' not in state_dict:
            return

        handle_bias = module.bias is not None
        if handle_bias:
            handle_bias = any(f"{prefix}{k}" in state_dict
                              for k in ("linear.bias", "bias"))

        base_w = _pop_first(state_dict,
                            [f"{prefix}linear.weight",
                             f"{prefix}weight"])
        if base_w is None:
            base_w = module.weight.data

        A = _pop_first(state_dict, [f"{prefix}lora.A"])
        B = _pop_first(state_dict, [f"{prefix}lora.B"])
        l_alpha = _pop_first(state_dict, [f"{prefix}lora.alpha"])
        l_rslora = _pop_first(state_dict, [f"{prefix}lora.use_rslora"])
        l_alpha = l_alpha.item() if l_alpha is not None else self.lora_alpha
        l_rslora = l_rslora.item() if l_rslora is not None else self.use_rslora
        merged_w = lora_merge(base_w, A.to(base_w.device), B.to(base_w.device), l_alpha, l_rslora)
        state_dict[prefix + "weight"] = merged_w

        if handle_bias:
            bias = _pop_first(state_dict,
                              [f"{prefix}linear.bias",
                               f"{prefix}bias"])
            state_dict[prefix + "bias"] = bias


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _pop_first(state_dict, keys):
    """Return & delete the first matching key in `state_dict`, else None."""
    for k in keys:
        if k in state_dict:
            return state_dict.pop(k)
    return None


def _get_weight_chunk(module: nn.Linear, idx: int, dim: int):
    """
    Slice **only the weight** for q / k / v from a fused linear layer.

    Parameters
    ----------
    module : nn.Linear
        The fused QKV (or KV) linear layer.
    idx : int
        0 for q, 1 for k, 2 for v   (or 0 for k, 1 for v in KV‑fused).
    dim : int
        in_features of the layer.
    """
    return module.weight.data[idx * dim:(idx + 1) * dim]
# ----------------------------------------------------------------------
