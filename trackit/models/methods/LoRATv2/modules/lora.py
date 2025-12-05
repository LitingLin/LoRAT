import copy
import math
from functools import partial
from typing import Tuple, Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 r: int, alpha: float,
                 dropout: float, init_method: str = 'bert'):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, r))
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if init_method == 'lora':
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif init_method == 'gaussian':
            nn.init.normal_(self.A, std=1. / self.r)
            nn.init.zeros_(self.B)
        elif init_method == 'bert':
            nn.init.trunc_normal_(self.A, std=.02)
            nn.init.trunc_normal_(self.B, std=.02)
        else:
            raise ValueError(f'Unknown init method: {init_method}')

    def forward(self, x: torch.Tensor):
        return self.dropout(x) @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, init_method: str='bert'):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, r, alpha, dropout, init_method)

    def forward(self, x: torch.Tensor):
        return self.linear(x) + self.lora(x)


class LinearWithLoRA_QKVFused(nn.Module):
    def __init__(self, qkv_linear: nn.Linear, r: int, alpha: float, dropout: float, init_method: str='bert'):
        super().__init__()
        dim = qkv_linear.in_features
        bias = qkv_linear.bias is not None
        q = nn.Linear(dim, dim, bias, device=qkv_linear.weight.device, dtype=qkv_linear.weight.dtype)
        k = nn.Linear(dim, dim, bias, device=qkv_linear.weight.device, dtype=qkv_linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=qkv_linear.weight.device, dtype=qkv_linear.weight.dtype)
        q.weight.data.copy_(qkv_linear.weight.data[:dim])
        k.weight.data.copy_(qkv_linear.weight.data[dim:2 * dim])
        v.weight.data.copy_(qkv_linear.weight.data[2 * dim:])
        q.weight.requires_grad = k.weight.requires_grad = v.weight.requires_grad = qkv_linear.weight.requires_grad
        if bias:
            q.bias.data.copy_(qkv_linear.bias.data[:dim])
            k.bias.data.copy_(qkv_linear.bias.data[dim:2 * dim])
            v.bias.data.copy_(qkv_linear.bias.data[2 * dim:])
            q.bias.requires_grad = k.bias.requires_grad = v.bias.requires_grad = qkv_linear.bias.requires_grad

        self.q = LinearWithLoRA(q, r, alpha, dropout, init_method)
        self.k = LinearWithLoRA(k, r, alpha, dropout, init_method)
        self.v = LinearWithLoRA(v, r, alpha, dropout, init_method)

    def forward(self, x):
        return torch.cat((self.q(x), self.k(x), self.v(x)), dim=-1)



def _lora_delta(lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float], use_rslora: bool) -> torch.Tensor:
    r = lora_A.size(0)
    if alpha is not None:
        if use_rslora:
            scaling = alpha / math.sqrt(r)
        else:
            scaling = alpha / r
    else:
        scaling = 1.
    return (lora_B @ lora_A) * scaling


def lora_merge(weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float], use_rslora: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _lora_delta(lora_A.to(torch.float32), lora_B.to(torch.float32), alpha, use_rslora)

    return (weight.to(torch.float32) + delta).to(original_dtype)


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



def build_linear_layer_replica_with_lora_adapter(linear: nn.Linear,
                                                 num_replica: int,
                                                 lora_rank: int,
                                                 lora_alpha: float,
                                                 lora_dropout: float,
                                                 lora_enable_bits: Tuple[bool, ...]):
    modules = []
    hooker = Hooker(lora_alpha)
    for i in range(num_replica):
        replica = copy.deepcopy(linear)
        if lora_enable_bits[i]:
            replica = LinearWithLoRA(linear, lora_rank, lora_alpha, lora_dropout)
        else:
            replica._register_load_state_dict_pre_hook(partial(hooker.linear_hook, replica))
        modules.append(replica)
    return nn.ModuleList(modules)


def build_qkv_linear_layer_replica_with_lora_adapter(qkv_linear: nn.Linear,
                                                     num_replica: int,
                                                     lora_rank: int,
                                                     lora_alpha: float,
                                                     lora_dropout: float,
                                                     lora_enable_bits: Tuple[bool, ...]):
    modules = []
    hooker = Hooker(lora_alpha)
    for i in range(num_replica):
        replica = copy.deepcopy(qkv_linear)
        if lora_enable_bits[i]:
            replica = LinearWithLoRA_QKVFused(qkv_linear, lora_rank, lora_alpha, lora_dropout)
        else:
            replica._register_load_state_dict_pre_hook(partial(hooker.qkv_linear_hook, replica))
        modules.append(replica)
    return nn.ModuleList(modules)
