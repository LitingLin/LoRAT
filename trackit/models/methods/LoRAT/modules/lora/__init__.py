import torch
import math
import torch.nn as nn
from timm.layers import trunc_normal_

from .merge import _lora_merge, _lora_unmerge

# https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
# https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
# https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/lora.py

r'''
             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮
    ┆                 ┆     \      B      / 
    ┆   pretrained    ┆      \    r*d    / 
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘
'''

class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, dropout: float,
                 rs_lora: bool = False, init_method: str = 'bert'):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, r))
        self.r = r
        self.alpha = alpha
        self.rs_lora = rs_lora
        if rs_lora:
            self.scaling = alpha / math.sqrt(r)
        else:
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
            trunc_normal_(self.A, std=.02)
            trunc_normal_(self.B, std=.02)
        else:
            raise ValueError(f'Unknown init method: {init_method}')

    def forward(self, x: torch.Tensor):
        return (self.dropout(x) @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, rs_lora: bool=False, init_method: str='bert', merge_on_eval: bool = False):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, r, alpha, dropout, rs_lora, init_method)
        self.merge_on_eval = merge_on_eval
        self.merged = False

    def forward(self, x: torch.Tensor):
        if self.merged:
            return self.linear(x)
        else:
            return self.linear(x) + self.lora(x)

    def train(self, mode: bool = True):
        if self.merge_on_eval:
            if not mode:
                self.merge()
            else:
                self.unmerge()

        return super().train(mode)

    def state_dict(self, **kwargs):
        merged = self.merged
        if merged:
            self.unmerge()
        state_dict = super().state_dict(**kwargs)
        if merged:
            self.merge()
        return state_dict

    def load_state_dict(self, **kwargs):
        merged = self.merged
        if merged:
            self.unmerge()
        ret = super().load_state_dict(**kwargs)
        if merged:
            self.merge()
        return ret

    def merge(self):
        if self.merged:
            return
        self.linear.weight.data.copy_(_lora_merge(self.linear.weight.data, self.lora.A, self.lora.B, self.lora.alpha, self.lora.rs_lora))
        self.merged = True

    def unmerge(self):
        if not self.merged:
            return
        self.linear.weight.data.copy_(_lora_unmerge(self.linear.weight.data, self.lora.A, self.lora.B, self.lora.alpha, self.lora.rs_lora))
        self.merged = False


class LinearWithLoRA_TimmQKV(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, rs_lora: bool = False,
                 init_method: str = 'bert',
                 target_q: bool = True, target_k: bool = True, target_v: bool = True):
        super().__init__()
        dim = linear.in_features
        bias = linear.bias is not None
        q = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        k = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        q.weight.data.copy_(linear.weight.data[:dim])
        k.weight.data.copy_(linear.weight.data[dim:2*dim])
        v.weight.data.copy_(linear.weight.data[2*dim:])
        q.weight.requires_grad = k.weight.requires_grad = v.weight.requires_grad = linear.weight.requires_grad
        if bias:
            q.bias.data.copy_(linear.bias.data[:dim])
            k.bias.data.copy_(linear.bias.data[dim:2*dim])
            v.bias.data.copy_(linear.bias.data[2*dim:])
            q.bias.requires_grad = k.bias.requires_grad = v.bias.requires_grad = linear.bias.requires_grad

        if target_q:
            self.q = LinearWithLoRA(q, r, alpha, dropout, rs_lora, init_method)
        else:
            self.q = q
        if target_k:
            self.k = LinearWithLoRA(k, r, alpha, dropout, rs_lora, init_method)
        else:
            self.k = k
        if target_v:
            self.v = LinearWithLoRA(v, r, alpha, dropout, rs_lora, init_method)
        else:
            self.v = v

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat((q, k, v), dim=-1)
