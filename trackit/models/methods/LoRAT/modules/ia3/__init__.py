from typing import Sequence

import torch
from torch import nn


class IA3TimmQKVLinearLayer(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.qkv = linear
        assert linear.in_features * 3 == linear.out_features
        self.k_ia3_weights = nn.Parameter(torch.ones(linear.in_features))
        self.v_ia3_weights = nn.Parameter(torch.ones(linear.in_features))

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        q, k, v = qkv.chunk(3, dim=-1)
        k = k * self.k_ia3_weights
        v = v * self.v_ia3_weights
        qkv = torch.cat([q, k, v], dim=-1)
        return qkv


class IA3FeedForwardLinearLayer(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.ia3_weight = nn.Parameter(torch.ones(linear.in_features))

    def forward(self, x: torch.Tensor):
        x = x * self.ia3_weight
        x = self.linear(x)
        return x


def apply_ia3(model: nn.Module, linear_layer_names: Sequence[str], ):
    for linear_layer_name in linear_layer_names:
        tokens = linear_layer_name.strip().split('.')
        layer = model
        for t in tokens[:-1]:
            if not t.isnumeric():
                layer = getattr(layer, t)
            else:
                layer = layer[int(t)]

        if 'qkv' == tokens[-1]:
            layer.qkv = IA3TimmQKVLinearLayer(layer.qkv)
        elif 'fc2' == tokens[-1] and 'mlp' in linear_layer_name:
            setattr(layer, tokens[-1], IA3FeedForwardLinearLayer(getattr(layer, tokens[-1])))
