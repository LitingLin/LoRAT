from typing import Sequence, Tuple

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.layers.mlp import Mlp

from .lora import build_linear_layer_replica_with_lora_adapter


class Mlp_StaticRoutedLoRAExpert(nn.Module):
    def __init__(
            self,
            expert_num_replica: int,
            lora_rank: int,
            lora_alpha: float,
            lora_dropout: float,
            lora_enable_bits: Tuple[bool, ...],
            in_features: int,
            hidden_features: int | None=None,
            out_features: int | None=None,
            act_layer=nn.GELU,
            norm_layer: nn.Module | None=None,
            bias: bool | Tuple[bool, bool]=True,
            drop: float | Tuple[float, float]=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = build_linear_layer_replica_with_lora_adapter(nn.Linear(in_features, hidden_features, bias=bias[0]),
                                                                expert_num_replica,
                                                                lora_rank, lora_alpha, lora_dropout, lora_enable_bits)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = build_linear_layer_replica_with_lora_adapter(nn.Linear(hidden_features, out_features, bias=bias[1]),
                                                                expert_num_replica,
                                                                lora_rank, lora_alpha, lora_dropout, lora_enable_bits)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor, expert_indices: Sequence[int] | int, chunk_sizes: Sequence[int] | None = None):
        if chunk_sizes is not None:
            x_chunks = x.split(chunk_sizes, dim=1)
            x_out = []
            for i, x in enumerate(x_chunks):
                expert_index = expert_indices if isinstance(expert_indices, int) else expert_indices[i]
                x = self.fc1[expert_index](x)
                x = self.act(x)
                x = self.drop1(x)
                x = self.norm(x)
                x = self.fc2[expert_index](x)
                x = self.drop2(x)
                x_out.append(x)
            return torch.cat(x_out, dim=1)
        else:
            assert isinstance(expert_indices, int)
            expert_index = expert_indices
            x = self.fc1[expert_index](x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2[expert_index](x)
            x = self.drop2(x)
            return x

    @staticmethod
    def copy_from_std_block(other: Mlp,
                            expert_num_replica: int,
                            lora_rank: int,
                            lora_alpha: float,
                            lora_dropout: float,
                            lora_enable_bits: Tuple[bool, ...]):
        mlp = Mlp_StaticRoutedLoRAExpert.__new__(Mlp_StaticRoutedLoRAExpert)
        nn.Module.__init__(mlp)
        mlp.act = other.act
        assert isinstance(other.fc1, nn.Linear) and isinstance(other.fc2, nn.Linear)
        mlp.fc1 = build_linear_layer_replica_with_lora_adapter(other.fc1,
                                                               expert_num_replica,
                                                               lora_rank,
                                                               lora_alpha,
                                                               lora_dropout,
                                                               lora_enable_bits)
        mlp.fc2 = build_linear_layer_replica_with_lora_adapter(other.fc2,
                                                               expert_num_replica,
                                                               lora_rank,
                                                               lora_alpha,
                                                               lora_dropout,
                                                               lora_enable_bits)
        mlp.norm = other.norm
        mlp.drop1 = other.drop1
        mlp.drop2 = other.drop2
        return mlp
