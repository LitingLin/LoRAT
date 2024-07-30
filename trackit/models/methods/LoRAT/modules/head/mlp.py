from typing import Tuple

import torch
from torch import nn
from timm.layers import trunc_normal_
from trackit.core.utils.anchor_free_reference_points import get_anchor_free_reference_points


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 num_layers=2,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layers = nn.ModuleList(
            [nn.Linear(hidden_features if i != 0 else in_features,
                       hidden_features if i != num_layers - 1 else out_features
                       ) for i in range(num_layers)]
        )

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        for i, linear in enumerate(self.layers):
            x = linear(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
            x = self.drop(x)
        return x


class MlpAnchorFreeHead(nn.Module):
    def __init__(self, dim: int, map_size: Tuple[int, int]):
        super(MlpAnchorFreeHead, self).__init__()
        self.cls_mlp = Mlp(dim, out_features=1, num_layers=3)
        self.reg_mlp = Mlp(dim, out_features=4, num_layers=3)
        self.map_size = map_size
        self.register_buffer('bbox_offset', get_anchor_free_reference_points(map_size), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, H * W, C) input feature map
            Returns:
                Dict: {
                    'cls_score' (torch.Tensor): (B, 1, H, W)
                    'bbox' (torch.Tensor): (B, H, W, 4)
                }
        '''
        W, H = self.map_size
        bbox_offset = self.bbox_offset

        score_map = self.cls_mlp(x).to(torch.float32)
        box_map = self.reg_mlp(x).to(torch.float32)

        B, L, C = score_map.shape
        score_map = score_map.view(B, H, W)

        box_map = box_map.view(B, H, W, 4)
        box_map = box_map.sigmoid()
        lt, rb = box_map.chunk(2, -1)
        x1y1 = bbox_offset.unsqueeze(0) - lt
        x2y2 = bbox_offset.unsqueeze(0) + rb
        box_map = torch.cat((x1y1, x2y2), dim=-1)

        return {'score_map': score_map, 'boxes': box_map}
