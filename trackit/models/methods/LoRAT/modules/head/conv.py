from typing import Tuple

import torch
import torch.nn as nn
from trackit.core.utils.anchor_free_reference_points import get_anchor_free_reference_points


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class Convs(nn.Module):
    def __init__(self, inplanes, channel, out_channel):
        super(Convs, self).__init__()

        self.conv1 = _conv(inplanes, channel,)
        self.conv2 = _conv(channel, channel // 2)
        self.conv3 = _conv(channel // 2, channel // 4)
        self.conv4 = _conv(channel // 4, channel // 8,)
        self.conv5 = nn.Conv2d(channel // 8, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class ConvAnchorFreeHead(nn.Module):
    def __init__(self, inplanes: int, channel: int, map_size: Tuple[int, int]):
        super(ConvAnchorFreeHead, self).__init__()
        self.cls = Convs(inplanes, channel, 1)
        self.reg = Convs(inplanes, channel, 4)
        self.map_size = map_size
        self.register_buffer('bbox_offset', get_anchor_free_reference_points(map_size), persistent=False)

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

        B = x.shape[0]

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        score_map = self.cls(x).to(torch.float32)
        box_map = self.reg(x).to(torch.float32)

        score_map = score_map.view(B, H, W)

        box_map = box_map.view(B, 4, H, W)
        box_map = box_map.permute(0, 2, 3, 1)
        box_map = box_map.sigmoid()
        lt, rb = box_map.chunk(2, -1)
        x1y1 = bbox_offset.view(1, H, W, 2) - lt
        x2y2 = bbox_offset.view(1, H, W, 2) + rb
        box_map = torch.cat((x1y1, x2y2), dim=-1)

        return {'score_map': score_map, 'boxes': box_map}
