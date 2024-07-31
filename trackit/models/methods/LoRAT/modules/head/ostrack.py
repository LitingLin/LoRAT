from typing import Tuple

import torch.nn as nn
import torch
from trackit.core.operator.bbox.format import bbox_cxcywh_to_xyxy_torch
from trackit.core.utils.anchor_free_reference_points import get_anchor_free_reference_points


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class OSTrackConvHead(nn.Module):
    def __init__(self, inplanes: int, channel: int, map_size: Tuple[int, int]):
        super(OSTrackConvHead, self).__init__()

        # corner predict
        self.conv1_ctr = _conv(inplanes, channel)
        self.conv2_ctr = _conv(channel, channel // 2)
        self.conv3_ctr = _conv(channel // 2, channel // 4)
        self.conv4_ctr = _conv(channel // 4, channel // 8)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = _conv(inplanes, channel)
        self.conv2_offset = _conv(channel, channel // 2)
        self.conv3_offset = _conv(channel // 2, channel // 4)
        self.conv4_offset = _conv(channel // 4, channel // 8)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = _conv(inplanes, channel)
        self.conv2_size = _conv(channel, channel // 2)
        self.conv3_size = _conv(channel // 2, channel // 4)
        self.conv4_size = _conv(channel // 4, channel // 8)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.map_size = map_size
        self.register_buffer('bbox_offset', get_anchor_free_reference_points(map_size), persistent=False)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, H * W, C) input feature map
            Returns:
                Dict: {
                    'score_map' (torch.Tensor): (B, 1, H, W)
                    'bbox' (torch.Tensor): (B, H, W, 4)
                }
        '''

        B = x.shape[0]
        W, H = self.map_size
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        offset_map = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        size_map = self.conv5_size(x_size4)

        size_map = size_map.sigmoid()

        offset_map = offset_map.permute(0, 2, 3, 1).reshape(B, H, W, 2)
        size_map = size_map.permute(0, 2, 3, 1).reshape(B, H, W, 2)
        score_map = score_map.view(B, H, W)

        box_map = torch.cat((self.bbox_offset.view(1, H, W, 2).expand(B, H, W, 2) + offset_map, size_map), dim=-1)
        box_map = bbox_cxcywh_to_xyxy_torch(box_map)
        return {'score_map': score_map, 'boxes': box_map}
