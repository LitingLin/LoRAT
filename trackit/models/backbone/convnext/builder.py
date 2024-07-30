import torch.nn as nn
from . import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge, ConvNeXt


def forward(self, x):
    for i in range(self.num_depths):
        x = self.downsample_layers[i](x)
        x = self.stages[i](x)
    return x.flatten(2).permute(0, 2, 1).contiguous()


def adjust_model_(model: ConvNeXt, num_depths: int):
    del model.head
    del model.norm
    model.num_depths = num_depths
    model.forward = forward.__get__(model, ConvNeXt)
    assert num_depths <= 4
    if num_depths < 4:
        model.downsample_layers = nn.ModuleList([model.downsample_layers[i] for i in range(num_depths)])
        model.stages = nn.ModuleList([model.stages[i] for i in range(num_depths)])


def build_convnext_backbone(name: str, load_pretrained: bool,  num_depths: int = 4, **kwargs):
    if name == 'ConvNeXt-T':
        model = convnext_tiny(pretrained=load_pretrained, **kwargs)
    elif name == 'ConvNeXt-S':
        model = convnext_small(pretrained=load_pretrained, **kwargs)
    elif name == 'ConvNeXt-B':
        model = convnext_base(pretrained=load_pretrained, **kwargs)
    elif name == 'ConvNeXt-L':
        model = convnext_large(pretrained=load_pretrained, **kwargs)
    elif name == 'ConvNeXt-XL':
        model = convnext_xlarge(pretrained=load_pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown ConvNeXt model: {name}")
    adjust_model_(model, num_depths)
    return model
