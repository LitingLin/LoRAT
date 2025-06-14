# modify from fairseq/modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def fp32_group_norm_forward(self, input):
    output = F.group_norm(
        input.float(),
        self.num_groups,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.eps,
    )
    return output.type_as(input)

def fp32_batch_norm_forward(self, input):
    if self.running_mean.dtype != torch.float:
        if isinstance(self, nn.SyncBatchNorm):
            self.running_mean = self.running_mean.float()
            self.running_var = self.running_var.float()
            if self.affine:
                try:
                    self.weight = self.weight.float()
                    self.bias = self.bias.float()
                except:
                    self.float()
        else:
            self.float()

    output = super(self.__class__, self).forward(input.float())
    return output.type_as(input)

def fp32_layer_norm_forward(self, input):
    output = F.layer_norm(
        input.float(),
        self.normalized_shape,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.eps,
    )
    return output.type_as(input)


def fp32_instance_norm_forward(self, input):
    return F.instance_norm(
        input.float(),
        running_mean=self.running_mean,
        running_var=self.running_var,
        weight=self.weight.float() if self.weight is not None else None,
        bias=self.bias.float() if self.bias is not None else None,
        use_input_stats=self.training or not self.track_running_stats,
        momentum=self.momentum,
        eps=self.eps,
    )


def mixed_precision_auto_force_fp32_(module):
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.GroupNorm):
            assert submodule.forward.__func__ is nn.GroupNorm.forward
            submodule.forward = fp32_group_norm_forward.__get__(submodule, submodule.__class__)
            print(f'amp: GroupNorm {name} is forced to fp32')
        elif isinstance(submodule, nn.LayerNorm):
            assert submodule.forward.__func__ is nn.LayerNorm.forward
            submodule.forward = fp32_layer_norm_forward.__get__(submodule, submodule.__class__)
            print(f'amp: LayerNorm {name} is forced to fp32')
        elif isinstance(submodule, _BatchNorm):
            if isinstance(submodule, nn.SyncBatchNorm):
                assert submodule.forward.__func__ is nn.SyncBatchNorm.forward
            else:
                assert submodule.forward.__func__ is _BatchNorm.forward
            submodule.forward = fp32_batch_norm_forward.__get__(submodule, submodule.__class__)
            print(f'amp: BatchNorm {name} is forced to fp32')
        elif isinstance(submodule, _InstanceNorm):
            assert submodule.forward.__func__ is _InstanceNorm.forward
            submodule.forward = fp32_instance_norm_forward.__get__(submodule, submodule.__class__)
            print(f'amp: InstanceNorm {name} is forced to fp32')
    return module
