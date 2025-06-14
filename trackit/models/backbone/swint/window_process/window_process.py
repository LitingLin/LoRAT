# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch.cuda
import os
from trackit.miscellanies.torch.native_extension_jit import load_cuda_extension


if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')
swin_window_process = load_cuda_extension(name='swin_window_process', sources=[
    os.path.join(os.path.dirname(__file__), 'swin_window_process.cpp'),
    os.path.join(os.path.dirname(__file__), 'swin_window_process_kernel.cu')])


class WindowProcess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = swin_window_process.roll_and_window_partition_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size
        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = swin_window_process.roll_and_window_partition_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = swin_window_process.window_merge_and_roll_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size

        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        #grad_out = ctx.saved_tensors[0]
        #grad_out = torch.zeros((B, H, W, C), dtype=dtype).cuda()
        grad_out = swin_window_process.window_merge_and_roll_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None



if __name__ == '__main__':
    WindowProcess()