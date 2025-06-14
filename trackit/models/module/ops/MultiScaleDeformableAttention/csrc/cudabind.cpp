#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor ms_deform_attn_cuda_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_cuda_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

Tensor ms_deform_attn_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_impl_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, CUDA,
                     ms_deform_attn_cuda_forward);
REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, CUDA,
                     ms_deform_attn_cuda_backward);
