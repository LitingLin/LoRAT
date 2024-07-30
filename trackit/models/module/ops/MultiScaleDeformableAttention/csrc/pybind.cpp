// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>

#include "pytorch_cpp_helper.hpp"

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight, const int im2col_step);

void ms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, const int im2col_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward,
        "forward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("im2col_step"));
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward,
        "backward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("grad_attn_weight"), py::arg("im2col_step"));
}
