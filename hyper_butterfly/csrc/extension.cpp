// #include <torch/extension.h>
// #include "utils/common_defs.h"
// #include "hyper_butterfly.h"
// #include "maps.h"
// #include "butterfly.h"
//
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     // CPU exports - 포인터 형식으로 함수 참조
//     m.def("log_map_origin_cpu", &log_map_origin_cpu_export, "Log map origin (CPU)");
//     m.def("exp_map_origin_cpu", &exp_map_origin_cpu_export, "Exp map origin (CPU)");
//     m.def("hyper_butterfly_cpu", &hyper_butterfly_cpu_export, "Hyper-Butterfly forward (CPU)");
//
// #ifdef WITH_CUDA
//     // CUDA exports - 포인터 형식으로 함수 참조
//     m.def("log_map_origin_cuda", &log_map_origin_cuda, "Log map origin (CUDA)");
//     m.def("exp_map_origin_cuda", &exp_map_origin_cuda, "Exp map origin (CUDA)");
//     m.def("hyper_butterfly_cuda", &hyper_butterfly_cuda, "Hyper-Butterfly forward (CUDA)");
//     // CUDA backward
//     m.def("hyper_butterfly_backward_cuda", &hyper_butterfly_backward_cuda, "Hyper-Butterfly backward (CUDA)");
// #endif
// }

// core/csrc/extension.cpp
#include <torch/extension.h>
#include "utils/common_defs.h"
#include "geometry/base_poincare.h"
#include "geometry/poincare/forward_poincare.h"
#include "geometry/poincare/poincare_backward.h"
#include "ops/butterfly/forward_butterfly.h"
#include "ops/butterfly/backward_butterfly.h"

use namespace std;
use namespace torch;
// CPU 함수 선언
Tensor log_map_cpu_export(Tensor x, float c);
Tensor exp_map_cpu_export(Tensor v, float c);
Tensor butterfly_forward_cpu_export(
    Tensor input,
    Tensor params,
    int layer_idx,
    int batch_size,
    int dim);
vector<Tensor> log_map_backward_cpu_export(
    Tensor grad_output,
    Tensor x,
    float c);
vector<Tensor> exp_map_backward_cpu_export(
    Tensor grad_output,
    Tensor v,
    float c);

#ifdef WITH_CUDA
// CUDA 함수 선언
Tensor log_map_cuda_export(Tensor x, float c);
Tensor exp_map_cuda_export(Tensor v, float c);
Tensor butterfly_forward_cuda_export(
    Tensor input,
    Tensor params,
    int layer_idx,
    int batch_size,
    int dim);
vector<Tensor> butterfly_backward_cuda_export(
    Tensor grad_out,
    Tensor input,
    Tensor params,
    int layer_idx);
vector<Tensor> log_map_backward_cuda_export(
    Tensor grad_output,
    Tensor x,
    float c);
vector<Tensor> exp_map_backward_cuda_export(
    Tensor grad_output,
    Tensor v,
    float c);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // CPU exports
    m.def("log_map_cpu", &log_map_cpu_export, "Log map (CPU)");
    m.def("exp_map_cpu", &exp_map_cpu_export, "Exp map (CPU)");
    // 포앵카레 기하학 역전파 함수 (CPU)
    m.def("log_map_backward_cpu", &log_map_backward_cpu_export, "Poincare log map backward (CPU)");
    m.def("exp_map_backward_cpu", &exp_map_backward_cpu_export, "Poincare exp map backward (CPU)");
    m.def("butterfly_forward_cpu", &butterfly_forward_cpu_export, "Butterfly forward (CPU)");
#ifdef WITH_CUDA
    // CUDA exports
    m.def("log_map_cuda", &log_map_cuda_export, "Log map (CUDA)");
    m.def("exp_map_cuda", &exp_map_cuda_export, "Exp map (CUDA)");
    m.def("butterfly_forward_cuda", &butterfly_forward_cuda_export, "Butterfly forward (CUDA)");
    m.def("butterfly_backward_cuda", &butterfly_backward_cuda_export, "Butterfly backward (CUDA)");
    // 포앵카레 기하학 역전파 함수 (CUDA)
    m.def("log_map_backward_cuda", &log_map_backward_cuda_export, "Poincare log map backward (CUDA)");
    m.def("exp_map_backward_cuda", &exp_map_backward_cuda_export, "Poincare exp map backward (CUDA)");

#endif
}