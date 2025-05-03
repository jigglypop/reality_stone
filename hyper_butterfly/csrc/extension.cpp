#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/ops/butterfly.h>
#include <hyper_butterfly/manifolds/poincare.h>
#include <hyper_butterfly/manifolds/geodesic.h>
#include <hyper_butterfly/ops/mobius.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;
namespace ops = hyper_butterfly::ops;
namespace manifolds = hyper_butterfly::manifolds;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // CPU exports - 포인터 형식으로 함수 참조
    m.def("log_map_cpu", &maps::log_map_cpu, "Log map origin (CPU)");
    m.def("exp_map_cpu", &maps::exp_map_cpu, "Exp map origin (CPU)");
    // 순전파
    m.def("poincare_forward_cpu", &manifolds::poincare_forward_cpu, "poincare_forward_cpu (CPU)");
    m.def("mobius_add_cpu", &ops::mobius_add_cpu, "Möbius addition (CPU)");
    m.def("mobius_scalar_cpu", &ops::mobius_scalar_cpu, "Möbius scalar multiplication (CPU)");
    m.def("mobius_sub_cpu", &manifolds::mobius_sub_cpu, "Möbius subtraction (CPU)");
    m.def("geodesic_cpu", &manifolds::geodesic_cpu, "Geodesic interpolation (CPU)");
#ifdef WITH_CUDA
    // CUDA exports - 포인터 형식으로 함수 참조
    m.def("log_map_forward_cuda", &maps::log_map_forward_cuda, "Log map origin (CUDA)");
    m.def("log_map_backward_cuda", &maps::log_map_backward_cuda, "Log map origin (CUDA)");
    m.def("exp_map_forward_cuda", &maps::exp_map_forward_cuda, "Exp map origin (CUDA)");
    m.def("exp_map_backward_cuda", &maps::exp_map_backward_cuda, "Exp map origin (CUDA)");
    // CUDA forward
    m.def("poincare_forward_cuda", &manifolds::poincare_forward_cuda, "poincare_forward_cuda (CUDA)");
    m.def("poincare_backward_cuda", &manifolds::poincare_backward_cuda, "poincare_backward_cuda (CUDA)");
    m.def("mobius_add_cuda", &ops::mobius_add_cuda, "Möbius addition (CUDA)");
    m.def("mobius_scalar_cuda", &ops::mobius_scalar_cuda, "Möbius subtraction (CUDA)");
    m.def("mobius_sub_cuda", &manifolds::mobius_sub_cuda, "Möbius subtraction (CUDA)");
    m.def("geodesic_cuda", &manifolds::geodesic_cuda, "Geodesic interpolation (CUDA)");
#endif
}