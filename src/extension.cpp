#include <torch/extension.h>
#include <utils/common_defs.h>
#include <utils/cuda_utils.h>
#include <maps/log_map.h>
#include <maps/exp_map.h>
#include <ops/butterfly.h>
#include <manifolds/poincare.h>
#include <manifolds/geodesic.h>
#include <ops/mobius.h>
#include <layers/geodesic.h>

namespace utils = reality_stone::utils;
namespace maps = reality_stone::maps;
namespace ops = reality_stone::ops;
namespace manifolds = reality_stone::manifolds;
namespace layers = reality_stone::layers;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("log_map_cpu", &maps::log_map_cpu, "Log map origin (CPU)");
    m.def("exp_map_cpu", &maps::exp_map_cpu, "Exp map origin (CPU)");
    // 순전파
    m.def("poincare_forward_cpu", &manifolds::poincare_forward_cpu, "poincare_forward_cpu (CPU)");
    m.def("mobius_add_cpu", &ops::mobius_add_cpu, "Möbius addition (CPU)");
    m.def("mobius_scalar_cpu", &ops::mobius_scalar_cpu, "Möbius scalar multiplication (CPU)");
    m.def("geodesic_cpu", &manifolds::geodesic_cpu, "Geodesic interpolation (CPU)");
    // 레이어
    m.def("geodesic_forward_cpu", &layers::geodesic_forward_cpu, "Geodesic forward (CPU)");
    m.def("geodesic_backward_cpu", &layers::geodesic_backward_cpu, "Geodesic backward (CPU)");
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
    m.def("geodesic_cuda", &manifolds::geodesic_cuda, "Geodesic interpolation (CUDA)");
    // 레이어
    m.def("geodesic_forward_cuda", &layers::geodesic_forward_cuda, "Geodesic forward (CUDA)");
    m.def("geodesic_backward_cuda", &layers::geodesic_backward_cuda, "Geodesic backward (CUDA)");
#endif
}