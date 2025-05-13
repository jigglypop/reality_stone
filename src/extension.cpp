#include <torch/extension.h>
#include <utils/common_defs.h>
#include <utils/cuda_utils.h>
#include <ops/mobius.h>
#include <layers/poincare_ball.h>

namespace utils = reality_stone::utils;
namespace ops = reality_stone::ops;
namespace layers = reality_stone::layers;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mobius_add_cpu", &ops::mobius_add_cpu, "Möbius addition (CPU)");
    m.def("mobius_scalar_cpu", &ops::mobius_scalar_cpu, "Möbius scalar multiplication (CPU)");
    // 레이어
    m.def("poincare_ball_forward_cpu", &layers::poincare_ball_forward_cpu, "poincare_ball forward (CPU)");
#ifdef WITH_CUDA
    m.def("mobius_add_cuda", &ops::mobius_add_cuda, "Möbius addition (CUDA)");
    m.def("mobius_scalar_cuda", &ops::mobius_scalar_cuda, "Möbius subtraction (CUDA)");
    // 레이어
    m.def("poincare_ball_forward_cuda", &layers::poincare_ball_forward_cuda, "poincare_ball forward (CUDA)");
#endif
}