#include <torch/extension.h>
#include <utils/common_defs.h>
#include <utils/cuda_utils.h>
#include <ops/mobius.h>
#include <ops/lorentz.h>
#include <ops/klein.h>
#include <layers/poincare_ball.h>
#include <layers/lorentz.h>
#include <layers/klein.h>

namespace utils = reality_stone::utils;
namespace ops = reality_stone::ops;
namespace layers = reality_stone::layers;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mobius_add_cpu", &ops::mobius_add_cpu, "Möbius addition (CPU)");
    m.def("mobius_scalar_cpu", &ops::mobius_scalar_cpu, "Möbius scalar multiplication (CPU)");
    m.def("poincare_ball_forward_cpu", &layers::poincare_ball_forward_cpu, "Poincare ball forward (CPU)");
    m.def("poincare_ball_backward_cpu", &layers::poincare_ball_backward_cpu, "Poincaré backward (CPU)");
    m.def("lorentz_forward_cpu", &layers::lorentz_forward_cpu, "Lorentz forward (CPU)");
    m.def("lorentz_backward_cpu", &layers::lorentz_backward_cpu, "Lorentz backward (CPU)");
    m.def("klein_forward_cpu", &layers::klein_forward_cpu, "Klein forward (CPU)");
    m.def("klein_backward_cpu", &layers::klein_backward_cpu, "Klein backward (CPU)");
    
    m.def("lorentz_add_cpu", &ops::lorentz_add_cpu, "Lorentz addition (CPU)");
    m.def("lorentz_scalar_cpu", &ops::lorentz_scalar_cpu, "Lorentz scalar multiplication (CPU)");
    m.def("lorentz_inner_cpu", &ops::lorentz_inner_cpu, "Lorentz inner product (CPU)");
    m.def("lorentz_distance_cpu", &ops::lorentz_distance_cpu, "Lorentz distance (CPU)");
    m.def("poincare_to_lorentz_cpu", &ops::poincare_to_lorentz_cpu, "Poincare to Lorentz conversion (CPU)");
    m.def("lorentz_to_poincare_cpu", &ops::lorentz_to_poincare_cpu, "Lorentz to Poincare conversion (CPU)");
    m.def("klein_add_cpu", &ops::klein_add_cpu, "Klein addition (CPU)");
    m.def("klein_scalar_cpu", &ops::klein_scalar_cpu, "Klein scalar multiplication (CPU)");
    m.def("klein_distance_cpu", &ops::klein_distance_cpu, "Klein distance (CPU)");
    m.def("poincare_to_klein_cpu", &ops::poincare_to_klein_cpu, "Poincare to Klein conversion (CPU)");
    m.def("klein_to_poincare_cpu", &ops::klein_to_poincare_cpu, "Klein to Poincare conversion (CPU)");
    m.def("lorentz_to_klein_cpu", &ops::lorentz_to_klein_cpu, "Lorentz to Klein conversion (CPU)");
    m.def("klein_to_lorentz_cpu", &ops::klein_to_lorentz_cpu, "Klein to Lorentz conversion (CPU)");
#ifdef WITH_CUDA
    m.def("mobius_add_cuda", &ops::mobius_add_cuda, "Möbius addition (CUDA)");
    m.def("mobius_scalar_cuda", &ops::mobius_scalar_cuda, "Möbius scalar multiplication (CUDA)");
    m.def("poincare_ball_forward_cuda", &layers::poincare_ball_forward_cuda, "Poincare ball forward (CUDA)");
    m.def("poincare_ball_backward_cuda", &layers::poincare_ball_backward_cuda, "Poincare ball forward (CUDA)");
    m.def("lorentz_forward_cuda", &layers::lorentz_forward_cuda, "Lorentz forward (CUDA)");
    m.def("lorentz_backward_cuda", &layers::lorentz_backward_cuda, "Lorentz backward (CUDA)");
    m.def("klein_forward_cuda", &layers::klein_forward_cuda, "Klein forward (CUDA)");
    m.def("klein_backward_cuda", &layers::klein_backward_cuda, "Klein backward (CUDA)");

    m.def("lorentz_add_cuda", &ops::lorentz_add_cuda, "Lorentz addition (CUDA)");
    m.def("lorentz_scalar_cuda", &ops::lorentz_scalar_cuda, "Lorentz scalar multiplication (CUDA)");
    m.def("lorentz_inner_cuda", &ops::lorentz_inner_cuda, "Lorentz inner product (CUDA)");
    m.def("lorentz_distance_cuda", &ops::lorentz_distance_cuda, "Lorentz distance (CUDA)");
    m.def("poincare_to_lorentz_cuda", &ops::poincare_to_lorentz_cuda, "Poincare to Lorentz conversion (CUDA)");
    m.def("lorentz_to_poincare_cuda", &ops::lorentz_to_poincare_cuda, "Lorentz to Poincare conversion (CUDA)");
    m.def("lorentz_forward_cuda", &layers::lorentz_forward_cuda, "Lorentz forward (CUDA)");
    m.def("klein_add_cuda", &ops::klein_add_cuda, "Klein addition (CUDA)");
    m.def("klein_scalar_cuda", &ops::klein_scalar_cuda, "Klein scalar multiplication (CUDA)");
    m.def("klein_distance_cuda", &ops::klein_distance_cuda, "Klein distance (CUDA)");
    m.def("poincare_to_klein_cuda", &ops::poincare_to_klein_cuda, "Poincare to Klein conversion (CUDA)");
    m.def("klein_to_poincare_cuda", &ops::klein_to_poincare_cuda, "Klein to Poincare conversion (CUDA)");
    m.def("lorentz_to_klein_cuda", &ops::lorentz_to_klein_cuda, "Lorentz to Klein conversion (CUDA)");
    m.def("klein_to_lorentz_cuda", &ops::klein_to_lorentz_cuda, "Klein to Lorentz conversion (CUDA)");
#endif
}