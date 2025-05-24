#include <torch/extension.h>
#include <utils/common_defs.h>
#include <utils/cuda_utils.h>
#include <ops/mobius.h>
#include <ops/lorentz.h>
#include <ops/klein.h>
#include <layers/poincare_ball.h>
#include <layers/lorentz.h>
#include <layers/klein.h>

// Advanced features headers
#include <advanced/fused_ops/fused_ops.h>
#include <advanced/dynamic_curvature/dynamic_curvature.h>
#include <advanced/regularization/hyperbolic_reqularization.h>
#include <advanced/geodesic_activation/geodesic_activation.h>

namespace utils = reality_stone::utils;
namespace ops = reality_stone::ops;
namespace layers = reality_stone::layers;
namespace advanced = reality_stone::advanced;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ===== CPU 기본 연산 =====
    m.def("mobius_add_cpu", &ops::mobius_add_cpu, "Möbius add CPU");
    m.def("mobius_scalar_cpu", &ops::mobius_scalar_cpu, "Möbius scalar CPU");
    
    // ===== CPU 레이어 =====
    m.def("poincare_ball_forward_cpu", &layers::poincare_ball_forward_cpu, "Poincare forward CPU");
    m.def("poincare_ball_backward_cpu", &layers::poincare_ball_backward_cpu, "Poincare backward CPU");
    m.def("lorentz_forward_cpu", &layers::lorentz_forward_cpu, "Lorentz forward CPU");
    m.def("lorentz_backward_cpu", &layers::lorentz_backward_cpu, "Lorentz backward CPU");
    m.def("klein_forward_cpu", &layers::klein_forward_cpu, "Klein forward CPU");
    m.def("klein_backward_cpu", &layers::klein_backward_cpu, "Klein backward CPU");
    
    // ===== CPU 변환 =====
    m.def("lorentz_add_cpu", &ops::lorentz_add_cpu, "Lorentz add CPU");
    m.def("lorentz_scalar_cpu", &ops::lorentz_scalar_cpu, "Lorentz scalar CPU");
    m.def("lorentz_inner_cpu", &ops::lorentz_inner_cpu, "Lorentz inner CPU");
    m.def("lorentz_distance_cpu", &ops::lorentz_distance_cpu, "Lorentz distance CPU");
    m.def("poincare_to_lorentz_cpu", &ops::poincare_to_lorentz_cpu, "P2L CPU");
    m.def("lorentz_to_poincare_cpu", &ops::lorentz_to_poincare_cpu, "L2P CPU");
    m.def("klein_add_cpu", &ops::klein_add_cpu, "Klein add CPU");
    m.def("klein_scalar_cpu", &ops::klein_scalar_cpu, "Klein scalar CPU");
    m.def("klein_distance_cpu", &ops::klein_distance_cpu, "Klein distance CPU");
    m.def("poincare_to_klein_cpu", &ops::poincare_to_klein_cpu, "P2K CPU");
    m.def("klein_to_poincare_cpu", &ops::klein_to_poincare_cpu, "K2P CPU");
    m.def("lorentz_to_klein_cpu", &ops::lorentz_to_klein_cpu, "L2K CPU");
    m.def("klein_to_lorentz_cpu", &ops::klein_to_lorentz_cpu, "K2L CPU");

    // ===== 고급 기능 - Fused Operations (CPU/CUDA 공용) =====
    m.def("fused_linear", &advanced::hyperbolic_linear_fused, "Fused hyperbolic linear");
    m.def("fused_mobius_chain", &advanced::mobius_chain_fused, "Fused Möbius chain");
    m.def("fused_transform_reg", &advanced::transform_regularize_fused, "Fused transform+reg");

#ifdef WITH_CUDA
    // ===== CUDA 기본 연산 =====
    m.def("mobius_add_cuda", &ops::mobius_add_cuda, "Möbius add CUDA");
    m.def("mobius_scalar_cuda", &ops::mobius_scalar_cuda, "Möbius scalar CUDA");
    
    // ===== CUDA 레이어 =====
    m.def("poincare_ball_forward_cuda", &layers::poincare_ball_forward_cuda, "Poincare forward CUDA");
    m.def("poincare_ball_backward_cuda", &layers::poincare_ball_backward_cuda, "Poincare backward CUDA");
    m.def("lorentz_forward_cuda", &layers::lorentz_forward_cuda, "Lorentz forward CUDA");
    m.def("lorentz_backward_cuda", &layers::lorentz_backward_cuda, "Lorentz backward CUDA");
    m.def("klein_forward_cuda", &layers::klein_forward_cuda, "Klein forward CUDA");
    m.def("klein_backward_cuda", &layers::klein_backward_cuda, "Klein backward CUDA");

    // ===== CUDA 변환 =====
    m.def("lorentz_add_cuda", &ops::lorentz_add_cuda, "Lorentz add CUDA");
    m.def("lorentz_scalar_cuda", &ops::lorentz_scalar_cuda, "Lorentz scalar CUDA");
    m.def("lorentz_inner_cuda", &ops::lorentz_inner_cuda, "Lorentz inner CUDA");
    m.def("lorentz_distance_cuda", &ops::lorentz_distance_cuda, "Lorentz distance CUDA");
    m.def("poincare_to_lorentz_cuda", &ops::poincare_to_lorentz_cuda, "P2L CUDA");
    m.def("lorentz_to_poincare_cuda", &ops::lorentz_to_poincare_cuda, "L2P CUDA");
    m.def("klein_add_cuda", &ops::klein_add_cuda, "Klein add CUDA");
    m.def("klein_scalar_cuda", &ops::klein_scalar_cuda, "Klein scalar CUDA");
    m.def("klein_distance_cuda", &ops::klein_distance_cuda, "Klein distance CUDA");
    m.def("poincare_to_klein_cuda", &ops::poincare_to_klein_cuda, "P2K CUDA");
    m.def("klein_to_poincare_cuda", &ops::klein_to_poincare_cuda, "K2P CUDA");
    m.def("lorentz_to_klein_cuda", &ops::lorentz_to_klein_cuda, "L2K CUDA");
    m.def("klein_to_lorentz_cuda", &ops::klein_to_lorentz_cuda, "K2L CUDA");

    // ===== 고급 기능 - Dynamic Curvature =====
    m.def("dynamic_curvature_pred", &advanced::dynamic_curvature_prediction_cuda, "Dynamic curvature prediction");
    m.def("dynamic_mobius_add", &advanced::dynamic_mobius_add_cuda, "Dynamic Möbius add");
    m.def("dynamic_poincare_layer", &advanced::dynamic_poincare_layer_cuda, "Dynamic Poincare layer");

    // ===== 고급 기능 - Regularization =====
    m.def("boundary_penalty", &advanced::boundary_penalty_cuda, "Boundary penalty");
    m.def("curvature_penalty", &advanced::curvature_adaptive_penalty_cuda, "Curvature penalty");
    m.def("geodesic_penalty", &advanced::geodesic_variance_penalty_cuda, "Geodesic penalty");
    m.def("combined_reg", &advanced::combined_regularization_cuda, "Combined regularization");

    // ===== 고급 기능 - Geodesic Activation =====
    m.def("geodesic_activation", &advanced::geodesic_activation_cuda, "Geodesic activation");
    m.def("einstein_midpoint", &advanced::einstein_midpoint_cuda, "Einstein midpoint");
    m.def("multi_geodesic", &advanced::multi_geodesic_mixing_cuda, "Multi geodesic mixing");
#endif
}