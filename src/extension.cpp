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
#include <advanced/regularization/hyperbolic_regularization.h>
#include <advanced/geodesic_activation/geodesic_activation.h>

// 새로 추가된 고급 기능들
#include <advanced/chebyshev/chebyshev.h>
#include <advanced/laplace_beltrami/laplace_beltrami.h>
#include <advanced/hyperbolic_fft/hyperbolic_fft.h>

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

    // ===== 새로 추가된 체비셰프 기능들 =====
    m.def("chebyshev_approximation_cpu", &advanced::chebyshev_approximation_cpu, "Chebyshev approximation CPU",
        py::arg("x"), py::arg("order") = 10, py::arg("curvature") = 1.0f);
    m.def("chebyshev_distance_cpu", &advanced::chebyshev_distance_cpu, "Chebyshev distance CPU",
        py::arg("x"), py::arg("y"), py::arg("curvature") = 1.0f);
    m.def("chebyshev_nodes_cpu", &advanced::chebyshev_nodes_cpu, "Chebyshev nodes CPU",
        py::arg("n"), py::arg("device") = torch::kCPU);
    m.def("fast_chebyshev_transform_cpu", &advanced::fast_chebyshev_transform_cpu, "Fast Chebyshev transform CPU");
    m.def("inverse_chebyshev_transform_cpu", &advanced::inverse_chebyshev_transform_cpu, "Inverse Chebyshev transform CPU");
    m.def("chebyshev_derivative_cpu", &advanced::chebyshev_derivative_cpu, "Chebyshev derivative CPU");
    m.def("chebyshev_integral_cpu", &advanced::chebyshev_integral_cpu, "Chebyshev integral CPU",
        py::arg("coeffs"), py::arg("constant") = 0.0f);

    // ===== 새로 추가된 라플라스-벨트라미 기능들 =====
    m.def("hyperbolic_laplacian_cpu", &advanced::hyperbolic_laplacian_cpu, "Hyperbolic Laplacian CPU",
        py::arg("f"), py::arg("curvature") = 1.0f);
    m.def("heat_kernel_cpu", &advanced::heat_kernel_cpu, "Heat kernel CPU",
        py::arg("x"), py::arg("t"), py::arg("curvature") = 1.0f);
    m.def("laplace_beltrami_eigen_cpu", &advanced::laplace_beltrami_eigen_cpu, "Laplace-Beltrami eigendecomposition CPU",
        py::arg("manifold_points"), py::arg("curvature") = 1.0f);
    m.def("spectral_graph_conv_cpu", &advanced::spectral_graph_conv_cpu, "Spectral graph convolution CPU");
    m.def("solve_diffusion_equation_cpu", &advanced::solve_diffusion_equation_cpu, "Solve diffusion equation CPU",
        py::arg("initial_condition"), py::arg("time_step"), py::arg("num_steps"), py::arg("curvature") = 1.0f);
    m.def("geodesic_distance_matrix_cpu", &advanced::geodesic_distance_matrix_cpu, "Geodesic distance matrix CPU",
        py::arg("points"), py::arg("curvature") = 1.0f);
    m.def("spectral_normalize_cpu", &advanced::spectral_normalize_cpu, "Spectral normalization CPU");

    // ===== 새로 추가된 FFT 및 리만 기하학 기능들 =====
    m.def("hyperbolic_fft_cpu", &advanced::hyperbolic_fft_cpu, "Hyperbolic FFT CPU",
        py::arg("x"), py::arg("curvature") = 1.0f);
    m.def("spherical_harmonics_cpu", &advanced::spherical_harmonics_cpu, "Spherical harmonics CPU");
    m.def("fast_spherical_conv_cpu", &advanced::fast_spherical_conv_cpu, "Fast spherical convolution CPU",
        py::arg("f"), py::arg("g"), py::arg("curvature") = 1.0f);
    m.def("ricci_curvature_cpu", &advanced::ricci_curvature_cpu, "Ricci curvature CPU");
    m.def("parallel_transport_cpu", &advanced::parallel_transport_cpu, "Parallel transport CPU",
        py::arg("v"), py::arg("path"), py::arg("curvature") = 1.0f);
    m.def("geodesic_flow_cpu", &advanced::geodesic_flow_cpu, "Geodesic flow CPU",
        py::arg("x"), py::arg("v"), py::arg("t"), py::arg("curvature") = 1.0f);
    m.def("riemannian_gradient_cpu", &advanced::riemannian_gradient_cpu, "Riemannian gradient CPU",
        py::arg("euclidean_grad"), py::arg("x"), py::arg("curvature") = 1.0f);
    m.def("geodesic_sgd_step_cpu", &advanced::geodesic_sgd_step_cpu, "Geodesic SGD step CPU",
        py::arg("x"), py::arg("grad"), py::arg("lr"), py::arg("curvature") = 1.0f);
    m.def("hyperbolic_wavelet_decomposition_cpu", &advanced::hyperbolic_wavelet_decomposition_cpu, "Hyperbolic wavelet decomposition CPU",
        py::arg("signal"), py::arg("num_levels"), py::arg("curvature") = 1.0f);
    m.def("frequency_domain_filter_cpu", &advanced::frequency_domain_filter_cpu, "Frequency domain filter CPU",
        py::arg("signal"), py::arg("filter"), py::arg("curvature") = 1.0f);

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
    m.def("dynamic_curvature_pred", &advanced::dynamic_curvature_prediction_cuda, "Dynamic curvature prediction",
        py::arg("features"), py::arg("weight"), py::arg("bias"), py::arg("c_base") = 1.0f);
    m.def("dynamic_mobius_add", &advanced::dynamic_mobius_add_cuda, "Dynamic Möbius add",
        py::arg("u"), py::arg("v"), py::arg("curvatures"));
    m.def("dynamic_poincare_layer", &advanced::dynamic_poincare_layer_cuda, "Dynamic Poincare layer",
        py::arg("u"), py::arg("v"), py::arg("curvatures"), py::arg("t"));

    // ===== 고급 기능 - Regularization =====
    m.def("boundary_penalty", &advanced::boundary_penalty_cuda, "Boundary penalty",
        py::arg("x"), py::arg("curvature"), py::arg("epsilon") = 0.01f);
    m.def("curvature_penalty", &advanced::curvature_adaptive_penalty_cuda, "Curvature penalty",
        py::arg("x"), py::arg("curvature"));
    m.def("geodesic_penalty", &advanced::geodesic_variance_penalty_cuda, "Geodesic penalty",
        py::arg("weights"), py::arg("curvature"));
    m.def("combined_reg", &advanced::combined_regularization_cuda, "Combined regularization",
        py::arg("x"), py::arg("weights"), py::arg("curvature"), 
        py::arg("lambda_boundary") = 1.0f, py::arg("lambda_curvature") = 0.1f, py::arg("lambda_geodesic") = 0.01f);

    // ===== 고급 기능 - Geodesic Activation =====
    m.def("geodesic_activation", &advanced::geodesic_activation_cuda, "Geodesic activation",
        py::arg("input"), py::arg("anchors"), py::arg("t_values"), py::arg("weights"), py::arg("curvature") = 1.0f);
    m.def("einstein_midpoint", &advanced::einstein_midpoint_cuda, "Einstein midpoint",
        py::arg("points"), py::arg("weights"), py::arg("curvature") = 1.0f);
    m.def("multi_geodesic", &advanced::multi_geodesic_mixing_cuda, "Multi geodesic mixing",
        py::arg("input"), py::arg("anchors"), py::arg("t_values"), py::arg("weights"), py::arg("curvature") = 1.0f);

    // ===== 새로 추가된 CUDA 기능들 =====
    m.def("chebyshev_approximation_cuda", &advanced::chebyshev_approximation_cuda, "Chebyshev approximation CUDA",
        py::arg("x"), py::arg("order"), py::arg("curvature"));
    m.def("chebyshev_distance_cuda", &advanced::chebyshev_distance_cuda, "Chebyshev distance CUDA",
        py::arg("x"), py::arg("y"), py::arg("curvature"));
    m.def("fast_chebyshev_transform_cuda", &advanced::fast_chebyshev_transform_cuda, "Fast Chebyshev transform CUDA",
        py::arg("values"));
    m.def("inverse_chebyshev_transform_cuda", &advanced::inverse_chebyshev_transform_cuda, "Inverse Chebyshev transform CUDA",
        py::arg("coeffs"), py::arg("eval_points"));
    m.def("chebyshev_derivative_cuda", &advanced::chebyshev_derivative_cuda, "Chebyshev derivative CUDA",
        py::arg("coeffs"));
    m.def("chebyshev_integral_cuda", &advanced::chebyshev_integral_cuda, "Chebyshev integral CUDA",
        py::arg("coeffs"), py::arg("constant") = 0.0f);
    m.def("hyperbolic_laplacian_cuda", &advanced::hyperbolic_laplacian_cuda, "Hyperbolic Laplacian CUDA",
        py::arg("f"), py::arg("curvature"));
    m.def("heat_kernel_cuda", &advanced::heat_kernel_cuda, "Heat kernel CUDA",
        py::arg("x"), py::arg("t"), py::arg("curvature"));
    m.def("laplace_beltrami_eigen_cuda", &advanced::laplace_beltrami_eigen_cuda, "Laplace-Beltrami eigen CUDA",
        py::arg("manifold_points"), py::arg("curvature"));
    m.def("spectral_graph_conv_cuda", &advanced::spectral_graph_conv_cuda, "Spectral graph conv CUDA",
        py::arg("x"), py::arg("laplacian"), py::arg("weight"));
    m.def("solve_diffusion_equation_cuda", &advanced::solve_diffusion_equation_cuda, "Solve diffusion equation CUDA",
        py::arg("initial_condition"), py::arg("time_step"), py::arg("num_steps"), py::arg("curvature"));
    m.def("geodesic_distance_matrix_cuda", &advanced::geodesic_distance_matrix_cuda, "Geodesic distance matrix CUDA",
        py::arg("points"), py::arg("curvature"));
    m.def("spectral_normalize_cuda", &advanced::spectral_normalize_cuda, "Spectral normalize CUDA",
        py::arg("adjacency_matrix"));
    m.def("hyperbolic_fft_cuda", &advanced::hyperbolic_fft_cuda, "Hyperbolic FFT CUDA",
        py::arg("x"), py::arg("curvature"));
    m.def("spherical_harmonics_cuda", &advanced::spherical_harmonics_cuda, "Spherical harmonics CUDA",
        py::arg("theta_phi"), py::arg("l_max"));
    m.def("fast_spherical_conv_cuda", &advanced::fast_spherical_conv_cuda, "Fast spherical convolution CUDA",
        py::arg("f"), py::arg("g"), py::arg("curvature"));
    m.def("ricci_curvature_cuda", &advanced::ricci_curvature_cuda, "Ricci curvature CUDA",
        py::arg("metric_tensor"));
    m.def("parallel_transport_cuda", &advanced::parallel_transport_cuda, "Parallel transport CUDA",
        py::arg("v"), py::arg("path"), py::arg("curvature"));
    m.def("geodesic_flow_cuda", &advanced::geodesic_flow_cuda, "Geodesic flow CUDA",
        py::arg("x"), py::arg("v"), py::arg("t"), py::arg("curvature"));
    m.def("riemannian_gradient_cuda", &advanced::riemannian_gradient_cuda, "Riemannian gradient CUDA",
        py::arg("euclidean_grad"), py::arg("x"), py::arg("curvature"));
    m.def("geodesic_sgd_step_cuda", &advanced::geodesic_sgd_step_cuda, "Geodesic SGD step CUDA",
        py::arg("x"), py::arg("grad"), py::arg("lr"), py::arg("curvature"));
    m.def("hyperbolic_wavelet_decomposition_cuda", &advanced::hyperbolic_wavelet_decomposition_cuda, "Hyperbolic wavelet decomposition CUDA",
        py::arg("signal"), py::arg("num_levels"), py::arg("curvature"));
    m.def("frequency_domain_filter_cuda", &advanced::frequency_domain_filter_cuda, "Frequency domain filter CUDA",
        py::arg("signal"), py::arg("filter"), py::arg("curvature"));
#endif
}