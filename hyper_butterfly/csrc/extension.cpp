#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/ops/butterfly.h>
#include <hyper_butterfly/manifolds/poincare.h>

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

#ifdef WITH_CUDA
    // CUDA exports - 포인터 형식으로 함수 참조
    m.def("log_map_cuda", &maps::log_map_cuda, "Log map origin (CUDA)");
    m.def("exp_map_cuda", &maps::exp_map_cuda, "Exp map origin (CUDA)");
    // CUDA forward
    m.def("poincare_forward_cuda", &manifolds::poincare_forward_cuda, "poincare_forward_cuda (CUDA)");
    // m.def("hyper_butterfly_backward_cuda", &hyper_butterfly_backward_cuda, "Hyper-Butterfly backward (CUDA)");
#endif
}