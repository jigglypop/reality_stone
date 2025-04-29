#include <torch/extension.h>
#include "hyper_butterfly.h"

// simple CPU matmul
torch::Tensor matmul_cpu(torch::Tensor &A, torch::Tensor &B)
{
    return torch::mm(A, B);
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", &matmul_cpu, "Matrix multiplication (CPU)");
    // CPU exports
    m.def("log_map_origin_cpu", &log_map_origin_cpu_export, "Log map origin (CPU)");
    m.def("exp_map_origin_cpu", &exp_map_origin_cpu_export, "Exp map origin (CPU)");
    m.def("hyper_butterfly_cpu", &hyper_butterfly_cpu_export, "HyperButterfly forward (CPU)");
#ifdef WITH_CUDA
    // CUDA exports
    m.def("log_map_origin_cuda", &log_map_origin_cuda, "Log map origin (CUDA)");
    m.def("exp_map_origin_cuda", &exp_map_origin_cuda, "Exp map origin (CUDA)");
    m.def("hyper_butterfly_cuda", &hyper_butterfly_cuda, "HyperButterfly forward (CUDA)");
    m.def("butterfly_layer_backward_cuda",&butterfly_layer_backward_cuda,"Butterfly layer backward (CUDA)");
#endif
}
