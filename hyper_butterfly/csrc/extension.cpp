#include <torch/extension.h>
#include "common_defs.h"
#include "hyper_butterfly.h"
#include "maps.h"
#include "butterfly.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // CPU exports - 포인터 형식으로 함수 참조
    m.def("log_map_origin_cpu", &log_map_origin_cpu_export, "Log map origin (CPU)");
    m.def("exp_map_origin_cpu", &exp_map_origin_cpu_export, "Exp map origin (CPU)");
    m.def("hyper_butterfly_cpu", &hyper_butterfly_cpu_export, "Hyper-Butterfly forward (CPU)");

#ifdef WITH_CUDA
    // CUDA exports - 포인터 형식으로 함수 참조
    m.def("log_map_origin_cuda", &log_map_origin_cuda, "Log map origin (CUDA)");
    m.def("exp_map_origin_cuda", &exp_map_origin_cuda, "Exp map origin (CUDA)");
    m.def("hyper_butterfly_cuda", &hyper_butterfly_cuda, "Hyper-Butterfly forward (CUDA)");
    // CUDA backward
    m.def("hyper_butterfly_backward_cuda", &hyper_butterfly_backward_cuda, "Hyper-Butterfly backward (CUDA)");
#endif
}