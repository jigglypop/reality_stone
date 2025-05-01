#pragma once
#include <torch/extension.h>
#include <hyper_butterfly/utils/common_defs.h>

namespace hyper_butterfly {

// CPU 함수 선언
torch::Tensor butterfly_layer_cpu(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim);

#ifdef WITH_CUDA
// CUDA 함수 선언
torch::Tensor butterfly_layer_cuda(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim);

std::vector<torch::Tensor> butterfly_layer_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx);
#endif

// CPU 구현 (인라인)
inline torch::Tensor butterfly_layer_cpu(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim) {

    auto output = torch::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_layer_cpu", ([&] {
        const auto* x_ptr = input.data_ptr<scalar_t>();
        auto* y_ptr = output.data_ptr<scalar_t>();
        const auto* p_ptr = params.data_ptr<scalar_t>();
        int block_size = 1 << layer_idx;
        int num_blocks = dim / (2 * block_size);
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < dim; f++) {
                int blk = (f / (2 * block_size)) % num_blocks;
                int loc = f % (2 * block_size);
                bool hi = loc >= block_size;
                int off = loc % block_size;
                int pidx = blk * 2;
                scalar_t a = p_ptr[pidx];
                scalar_t bb = p_ptr[pidx + 1];
                int base = b * dim + blk * 2 * block_size;
                scalar_t x1 = x_ptr[base + off];
                scalar_t x2 = x_ptr[base + off + block_size];
                y_ptr[b * dim + f] = hi ? (-bb * x1 + a * x2) : (a * x1 + bb * x2);
            }
        }
        }));
    return output;
}

} // namespace hyper_butterfly 