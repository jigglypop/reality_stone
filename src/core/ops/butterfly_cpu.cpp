#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <hyper_butterfly/utils/common_defs.h>
#include <hyper_butterfly/utils/cuda_utils.h>
#include <hyper_butterfly/maps/exp_map.h>
#include <hyper_butterfly/maps/log_map.h>
#include <hyper_butterfly/ops/butterfly.h>

namespace utils = hyper_butterfly::utils;
namespace maps = hyper_butterfly::maps;

namespace hyper_butterfly::ops {
    torch::Tensor butterfly_forward_cpu(
        torch::Tensor input,
        torch::Tensor params,
        int layer_idx,
        int batch_size,
        int dim) {
        auto output = torch::empty_like(input);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_forward_cpu", ([&] {
            const scalar_t* x_ptr = input.data_ptr<scalar_t>();
            scalar_t* y_ptr = output.data_ptr<scalar_t>();
            const scalar_t* p_ptr = params.data_ptr<scalar_t>();
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
            } }));
            return output;
    }
}