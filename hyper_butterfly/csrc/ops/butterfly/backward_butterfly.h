// core/csrc/ops/butterfly/backward.h
#pragma once
#include <torch/extension.h>
#include "../../utils/common_defs.h"

namespace hyper_butterfly
{
    namespace ops
    {
        namespace butterfly
        {

#ifdef WITH_CUDA
            // CUDA 함수 선언
            torch::Tensor butterfly_backward_cuda(
                torch::Tensor grad_out,
                torch::Tensor input,
                torch::Tensor params,
                int layer_idx);
#endif

        } // namespace butterfly
    } // namespace ops
} // namespace hyper_butterfly