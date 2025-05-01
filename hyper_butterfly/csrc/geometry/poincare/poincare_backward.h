#pragma once
#include <torch/extension.h>
#include "../base_poincare.h"
#include "../../utils/common_defs.h"

use namespace std;
use namespace torch;

namespace hyper_butterfly
{
    namespace geometry
    {
        namespace poincare
        {
            // 역전파 함수 선언 (CPU)
            vector<Tensor> log_map_backward_cpu(
                Tensor grad_output,
                Tensor x,
                float c);

            vector<Tensor> exp_map_backward_cpu(
                Tensor grad_output,
                Tensor v,
                float c);

#ifdef WITH_CUDA
            // 역전파 함수 선언 (CUDA)
            vector<Tensor> log_map_backward_cuda(
                Tensor grad_output,
                Tensor x,
                float c);

            vector<Tensor> exp_map_backward_cuda(
                Tensor grad_output,
                Tensor v,
                float c);
#endif

            // 포앵카레 역전파 클래스
            class PoincareBackward
            {
            public:
                // 로그 맵 역전파
                static vector<Tensor> log_map_backward(
                    Tensor grad_output,
                    Tensor x,
                    float c)
                {

                    if (grad_output.is_cuda())
                    {
#ifdef WITH_CUDA
                        return log_map_backward_cuda(grad_output, x, c);
#else
                        TORCH_CHECK(false, "CUDA support not available");
#endif
                    }
                    else
                    {
                        return log_map_backward_cpu(grad_output, x, c);
                    }
                }

                // 지수 맵 역전파
                static vector<Tensor> exp_map_backward(
                    Tensor grad_output,
                    Tensor v,
                    float c)
                {

                    if (grad_output.is_cuda())
                    {
#ifdef WITH_CUDA
                        return exp_map_backward_cuda(grad_output, v, c);
#else
                        TORCH_CHECK(false, "CUDA support not available");
#endif
                    }
                    else
                    {
                        return exp_map_backward_cpu(grad_output, v, c);
                    }
                }
            };

        }
    }
} // namespace hyper_butterfly::geometry::poincare