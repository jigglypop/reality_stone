#pragma once
#include <torch/extension.h>
#include "../base_poincare.h"
#include "../../utils/common_defs.h"

namespace hyper_butterfly
{
    namespace geometry
    {
        namespace poincare
        {

            // CPU 함수 선언
            torch::Tensor log_map_cpu(torch::Tensor x, float c);
            torch::Tensor exp_map_cpu(torch::Tensor v, float c);

#ifdef WITH_CUDA
            // CUDA 함수 선언
            torch::Tensor log_map_cuda(torch::Tensor x, float c);
            torch::Tensor exp_map_cuda(torch::Tensor v, float c);
#endif

            // 포앵카레 기하학 구현 클래스
            class PoincareGeometry : public RiemannianGeometry
            {
            public:
                torch::Tensor log_map(const torch::Tensor &x, float c) override
                {
                    if (x.is_cuda())
                    {
#ifdef WITH_CUDA
                        return log_map_cuda(x, c);
#else
                        TORCH_CHECK(false, "CUDA support not available");
#endif
                    }
                    else
                    {
                        return log_map_cpu(x, c);
                    }
                }

                torch::Tensor exp_map(const torch::Tensor &v, float c) override
                {
                    if (v.is_cuda())
                    {
#ifdef WITH_CUDA
                        return exp_map_cuda(v, c);
#else
                        TORCH_CHECK(false, "CUDA support not available");
#endif
                    }
                    else
                    {
                        return exp_map_cpu(v, c);
                    }
                }
            };

            // CPU 구현 (인라인)
            inline torch::Tensor log_map_cpu(torch::Tensor x, float c)
            {
                auto norm = torch::norm(x, 2, 1, true).clamp(EPS);
                float sqrt_c = std::sqrt(c);
                auto scn = (sqrt_c * norm).clamp(EPS, MAX_NORM_TANH);
                auto denom = scn + EPS;
                auto numer = torch::atanh(scn);
                auto factor = numer / denom;
                return factor * x;
            }

            inline torch::Tensor exp_map_cpu(torch::Tensor v, float c)
            {
                auto norm = torch::norm(v, 2, 1, true).clamp(EPS);
                float sqrt_c = std::sqrt(c);
                auto scn = (sqrt_c * norm).clamp(EPS, MAX_TANH_INPUT);
                auto denom = scn + 1e-3f;
                auto numer = torch::tanh(scn);
                auto factor = numer / denom;
                return factor * v;
            }

        } // namespace poincare
    } // namespace geometry
} // namespace hyper_butterfly