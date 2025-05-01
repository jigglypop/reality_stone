#include "poincare_backward.h"
#include <cmath>

use namespace std;
use namespace torch;

namespace hyper_butterfly
{
    namespace geometry
    {
        namespace poincare
        {
            // 로그 맵 역전파 CPU 구현
            vector<Tensor> log_map_backward_cpu(
                Tensor grad_output,
                Tensor x,
                float c)
            {
                auto norm = norm(x, 2, 1, true).clamp(EPS);
                float sqrt_c = sqrt(c);
                auto scn = (sqrt_c * norm).clamp(EPS, MAX_NORM_TANH);
                // atanh(x) = 0.5 * log((1+x)/(1-x))
                // d/dx atanh(x) = 1/(1-x²)
                auto scn_sq = scn * scn;
                auto atanh_deriv = 1.0 / (1.0 - scn_sq);
                // factor = atanh(scn) / scn
                auto atanh_val = atanh(scn);
                auto factor = atanh_val / scn;
                // Compute gradients
                // 1. Direct term: factor * grad_output
                auto direct_term = factor * grad_output;
                // 2. Indirect term via norm: d(factor)/d(norm) * x * grad_output
                auto dnorm_dx = x / norm;
                auto dscn_dnorm = sqrt_c * ones_like(norm);
                auto dfactor_dscn = (atanh_deriv - atanh_val / scn) / scn;
                auto dfactor_dnorm = dfactor_dscn * dscn_dnorm;
                // Multiplier for each dimension based on gradient projection
                auto proj = sum(x * grad_output, 1, true) / (norm * norm);
                auto indirect_term = dfactor_dnorm * proj * x;
                // Total gradient
                auto dx = direct_term + indirect_term;
                return {dx};
            }

            // 지수 맵 역전파 CPU 구현
            vector<Tensor> exp_map_backward_cpu(
                Tensor grad_output,
                Tensor v,
                float c)
            {
                auto norm = norm(v, 2, 1, true).clamp(EPS);
                float sqrt_c = sqrt(c);
                auto scn = (sqrt_c * norm).clamp(EPS, MAX_TANH_INPUT);
                // tanh(x) derivative: 1 - tanh²(x)
                auto tanh_val = tanh(scn);
                auto tanh_deriv = 1.0 - tanh_val * tanh_val;
                // factor = tanh(scn) / scn
                auto factor = tanh_val / scn;
                // Compute gradients
                // 1. Direct term: factor * grad_output
                auto direct_term = factor * grad_output;
                // 2. Indirect term via norm
                auto dnorm_dv = v / norm;
                auto dscn_dnorm = sqrt_c * ones_like(norm);
                auto dfactor_dscn = (tanh_deriv - tanh_val / scn) / scn;
                auto dfactor_dnorm = dfactor_dscn * dscn_dnorm;
                // Multiplier for each dimension based on gradient projection
                auto proj = sum(v * grad_output, 1, true) / (norm * norm);
                auto indirect_term = dfactor_dnorm * proj * v;
                // Total gradient
                auto dv = direct_term + indirect_term;
                return {dv};
            }
        }
    }
} // namespace core::geometry::poincare

// Python 바인딩 함수
vector<Tensor> log_map_backward_cpu_export(
    Tensor grad_output,
    Tensor x,
    float c)
{
    return hyper_butterfly::geometry::poincare::log_map_backward_cpu(grad_output, x, c);
}

vector<Tensor> exp_map_backward_cpu_export(
    Tensor grad_output,
    Tensor v,
    float c)
{
    return hyper_butterfly::geometry::poincare::exp_map_backward_cpu(grad_output, v, c);
}