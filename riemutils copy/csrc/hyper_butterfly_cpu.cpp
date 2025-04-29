#include <torch/extension.h>
#include <cmath>
#include <vector>
#include "hyper_butterfly.h"

// CPU에서의 로그 맵 구현 (클램핑 적용)
torch::Tensor log_map_origin_cpu_export(torch::Tensor x, float c)
{
    // 최소 norm과 분모 안전화를 위한 EPS
    static constexpr float EPS = 1e-6f;

    // 1) 각 배치별 L2 norm 계산 및 최소값 클램핑
    auto norm = torch::norm(x, 2, 1, true).clamp(EPS);
    float sqrt_c = std::sqrt(c);

    // 2) atanh 인자 클램핑: [EPS, 1 - 1e-6]
    auto scn = (sqrt_c * norm).clamp(EPS, 1.0f - 1e-6f);

    // 3) 분모 안전화
    auto denom = scn + EPS;

    // 4) atanh 계산
    auto numer = torch::atanh(scn);
    auto factor = numer / denom;

    // 5) 최종 스케일 적용
    return factor * x;
}

// CPU에서의 지수 맵 구현 (클램핑 적용)
torch::Tensor exp_map_origin_cpu_export(torch::Tensor v, float c)
{
    static constexpr float EPS = 1e-6f;

    // 1) 각 배치별 L2 norm 계산 및 최소값 클램핑
    auto norm = torch::norm(v, 2, 1, true).clamp(EPS);
    float sqrt_c = std::sqrt(c);

    // 2) tanh 인자 클램핑: [EPS, 10]
    auto scn = (sqrt_c * norm).clamp(EPS, 10.0f);

    // 3) 분모에 여유를 더 준다
    auto denom = scn + 1e-3f;

    // 4) tanh 계산
    auto numer = torch::tanh(scn);
    auto factor = numer / denom;

    // 5) 최종 스케일 적용
    return factor * v;
}

// 단일 Butterfly 레이어 적용
torch::Tensor butterfly_layer_cpu(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim)
{
    auto output = torch::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_layer_cpu", ([&]
                                                                            {
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
                scalar_t a  = p_ptr[pidx];
                scalar_t bb = p_ptr[pidx+1];
                int base = b * dim + blk * 2 * block_size;
                scalar_t x1 = x_ptr[base + off];
                scalar_t x2 = x_ptr[base + off + block_size];
                y_ptr[b*dim + f] = hi ? (-bb * x1 + a * x2) : (a * x1 + bb * x2);
            }
        } }));
    return output;
}

// 전체 Hyper-Butterfly CPU 구현
std::vector<torch::Tensor> hyper_butterfly_cpu_export(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor /*args*/,
    float c,
    int L)
{
    // 1) 로그 맵 적용
    auto u = log_map_origin_cpu_export(x, c);

    // 2) Butterfly 변환 적용
    auto v = u;
    int batch_size = x.size(0);
    int dim = x.size(1);
    for (int l = 0; l < L; ++l)
    {
        int layer_idx = l % int(std::log2(dim));
        v = butterfly_layer_cpu(v, params, layer_idx, batch_size, dim);
    }

    // 3) 지수 맵 적용
    auto y = exp_map_origin_cpu_export(v, c);

    return {y, u, v};
}
