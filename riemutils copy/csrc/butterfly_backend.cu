// riemutils/csrc/butterfly_backward.cu

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA_CONTIGUOUS(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be CUDA tensor")

// Butterfly 레이어 순전파에서 한 블록당 두 파라미터(a,b)로
// Givens 회전을 수행했을 때의 역전파 커널.
// grad_out: [B, D]  순전파 출력에 대한 upstream gradient
// input:    [B, D]  순전파 입력(u)
// params:   [P]     파라미터 벡터 (P = sum_l 2*(D/2^l))
// layer_idx: 실행할 레이어 인덱스 l
template <typename scalar_t>
__global__ void butterfly_layer_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ params,
    scalar_t*       __restrict__ grad_input,
    scalar_t*       __restrict__ grad_params,
    int B, int D, int layer_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N   = B * D;
    if (idx >= N) return;

    // 배치, feature 인덱스 분리
    int b = idx / D;
    int f = idx % D;

    // 이 레이어 블록 크기와 파라미터 오프셋 계산
    int bs = 1 << layer_idx;
    int nb = D / (2 * bs);
    int p_block_offset = 0;
    for (int l = 0; l < layer_idx; ++l) {
        int bsl = 1 << l;
        p_block_offset += 2 * (D / (2 * bsl));
    }

    // 해당 feature 가 속한 블록, 로컬 인덱스 계산
    int block = (f / (2 * bs)) % nb;
    int loc   = f % (2 * bs);
    bool hi   = loc >= bs;
    int off   = loc % bs;

    int p_idx = p_block_offset + 2*block;
    scalar_t a = params[p_idx + 0];
    scalar_t bval = params[p_idx + 1];

    // 순전파에서 x1,x2 의 위치
    int base = b * D + block * (2 * bs);
    int i1   = base + off;
    int i2   = base + off + bs;

    // upstream gradient
    scalar_t g = grad_out[idx];

    // grad_input: 회전 행렬 G = [[a, b],[-b, a]] 의 transpose 곱
    if (!hi) {
        // y = a*x1 + b*x2
        atomicAdd(&grad_input[i1], a * g);
        atomicAdd(&grad_input[i2], bval * g);
    } else {
        // y = -b*x1 + a*x2
        atomicAdd(&grad_input[i1], -bval * g);
        atomicAdd(&grad_input[i2],  a * g);
    }

    // grad_params: 한 블록당 한 스레드만 계산하도록
    // offset==0 && hi==false 조건을 주면 블록당 1개만 실행
    if (off == 0 && !hi) {
        scalar_t grad_a = 0, grad_b = 0;
        // 이 블록 내부 bs 쌍들 전체를 합산
        for (int j = 0; j < bs; ++j) {
            int idx1 = base + j;
            int idx2 = base + j + bs;
            scalar_t x1 = input[idx1], x2 = input[idx2];
            scalar_t go1 = grad_out[idx1], go2 = grad_out[idx2];
            // ∂L/∂a = x1*go1 + x2*go2
            grad_a += x1 * go1 + x2 * go2;
            // ∂L/∂b = x2*go1 - x1*go2
            grad_b += x2 * go1 - x1 * go2;
        }
        atomicAdd(&grad_params[p_idx + 0], grad_a);
        atomicAdd(&grad_params[p_idx + 1], grad_b);
    }
}

std::vector<torch::Tensor> butterfly_layer_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx)
{
    CHECK_CUDA_CONTIGUOUS(grad_out);
    CHECK_CUDA_CONTIGUOUS(input);
    CHECK_CUDA_CONTIGUOUS(params);

    int B = grad_out.size(0), D = grad_out.size(1);
    auto grad_input  = torch::zeros_like(input);
    auto grad_params = torch::zeros_like(params);

    int threads = 256;
    int blocks  = (B*D + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "butterfly_layer_backward_cuda", ([&] {
        butterfly_layer_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            params.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            grad_params.data_ptr<scalar_t>(),
            B, D, layer_idx
        );
    }));
    cudaDeviceSynchronize();

    return { grad_input, grad_params };
}
