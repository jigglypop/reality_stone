#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace reality_stone::advanced {

// 올바른 tanh 체비셰프 계수
__constant__ float d_tanh_coeffs[16] = {
    0.0f,           // T_0
    0.7615941560f,  // T_1
    0.0f,           // T_2
    -0.2299769450f, // T_3
    0.0f,           // T_4
    0.0469100770f,  // T_5
    0.0f,           // T_6
    -0.0081518632f, // T_7
    0.0f,           // T_8
    0.0013182420f,  // T_9
    0.0f,           // T_10
    -0.0002049517f, // T_11
    0.0f,           // T_12
    0.0000308396f,  // T_13
    0.0f,           // T_14
    -0.0000045292f  // T_15
};

// 동적 곡률 제한 해제 함수들
__device__ float get_dynamic_limit(float curvature) {
    // 곡률에 따른 동적 제한값 계산
    float base_limit = 0.999f;
    float curvature_factor = fminf(curvature, 10.0f);  // 최대 10까지
    return base_limit + (1.0f - base_limit) * (curvature_factor / 10.0f);
}

__device__ float get_dynamic_scale_limit(float curvature) {
    // 스케일링 제한값을 곡률에 따라 동적 조정
    if (curvature <= 1.0f) return 3.0f;      // 기본값
    if (curvature <= 5.0f) return 5.0f;      // 중간 곡률
    return 10.0f + curvature * 0.5f;         // 높은 곡률: 더 넓은 범위
}

__device__ float chebyshev_polynomial(float x, int n) {
    // Clenshaw 재귀로 효율적 계산
    if (n == 0) return 1.0f;
    if (n == 1) return x;
    // T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
    float T_prev2 = 1.0f;
    float T_prev1 = x;
    for (int i = 2; i <= n; ++i) {
        float T_curr = 2.0f * x * T_prev1 - T_prev2;
        T_prev2 = T_prev1;
        T_prev1 = T_curr;
    }
    
    return T_prev1;
}

__global__ void chebyshev_approximation_kernel(
    const float* x,
    float* result,
    int batch_size,
    int dim,
    int order,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;
    
    float x_val = x[idx];
    float sqrt_c = sqrtf(curvature);
    
    // 동적 곡률 제한 해제: 곡률에 따른 스케일링 범위 조정
    const float L = get_dynamic_scale_limit(curvature);  // 동적 스케일 범위
    float scaled_x = sqrt_c * x_val / L;
    
    // 범위 확인 (동적 제한 적용)
    if (fabsf(sqrt_c * x_val) > L) {
        // 범위 밖: tanh(큰값) ≈ ±1
        result[idx] = (x_val > 0) ? 0.99999f : -0.99999f;
        return;
    }
    
    // 동적 제한으로 클리핑 범위 확장
    float dynamic_limit = get_dynamic_limit(curvature);
    scaled_x = fmaxf(-dynamic_limit, fminf(dynamic_limit, scaled_x));
    
    float sum = 0.0f;
    int max_order = min(order, 15);  // 계수 배열 크기 제한
    for (int n = 0; n <= max_order; ++n) {
        float coeff = d_tanh_coeffs[n];
        if (fabsf(coeff) < 1e-8f) continue;  // 0인 계수 스킵
        float T_n = chebyshev_polynomial(scaled_x, n);
        sum += coeff * T_n;
    }
    result[idx] = sum;
}

__global__ void chebyshev_distance_kernel(
    const float* x,
    const float* y,
    float* result,
    int batch_size,
    int dim,
    float curvature
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    // 체비셰프 거리 (max norm) 계산
    float max_diff = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = fabsf(x[batch_idx * dim + d] - y[batch_idx * dim + d]);
        max_diff = fmaxf(max_diff, diff);
    }
    // 하이퍼볼릭 공간 변환: d_H = (1/√c) * atanh(√c * d_cheb)
    float sqrt_c = sqrtf(curvature);
    
    // 동적 곡률 제한 해제: 거리 계산시 제한값 확장
    float distance_limit = get_dynamic_limit(curvature);
    float scaled_dist = fmaxf(0.0f, fminf(distance_limit, sqrt_c * max_diff));
    
    result[batch_idx] = (1.0f / sqrt_c) * atanhf(scaled_dist);
}

__global__ void fast_chebyshev_transform_kernel(
    const float* values,
    float* coeffs,
    int n
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    
    float coeff = 0.0f;
    
    // Type-I DCT 구현
    for (int j = 0; j < n; ++j) {
        float x_j = cosf(M_PI * j / (n - 1));  // 체비셰프 노드
        float T_k = chebyshev_polynomial(x_j, k);
        coeff += values[j] * T_k;
    }
    
    // 정규화
    float norm_factor = (k == 0 || k == n - 1) ? 1.0f : 2.0f;
    coeffs[k] = norm_factor * coeff / n;
}

torch::Tensor chebyshev_approximation_cuda(
    const torch::Tensor& x,
    int order,
    float curvature
) {
    auto batch_size = x.size(0);
    auto dim = x.size(-1);
    auto result = torch::zeros_like(x);
    
    const int threads = 256;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    chebyshev_approximation_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size,
        dim,
        order,
        curvature
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor chebyshev_distance_cuda(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
) {
    auto batch_size = x.size(0);
    auto dim = x.size(-1);
    auto result = torch::zeros({batch_size}, x.options());
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    chebyshev_distance_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size,
        dim,
        curvature
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor fast_chebyshev_transform_cuda(
    const torch::Tensor& values
) {
    auto n = values.size(-1);
    auto coeffs = torch::zeros_like(values);
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    fast_chebyshev_transform_kernel<<<blocks, threads>>>(
        values.data_ptr<float>(),
        coeffs.data_ptr<float>(),
        n
    );
    
    cudaDeviceSynchronize();
    return coeffs;
}

__global__ void inverse_chebyshev_transform_kernel(
    const float* coeffs,
    const float* eval_points,
    float* result,
    int batch_size,
    int order,
    int eval_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * eval_size) return;
    
    int batch_idx = idx / eval_size;
    int eval_idx = idx % eval_size;
    
    float x = eval_points[eval_idx];
    x = fmaxf(-0.999f, fminf(0.999f, x));  // clamp
    
    float sum = 0.0f;
    for (int k = 0; k <= order; ++k) {
        float T_k = chebyshev_polynomial(x, k);
        sum += coeffs[batch_idx * (order + 1) + k] * T_k;
    }
    
    result[idx] = sum;
}

__global__ void chebyshev_derivative_kernel(
    const float* coeffs,
    float* d_coeffs,
    int batch_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * (n - 1)) return;
    
    int batch_idx = idx / (n - 1);
    int k = idx % (n - 1);
    
    float d_coeff = 0.0f;
    
    // 체비셰프 미분 점화식
    if (k == n - 2) {
        d_coeff = 2.0f * (k + 1) * coeffs[batch_idx * n + (k + 1)];
    } else {
        d_coeff = d_coeffs[batch_idx * (n - 1) + (k + 2)] + 2.0f * (k + 1) * coeffs[batch_idx * n + (k + 1)];
    }
    
    d_coeffs[idx] = d_coeff;
}

__global__ void chebyshev_integral_kernel(
    const float* coeffs,
    float* int_coeffs,
    float constant,
    int batch_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * (n + 1)) return;
    
    int batch_idx = idx / (n + 1);
    int k = idx % (n + 1);
    
    if (k == 0) {
        int_coeffs[idx] = constant;
    } else if (k == 1 && n > 2) {
        int_coeffs[idx] = coeffs[batch_idx * n + 0] - coeffs[batch_idx * n + 2] / 4.0f;
    } else if (k == 1) {
        int_coeffs[idx] = coeffs[batch_idx * n + 0];
    } else if (k == n) {
        int_coeffs[idx] = coeffs[batch_idx * n + (n - 1)] / (2.0f * k);
    } else {
        int_coeffs[idx] = (coeffs[batch_idx * n + (k - 1)] - coeffs[batch_idx * n + (k + 1)]) / (2.0f * k);
    }
}

torch::Tensor inverse_chebyshev_transform_cuda(
    const torch::Tensor& coeffs,
    const torch::Tensor& eval_points
) {
    auto batch_size = coeffs.size(0);
    auto order = coeffs.size(-1) - 1;
    auto eval_size = eval_points.size(-1);
    auto result = torch::zeros({batch_size, eval_size}, coeffs.options());
    
    const int threads = 256;
    const int blocks = (batch_size * eval_size + threads - 1) / threads;
    
    inverse_chebyshev_transform_kernel<<<blocks, threads>>>(
        coeffs.data_ptr<float>(),
        eval_points.data_ptr<float>(),
        result.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(order),
        static_cast<int>(eval_size)
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor chebyshev_derivative_cuda(
    const torch::Tensor& coeffs
) {
    auto batch_size = coeffs.size(0);
    auto n = coeffs.size(-1);
    if (n <= 1) return torch::zeros({batch_size, 1}, coeffs.options());
    auto d_coeffs = torch::zeros({batch_size, n - 1}, coeffs.options());
    const int threads = 256;
    const int blocks = (batch_size * (n - 1) + threads - 1) / threads;
    // 역순으로 계산 (점화식 때문)
    for (int k = n - 2; k >= 0; --k) {
        dim3 grid((batch_size + threads - 1) / threads);
        chebyshev_derivative_kernel<<<grid, threads>>>(
            coeffs.data_ptr<float>(),
            d_coeffs.data_ptr<float>(),
            static_cast<int>(batch_size),
            static_cast<int>(n)
        );
    }
    
    cudaDeviceSynchronize();
    return d_coeffs;
}

torch::Tensor chebyshev_integral_cuda(
    const torch::Tensor& coeffs,
    float constant
) {
    auto batch_size = coeffs.size(0);
    auto n = coeffs.size(-1);
    auto int_coeffs = torch::zeros({batch_size, n + 1}, coeffs.options());
    
    const int threads = 256;
    const int blocks = (batch_size * (n + 1) + threads - 1) / threads;
    
    chebyshev_integral_kernel<<<blocks, threads>>>(
        coeffs.data_ptr<float>(),
        int_coeffs.data_ptr<float>(),
        constant,
        static_cast<int>(batch_size),
        static_cast<int>(n)
    );
    
    cudaDeviceSynchronize();
    return int_coeffs;
}

}