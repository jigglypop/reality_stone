#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#include <torch/extension.h>
#include <advanced/hyperbolic_fft/hyperbolic_fft.h>
#include <cmath>
#include <cuComplex.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

namespace reality_stone::advanced {

#ifdef WITH_CUDA

// CUDA 유틸리티 함수들
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float safe_norm(float x, float y, float z = 0.0f) {
    return sqrtf(fmaxf(x*x + y*y + z*z, 1e-8f));
}

// 하이퍼볼릭 FFT 커널
__global__ void hyperbolic_fft_kernel(
    const float* __restrict__ x,        // [B, D]
    float* __restrict__ coeffs,         // [B, (max_l+1)^2]
    float curvature,
    int batch_size,
    int dim,
    int max_l
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* x_batch = x + bid * dim;
    float* coeffs_batch = coeffs + bid * (max_l + 1) * (max_l + 1);
    
    __shared__ float r_shared;
    __shared__ float theta_shared;
    __shared__ float phi_shared;
    
    // 구면 좌표 계산 (3D 가정)
    if (tid == 0) {
        float x_val = x_batch[0];
        float y_val = dim > 1 ? x_batch[1] : 0.0f;
        float z_val = dim > 2 ? x_batch[2] : 0.0f;
        
        r_shared = safe_norm(x_val, y_val, z_val);
        theta_shared = acosf(fmaxf(fminf(z_val / r_shared, 1.0f), -1.0f));
        phi_shared = atan2f(y_val, x_val);
    }
    __syncthreads();
    
    // 각 스레드가 다른 (l, m) 조합 처리
    int total_coeffs = (max_l + 1) * (max_l + 1);
    
    for (int idx = tid; idx < total_coeffs; idx += blockDim.x) {
        // idx에서 l, m 추출
        int l = (int)sqrtf((float)idx);
        while (l * l + 2 * l < idx) l++;
        int m = idx - l * l - l;
        
        if (l <= max_l && abs(m) <= l) {
            // 구면 조화 함수 계산 (간단한 버전)
            float cos_theta = cosf(theta_shared);
            float legendre_val = 1.0f; // P_l^m(cos_theta) 근사
            
            // Legendre 다항식 근사
            if (l > 0) {
                legendre_val = cos_theta;
                for (int n = 2; n <= l; ++n) {
                    legendre_val = ((2*n-1) * cos_theta * legendre_val) / n;
                }
            }
            
            // 구면 조화 함수
            float normalization = sqrtf((2.0f * l + 1) / (4.0f * M_PI));
            float harmonic = normalization * legendre_val * cosf(m * phi_shared);
            
            coeffs_batch[idx] = harmonic * r_shared;
        }
    }
}

// 구면 조화 함수 커널
__global__ void spherical_harmonics_kernel(
    const float* __restrict__ theta_phi,  // [B, 2]
    float* __restrict__ harmonics,        // [B, (l_max+1)^2]
    int l_max,
    int batch_size
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    float theta = theta_phi[bid * 2];
    float phi = theta_phi[bid * 2 + 1];
    float cos_theta = cosf(theta);
    
    float* harmonics_batch = harmonics + bid * (l_max + 1) * (l_max + 1);
    
    int total_coeffs = (l_max + 1) * (l_max + 1);
    
    for (int idx = tid; idx < total_coeffs; idx += blockDim.x) {
        int l = (int)sqrtf((float)idx);
        while (l * l + 2 * l < idx) l++;
        int m = idx - l * l - l;
        
        if (l <= l_max && abs(m) <= l) {
            // Legendre 다항식 계산
            float legendre_val = 1.0f;
            if (l > 0) {
                legendre_val = cos_theta;
                for (int n = 2; n <= l; ++n) {
                    legendre_val = ((2*n-1) * cos_theta * legendre_val) / n;
                }
            }
            
            float normalization = sqrtf((2.0f * l + 1) / (4.0f * M_PI));
            harmonics_batch[idx] = normalization * legendre_val * cosf(m * phi);
        }
    }
}

// 빠른 구면 컨볼루션 커널
__global__ void fast_spherical_conv_kernel(
    const float* __restrict__ f_coeffs,
    const float* __restrict__ g_coeffs,
    float* __restrict__ result_coeffs,
    int batch_size,
    int num_coeffs
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* f_batch = f_coeffs + bid * num_coeffs;
    const float* g_batch = g_coeffs + bid * num_coeffs;
    float* result_batch = result_coeffs + bid * num_coeffs;
    
    for (int i = tid; i < num_coeffs; i += blockDim.x) {
        result_batch[i] = f_batch[i] * g_batch[i];
    }
}

// 리치 곡률 커널
__global__ void ricci_curvature_kernel(
    const float* __restrict__ metric_tensor,
    float* __restrict__ ricci_scalars,
    float curvature,
    int batch_size,
    int dim
) {
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    // 하이퍼볼릭 공간에서 리치 곡률은 상수
    ricci_scalars[bid] = -(dim - 1) * curvature;
}

// 평행 이동 커널
__global__ void parallel_transport_kernel(
    const float* __restrict__ vector,
    const float* __restrict__ path,
    float* __restrict__ transported,
    float curvature,
    float path_parameter,
    int batch_size,
    int dim
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* vector_batch = vector + bid * dim;
    const float* start_point = path + bid * dim * 2;  // 시작점
    float* transported_batch = transported + bid * dim;
    
    __shared__ float dot_product;
    
    // start_point와 vector의 내적 계산
    float local_dot = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        local_dot += start_point[d] * vector_batch[d];
    }
    
    local_dot = warp_reduce_sum(local_dot);
    if (threadIdx.x == 0) {
        dot_product = local_dot;
    }
    __syncthreads();
    
    // 평행 이동 인수 계산
    float transport_factor = 1.0f - path_parameter * curvature * dot_product;
    
    for (int d = tid; d < dim; d += blockDim.x) {
        transported_batch[d] = vector_batch[d] * transport_factor;
    }
}

// 지오데식 플로우 커널
__global__ void geodesic_flow_kernel(
    const float* __restrict__ initial_point,
    const float* __restrict__ initial_velocity,
    float* __restrict__ result,
    float time,
    float curvature,
    int batch_size,
    int dim
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* point_batch = initial_point + bid * dim;
    const float* velocity_batch = initial_velocity + bid * dim;
    float* result_batch = result + bid * dim;
    
    __shared__ float velocity_norm;
    
    // 속도 노름 계산
    float local_norm = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = velocity_batch[d];
        local_norm += v * v;
    }
    
    local_norm = warp_reduce_sum(local_norm);
    if (threadIdx.x == 0) {
        velocity_norm = sqrtf(fmaxf(local_norm, 1e-8f));
    }
    __syncthreads();
    
    float sqrt_c = sqrtf(curvature);
    float sinh_term = sinhf(sqrt_c * velocity_norm * time) / (sqrt_c * velocity_norm);
    float cosh_term = coshf(sqrt_c * velocity_norm * time);
    
    for (int d = tid; d < dim; d += blockDim.x) {
        float unit_velocity = velocity_batch[d] / velocity_norm;
        result_batch[d] = point_batch[d] * cosh_term + unit_velocity * sinh_term;
    }
}

// 호스트 함수들
torch::Tensor hyperbolic_fft_cuda(const torch::Tensor& x, float curvature) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    int max_l = 10; // 기본값
    
    auto coeffs = torch::zeros({batch_size, (max_l + 1) * (max_l + 1)}, x.options());
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    hyperbolic_fft_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        coeffs.data_ptr<float>(),
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(dim),
        max_l
    );
    
    cudaDeviceSynchronize();
    return coeffs;
}

torch::Tensor inverse_hyperbolic_fft_cuda(const torch::Tensor& coeffs, float curvature) {
    int64_t N = coeffs.size(1);
    int64_t max_l = static_cast<int64_t>(std::floor(std::sqrt(static_cast<double>(N)))) - 1;
    HyperbolicFFT fft(curvature, static_cast<int>(max_l));
    return fft.inverse_transform(coeffs);
}

torch::Tensor spherical_harmonics_cuda(const torch::Tensor& theta_phi, int l_max) {
    auto batch_size = theta_phi.size(0);
    auto harmonics = torch::zeros({batch_size, (l_max + 1) * (l_max + 1)}, theta_phi.options());
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    spherical_harmonics_kernel<<<blocks, threads>>>(
        theta_phi.data_ptr<float>(),
        harmonics.data_ptr<float>(),
        l_max,
        static_cast<int>(batch_size)
    );
    
    cudaDeviceSynchronize();
    return harmonics;
}

torch::Tensor fast_spherical_conv_cuda(
    const torch::Tensor& f,
    const torch::Tensor& g,
    float curvature
) {
    // FFT 변환
    auto f_fft = hyperbolic_fft_cuda(f, curvature);
    auto g_fft = hyperbolic_fft_cuda(g, curvature);
    
    auto batch_size = f_fft.size(0);
    auto num_coeffs = f_fft.size(1);
    
    auto result = torch::zeros_like(f_fft);
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    fast_spherical_conv_kernel<<<blocks, threads>>>(
        f_fft.data_ptr<float>(),
        g_fft.data_ptr<float>(),
        result.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(num_coeffs)
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor ricci_curvature_cuda(const torch::Tensor& metric_tensor) {
    auto batch_size = metric_tensor.size(0);
    auto dim = metric_tensor.size(1);
    auto ricci_scalars = torch::zeros({batch_size}, metric_tensor.options());
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = 1;
    
    ricci_curvature_kernel<<<blocks, threads>>>(
        metric_tensor.data_ptr<float>(),
        ricci_scalars.data_ptr<float>(),
        1.0f, // 기본 곡률
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return ricci_scalars;
}

torch::Tensor parallel_transport_cuda(
    const torch::Tensor& v,
    const torch::Tensor& path,
    float curvature
) {
    auto batch_size = v.size(0);
    auto dim = v.size(1);
    auto transported = torch::zeros_like(v);
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    parallel_transport_kernel<<<blocks, threads>>>(
        v.data_ptr<float>(),
        path.data_ptr<float>(),
        transported.data_ptr<float>(),
        curvature,
        1.0f, // path_parameter
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return transported;
}

torch::Tensor geodesic_flow_cuda(
    const torch::Tensor& x,
    const torch::Tensor& v,
    float t,
    float curvature
) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto result = torch::zeros_like(x);
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    geodesic_flow_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        result.data_ptr<float>(),
        t,
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return result;
}

__global__ void riemannian_gradient_kernel(
    const float* __restrict__ euclidean_grad,
    const float* __restrict__ point,
    float* __restrict__ riemannian_grad,
    float curvature,
    int batch_size,
    int dim
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* point_batch = point + bid * dim;
    const float* grad_batch = euclidean_grad + bid * dim;
    float* riemannian_batch = riemannian_grad + bid * dim;
    
    __shared__ float point_norm_sq;
    
    // 점의 노름 제곱 계산
    float local_norm_sq = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float p = point_batch[d];
        local_norm_sq += p * p;
    }
    
    local_norm_sq = warp_reduce_sum(local_norm_sq);
    if (threadIdx.x == 0) {
        point_norm_sq = local_norm_sq;
    }
    __syncthreads();
    
    // 포인카레 모델의 conformal factor
    float conformal_factor = powf(1.0f - curvature * point_norm_sq, 2) / 4.0f;
    
    for (int d = tid; d < dim; d += blockDim.x) {
        riemannian_batch[d] = grad_batch[d] * conformal_factor;
    }
}

__global__ void geodesic_sgd_step_kernel(
    const float* __restrict__ point,
    const float* __restrict__ grad,
    float* __restrict__ new_point,
    float lr,
    float curvature,
    int batch_size,
    int dim
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* point_batch = point + bid * dim;
    const float* grad_batch = grad + bid * dim;
    float* new_point_batch = new_point + bid * dim;
    
    // 리만 그래디언트 계산
    __shared__ float point_norm_sq;
    
    float local_norm_sq = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float p = point_batch[d];
        local_norm_sq += p * p;
    }
    
    local_norm_sq = warp_reduce_sum(local_norm_sq);
    if (threadIdx.x == 0) {
        point_norm_sq = local_norm_sq;
    }
    __syncthreads();
    
    float conformal_factor = powf(1.0f - curvature * point_norm_sq, 2) / 4.0f;
    
    for (int d = tid; d < dim; d += blockDim.x) {
        float riemannian_grad = grad_batch[d] * conformal_factor;
        float scaled_grad = -lr * riemannian_grad;
        
        // 간단한 지오데식 업데이트 (근사)
        new_point_batch[d] = point_batch[d] + scaled_grad;
    }
}

torch::Tensor riemannian_gradient_cuda(
    const torch::Tensor& euclidean_grad,
    const torch::Tensor& x,
    float curvature
) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto riemannian_grad = torch::zeros_like(euclidean_grad);
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    riemannian_gradient_kernel<<<blocks, threads>>>(
        euclidean_grad.data_ptr<float>(),
        x.data_ptr<float>(),
        riemannian_grad.data_ptr<float>(),
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return riemannian_grad;
}

torch::Tensor geodesic_sgd_step_cuda(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    float lr,
    float curvature
) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto new_point = torch::zeros_like(x);
    
    const int blocks = static_cast<int>(batch_size);
    const int threads = BLOCK_SIZE;
    
    geodesic_sgd_step_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        grad.data_ptr<float>(),
        new_point.data_ptr<float>(),
        lr,
        curvature,
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    
    cudaDeviceSynchronize();
    return new_point;
}

torch::Tensor hyperbolic_wavelet_decomposition_cuda(
    const torch::Tensor& signal,
    int num_levels,
    float curvature
) {
    // 간단한 구현: FFT 기반 웨이블릿
    auto coeffs = torch::zeros_like(signal);
    auto current = signal.clone();
    
    for (int level = 0; level < num_levels; ++level) {
        auto fft_result = hyperbolic_fft_cuda(current, curvature);
        
        // 주파수 필터링 (GPU에서)
        auto filter = torch::exp(-torch::arange(fft_result.size(1), signal.options()) * 0.1f * (level + 1));
        auto filtered = fft_result * filter.unsqueeze(0);
        
        coeffs += filtered;
        current = filtered * 0.5f; // 다운샘플링
    }
    
    return coeffs;
}

torch::Tensor frequency_domain_filter_cuda(
    const torch::Tensor& signal,
    const torch::Tensor& filter,
    float curvature
) {
    auto signal_fft = hyperbolic_fft_cuda(signal, curvature);
    auto filtered_fft = signal_fft * filter;
    
    return filtered_fft; // 역변환은 생략
}

#else
// CUDA가 없을 때의 더미 함수들
torch::Tensor hyperbolic_fft_cuda(const torch::Tensor& x, float curvature) {
    TORCH_CHECK(false, "hyperbolic_fft_cuda requires CUDA");
}

torch::Tensor spherical_harmonics_cuda(const torch::Tensor& theta_phi, int l_max) {
    TORCH_CHECK(false, "spherical_harmonics_cuda requires CUDA");
}

torch::Tensor fast_spherical_conv_cuda(
    const torch::Tensor& f,
    const torch::Tensor& g,
    float curvature
) {
    TORCH_CHECK(false, "fast_spherical_conv_cuda requires CUDA");
}

torch::Tensor ricci_curvature_cuda(const torch::Tensor& metric_tensor) {
    TORCH_CHECK(false, "ricci_curvature_cuda requires CUDA");
}

torch::Tensor parallel_transport_cuda(
    const torch::Tensor& v,
    const torch::Tensor& path,
    float curvature
) {
    TORCH_CHECK(false, "parallel_transport_cuda requires CUDA");
}

torch::Tensor geodesic_flow_cuda(
    const torch::Tensor& x,
    const torch::Tensor& v,
    float t,
    float curvature
) {
    TORCH_CHECK(false, "geodesic_flow_cuda requires CUDA");
}

torch::Tensor riemannian_gradient_cuda(
    const torch::Tensor& euclidean_grad,
    const torch::Tensor& x,
    float curvature
) {
    TORCH_CHECK(false, "riemannian_gradient_cuda requires CUDA");
}

torch::Tensor geodesic_sgd_step_cuda(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    float lr,
    float curvature
) {
    TORCH_CHECK(false, "geodesic_sgd_step_cuda requires CUDA");
}

torch::Tensor hyperbolic_wavelet_decomposition_cuda(
    const torch::Tensor& signal,
    int num_levels,
    float curvature
) {
    TORCH_CHECK(false, "hyperbolic_wavelet_decomposition_cuda requires CUDA");
}

torch::Tensor frequency_domain_filter_cuda(
    const torch::Tensor& signal,
    const torch::Tensor& filter,
    float curvature
) {
    TORCH_CHECK(false, "frequency_domain_filter_cuda requires CUDA");
}
#endif

} // namespace reality_stone::advanced 