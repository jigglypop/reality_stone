#pragma once

#include <torch/extension.h>
#include <config/constant.h>

namespace reality_stone::advanced {

// CUDA 함수 선언들
torch::Tensor hyperbolic_laplacian_cuda(
    const torch::Tensor& f,
    float curvature
);

torch::Tensor heat_kernel_cuda(
    const torch::Tensor& x,
    float t,
    float curvature
);

std::tuple<torch::Tensor, torch::Tensor> laplace_beltrami_eigen_cuda(
    const torch::Tensor& manifold_points,
    float curvature
);

torch::Tensor spectral_graph_conv_cuda(
    const torch::Tensor& x,
    const torch::Tensor& laplacian,
    const torch::Tensor& weight
);

torch::Tensor solve_diffusion_equation_cuda(
    const torch::Tensor& initial_condition,
    float time_step,
    int num_steps,
    float curvature
);

torch::Tensor geodesic_distance_matrix_cuda(
    const torch::Tensor& points,
    float curvature
);

torch::Tensor spectral_normalize_cuda(
    const torch::Tensor& adjacency_matrix
);

/**
 * 라플라스-벨트라미 연산자 클래스
 * 하이퍼볼릭 공간에서의 스펙트럴 분석을 위한 도구
 */
class LaplaceBeltramiOperator {
public:
    LaplaceBeltramiOperator(float curvature = 1.0f, int max_eigenvalues = 100);
    
    // 하이퍼볼릭 라플라시안 계산
    torch::Tensor compute_laplacian(const torch::Tensor& f);
    
    // 열 커널 (Heat kernel) 계산
    torch::Tensor compute_heat_kernel(
        const torch::Tensor& x,
        float t
    );
    
    // 고유값 분해
    std::tuple<torch::Tensor, torch::Tensor> eigen_decomposition(
        const torch::Tensor& manifold_points
    );
    
    // 스펙트럴 클러스터링
    torch::Tensor spectral_clustering(
        const torch::Tensor& points,
        int num_clusters
    );
    
    // 스펙트럴 임베딩
    torch::Tensor spectral_embedding(
        const torch::Tensor& points,
        int embed_dim
    );
    
    // 그래프 라플라시안 (그래프 신경망용)
    torch::Tensor graph_laplacian(
        const torch::Tensor& adjacency_matrix
    );
    
    // 매니폴드 거리 계산 (public으로 이동)
    torch::Tensor compute_manifold_distance(
        const torch::Tensor& x,
        const torch::Tensor& y
    );
    
    // 가우시안 커널
    torch::Tensor gaussian_kernel(
        const torch::Tensor& distances,
        float sigma
    );
    
private:
    float curvature;
    int max_eigenvalues;
    torch::Tensor eigenvalues_cache;
    torch::Tensor eigenvectors_cache;
    bool cache_valid;
};

/**
 * 편의 함수들 (절차적 인터페이스)
 */

// 하이퍼볼릭 라플라시안
torch::Tensor hyperbolic_laplacian_cpu(
    const torch::Tensor& f,
    float curvature = 1.0f
);

// 열 확산 (Heat diffusion)
torch::Tensor heat_kernel_cpu(
    const torch::Tensor& x,
    float t,
    float curvature = 1.0f
);

// 고유값 분해
std::tuple<torch::Tensor, torch::Tensor> laplace_beltrami_eigen_cpu(
    const torch::Tensor& manifold_points,
    float curvature = 1.0f
);

// 스펙트럴 그래프 컨볼루션
torch::Tensor spectral_graph_conv_cpu(
    const torch::Tensor& x,
    const torch::Tensor& laplacian,
    const torch::Tensor& weight
);

// 확산 방정식 해법
torch::Tensor solve_diffusion_equation_cpu(
    const torch::Tensor& initial_condition,
    float time_step,
    int num_steps,
    float curvature = 1.0f
);

// 측지선 거리 계산 (더 정확한 방법)
torch::Tensor geodesic_distance_matrix_cpu(
    const torch::Tensor& points,
    float curvature = 1.0f
);

// 스펙트럴 정규화
torch::Tensor spectral_normalize_cpu(
    const torch::Tensor& adjacency_matrix
);

} // namespace reality_stone::advanced 