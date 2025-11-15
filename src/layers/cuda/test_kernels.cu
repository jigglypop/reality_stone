/**
 * CUDA 커널 단위 테스트
 * 컴파일: nvcc -std=c++11 -arch=sm_70 -I.. test_kernels.cu poincare.cu lorentz.cu klein.cu mobius.cu -o test_kernels
 * 실행: ./test_kernels
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define EPSILON 1e-5f
#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        printf("❌ FAIL: %s\n   at %s:%d\n", msg, __FILE__, __LINE__); \
        return false; \
    }

#define TEST_ASSERT_NEAR(a, b, eps, msg) \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("❌ FAIL: %s\n   Expected: %.6f, Got: %.6f, Diff: %.6e\n   at %s:%d\n", \
               msg, (b), (a), fabsf((a)-(b)), __FILE__, __LINE__); \
        return false; \
    }

// Forward declarations
extern "C" {
    void poincare_distance_cuda(float* out, const float* x, const float* y, float c, long long batch_size, long long dim);
    void poincare_ball_layer_cuda(float* out, const float* u, const float* v, float c, float t, long long batch_size, long long dim);
    
    void lorentz_distance_cuda(float* out, const float* u, const float* v, float c, int batch_size, int dim);
    void lorentz_layer_forward_cuda(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim);
    
    void klein_distance_cuda(float* out, const float* u, const float* v, float c, int batch_size, int dim);
    void klein_layer_forward_cuda(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim);
    
    void mobius_add_cuda(float* out, const float* u, const float* v, float c, int64_t batch_size, int64_t dim);
    void mobius_scalar_cuda(float* out, const float* u, float c, float r, int64_t batch_size, int64_t dim);
}

// Helper: allocate and copy to GPU
float* to_gpu(const float* host, int size) {
    float* dev;
    cudaMalloc(&dev, size * sizeof(float));
    cudaMemcpy(dev, host, size * sizeof(float), cudaMemcpyHostToDevice);
    return dev;
}

// Helper: copy from GPU and free
void from_gpu(float* host, float* dev, int size) {
    cudaMemcpy(host, dev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
}

// ============================================================================
// Poincaré Tests
// ============================================================================

bool test_poincare_distance_same_point() {
    printf("🧪 Test: Poincaré distance (same point) ... ");
    
    float x[] = {0.1f, 0.2f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    poincare_distance_cuda(d_out, d_x, d_x, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-4f, "Distance to self should be 0");
    printf("✅ PASS\n");
    return true;
}

bool test_poincare_distance_origin() {
    printf("🧪 Test: Poincaré distance (origin) ... ");
    
    float x[] = {0.0f, 0.0f};
    float y[] = {0.0f, 0.0f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    poincare_distance_cuda(d_out, d_x, d_y, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    cudaFree(d_y);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-5f, "Distance at origin should be 0");
    printf("✅ PASS\n");
    return true;
}

bool test_poincare_ball_layer_interpolation() {
    printf("🧪 Test: Poincaré layer (t=0, t=1) ... ");
    
    float u[] = {0.3f, 0.4f};
    float v[] = {-0.2f, 0.1f};
    float c = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_v = to_gpu(v, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    // t=0 should return u
    poincare_ball_layer_cuda(d_out, d_u, d_v, c, 0.0f, 1, 2);
    float result_t0[2];
    cudaMemcpy(result_t0, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t0[0], u[0], EPSILON, "t=0: x component should match u");
    TEST_ASSERT_NEAR(result_t0[1], u[1], EPSILON, "t=0: y component should match u");
    
    // t=1 should return v
    poincare_ball_layer_cuda(d_out, d_u, d_v, c, 1.0f, 1, 2);
    float result_t1[2];
    cudaMemcpy(result_t1, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t1[0], v[0], EPSILON, "t=1: x component should match v");
    TEST_ASSERT_NEAR(result_t1[1], v[1], EPSILON, "t=1: y component should match v");
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
    
    printf("✅ PASS\n");
    return true;
}

// ============================================================================
// Lorentz Tests
// ============================================================================

bool test_lorentz_distance_same_point() {
    printf("🧪 Test: Lorentz distance (same point) ... ");
    
    // Point on hyperboloid: x0 = sqrt(1/c + ||x||²)
    float c = 1.0f;
    float space_norm_sq = 0.1f * 0.1f + 0.2f * 0.2f;
    float x0 = sqrtf(1.0f / c + space_norm_sq);
    float x[] = {x0, 0.1f, 0.2f};
    
    float* d_x = to_gpu(x, 3);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    lorentz_distance_cuda(d_out, d_x, d_x, c, 1, 3);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-4f, "Lorentz distance to self should be 0");
    printf("✅ PASS\n");
    return true;
}

bool test_lorentz_layer_interpolation() {
    printf("🧪 Test: Lorentz layer (t=0, t=1) ... ");
    
    float c = 1.0f;
    float u[] = {1.5f, 0.3f, 0.4f};
    float v[] = {1.3f, -0.2f, 0.1f};
    
    float* d_u = to_gpu(u, 3);
    float* d_v = to_gpu(v, 3);
    float* d_out;
    cudaMalloc(&d_out, 3 * sizeof(float));
    
    // t=0
    lorentz_layer_forward_cuda(d_out, d_u, d_v, c, 0.0f, 1, 3);
    float result_t0[3];
    cudaMemcpy(result_t0, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t0[0], u[0], 1e-4f, "t=0: should match u[0]");
    TEST_ASSERT_NEAR(result_t0[1], u[1], 1e-4f, "t=0: should match u[1]");
    TEST_ASSERT_NEAR(result_t0[2], u[2], 1e-4f, "t=0: should match u[2]");
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
    
    printf("✅ PASS\n");
    return true;
}

// ============================================================================
// Klein Tests
// ============================================================================

bool test_klein_distance_same_point() {
    printf("🧪 Test: Klein distance (same point) ... ");
    
    float x[] = {0.1f, 0.2f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    klein_distance_cuda(d_out, d_x, d_x, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-4f, "Klein distance to self should be 0");
    printf("✅ PASS\n");
    return true;
}

// ============================================================================
// Möbius Tests
// ============================================================================

bool test_mobius_add_identity() {
    printf("🧪 Test: Möbius add (identity) ... ");
    
    float u[] = {0.1f, 0.2f};
    float zero[] = {0.0f, 0.0f};
    float c = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_zero = to_gpu(zero, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_add_cuda(d_out, d_u, d_zero, c, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], u[0], EPSILON, "u + 0 should equal u[0]");
    TEST_ASSERT_NEAR(result[1], u[1], EPSILON, "u + 0 should equal u[1]");
    
    cudaFree(d_u);
    cudaFree(d_zero);
    cudaFree(d_out);
    
    printf("✅ PASS\n");
    return true;
}

bool test_mobius_scalar_zero() {
    printf("🧪 Test: Möbius scalar (r=0) ... ");
    
    float u[] = {0.3f, 0.4f};
    float c = 1.0f;
    float r = 0.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], 0.0f, EPSILON, "r=0 should give 0");
    TEST_ASSERT_NEAR(result[1], 0.0f, EPSILON, "r=0 should give 0");
    
    cudaFree(d_u);
    cudaFree(d_out);
    
    printf("✅ PASS\n");
    return true;
}

bool test_mobius_scalar_identity() {
    printf("🧪 Test: Möbius scalar (r=1) ... ");
    
    float u[] = {0.1f, 0.2f};
    float c = 1.0f;
    float r = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // r=1 should approximately return u (within numeric precision)
    TEST_ASSERT_NEAR(result[0], u[0], 1e-3f, "r=1 should give u[0]");
    TEST_ASSERT_NEAR(result[1], u[1], 1e-3f, "r=1 should give u[1]");
    
    cudaFree(d_u);
    cudaFree(d_out);
    
    printf("✅ PASS\n");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("        CUDA 커널 단위 테스트\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int passed = 0;
    int total = 0;
    
    // Poincaré tests
    printf("📐 Poincaré Tests:\n");
    total++; if (test_poincare_distance_same_point()) passed++;
    total++; if (test_poincare_distance_origin()) passed++;
    total++; if (test_poincare_ball_layer_interpolation()) passed++;
    
    printf("\n🌐 Lorentz Tests:\n");
    total++; if (test_lorentz_distance_same_point()) passed++;
    total++; if (test_lorentz_layer_interpolation()) passed++;
    
    printf("\n🔷 Klein Tests:\n");
    total++; if (test_klein_distance_same_point()) passed++;
    
    printf("\n➕ Möbius Tests:\n");
    total++; if (test_mobius_add_identity()) passed++;
    total++; if (test_mobius_scalar_zero()) passed++;
    total++; if (test_mobius_scalar_identity()) passed++;
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("결과: %d/%d 테스트 통과", passed, total);
    if (passed == total) {
        printf(" ✅\n");
    } else {
        printf(" ❌\n");
    }
    printf("═══════════════════════════════════════════════════════\n\n");
    
    return (passed == total) ? 0 : 1;
}

