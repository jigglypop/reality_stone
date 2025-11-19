/**
 * CUDA kernel unit tests
 * Compile: nvcc -std=c++11 -arch=sm_70 -I.. test_kernels.cu poincare.cu lorentz.cu klein.cu mobius.cu -o test_kernels
 * Run:     ./test_kernels
 */

#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define EPSILON 1e-5f
#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        printf("FAIL: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
        return false; \
    }

#define TEST_ASSERT_NEAR(a, b, eps, msg) \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s\n  expected=%.6f got=%.6f diff=%.6e (tol=%.6e)\n  at %s:%d\n", \
               msg, (b), (a), fabsf((a)-(b)), (eps), __FILE__, __LINE__); \
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
// Poincare Tests
// ============================================================================

bool test_poincare_distance_same_point() {
    printf("Poincare distance: same point ... ");
    
    float x[] = {0.1f, 0.2f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    poincare_distance_cuda(d_out, d_x, d_x, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-4f, "Poincare distance to self should be 0");
    printf("PASS\n");
    return true;
}

bool test_poincare_distance_origin() {
    printf("Poincare distance: origin ... ");
    
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
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-5f, "Poincare distance at origin should be 0");
    printf("PASS\n");
    return true;
}

bool test_poincare_ball_layer_interpolation() {
    printf("Poincare layer: endpoints t=0, t=1 ... ");
    
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
    
    printf("PASS\n");
    return true;
}

bool test_poincare_distance_symmetry() {
    printf("Poincare distance: symmetry d(x,y)=d(y,x) ... ");

    float x[] = {0.1f, 0.2f};
    float y[] = {0.2f, -0.1f};
    float c = 1.0f;

    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_out1;
    float* d_out2;
    cudaMalloc(&d_out1, sizeof(float));
    cudaMalloc(&d_out2, sizeof(float));

    poincare_distance_cuda(d_out1, d_x, d_y, c, 1, 2);
    poincare_distance_cuda(d_out2, d_y, d_x, c, 1, 2);

    float d_xy;
    float d_yx;
    from_gpu(&d_xy, d_out1, 1);
    from_gpu(&d_yx, d_out2, 1);
    cudaFree(d_x);
    cudaFree(d_y);

    TEST_ASSERT_NEAR(d_xy, d_yx, 1e-5f, "Poincare distance symmetry violated");
    printf("PASS\n");
    return true;
}

bool test_poincare_triangle_inequality() {
    printf("Poincare distance: triangle inequality ... ");

    float x[] = {0.0f, 0.0f};
    float y[] = {0.1f, 0.1f};
    float z[] = {0.2f, -0.05f};
    float c = 1.0f;

    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_z = to_gpu(z, 2);

    float* d_out;
    cudaMalloc(&d_out, sizeof(float));

    poincare_distance_cuda(d_out, d_x, d_y, c, 1, 2);
    float d_xy;
    from_gpu(&d_xy, d_out, 1);

    cudaMalloc(&d_out, sizeof(float));
    poincare_distance_cuda(d_out, d_y, d_z, c, 1, 2);
    float d_yz;
    from_gpu(&d_yz, d_out, 1);

    cudaMalloc(&d_out, sizeof(float));
    poincare_distance_cuda(d_out, d_x, d_z, c, 1, 2);
    float d_xz;
    from_gpu(&d_xz, d_out, 1);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    TEST_ASSERT(d_xz <= d_xy + d_yz + 1e-4f, "Poincare triangle inequality violated");
    printf("PASS\n");
    return true;
}

bool test_poincare_ball_layer_inside_ball() {
    printf("Poincare layer: t=0.5 stays inside ball ... ");

    float u[] = {0.3f, 0.4f};
    float v[] = {-0.2f, 0.1f};
    float c = 1.0f;

    float* d_u = to_gpu(u, 2);
    float* d_v = to_gpu(v, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    poincare_ball_layer_cuda(d_out, d_u, d_v, c, 0.5f, 1, 2);
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(result[0] * result[0] + result[1] * result[1]);
    float radius = 1.0f / sqrtf(c);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);

    TEST_ASSERT(norm < radius - 1e-3f, "Poincare interpolation exceeds ball boundary");
    printf("PASS\n");
    return true;
}

// ============================================================================
// Lorentz Tests
// ============================================================================

bool test_lorentz_distance_same_point() {
    printf("Lorentz distance: same point ... ");
    
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
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-3f, "Lorentz distance to self should be 0");
    printf("PASS\n");
    return true;
}

bool test_lorentz_layer_interpolation() {
    printf("Lorentz layer: endpoints t=0, t=1 and constraint ... ");
    
    float c = 1.0f;
    float u_space_norm_sq = 0.3f * 0.3f + 0.4f * 0.4f;
    float u0 = sqrtf(1.0f / c + u_space_norm_sq);
    float u[] = {u0, 0.3f, 0.4f};
    float v_space_norm_sq = (-0.2f) * (-0.2f) + 0.1f * 0.1f;
    float v0 = sqrtf(1.0f / c + v_space_norm_sq);
    float v[] = {v0, -0.2f, 0.1f};
    
    float* d_u = to_gpu(u, 3);
    float* d_v = to_gpu(v, 3);
    float* d_out;
    cudaMalloc(&d_out, 3 * sizeof(float));
    
    lorentz_layer_forward_cuda(d_out, d_u, d_v, c, 0.0f, 1, 3);
    float result_t0[3];
    cudaMemcpy(result_t0, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t0[0], u[0], 1e-4f, "t=0: time component mismatch with u[0]");
    TEST_ASSERT_NEAR(result_t0[1], u[1], 1e-4f, "t=0: spatial component u[1] mismatch");
    TEST_ASSERT_NEAR(result_t0[2], u[2], 1e-4f, "t=0: spatial component u[2] mismatch");

    lorentz_layer_forward_cuda(d_out, d_u, d_v, c, 1.0f, 1, 3);
    float result_t1[3];
    cudaMemcpy(result_t1, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    TEST_ASSERT_NEAR(result_t1[0], v[0], 1e-4f, "t=1: time component mismatch with v[0]");
    TEST_ASSERT_NEAR(result_t1[1], v[1], 1e-4f, "t=1: spatial component v[1] mismatch");
    TEST_ASSERT_NEAR(result_t1[2], v[2], 1e-4f, "t=1: spatial component v[2] mismatch");

    float diff_u = fabsf(u[0] * u[0] - (u[1] * u[1] + u[2] * u[2]) - 1.0f / c);
    float diff_v = fabsf(v[0] * v[0] - (v[1] * v[1] + v[2] * v[2]) - 1.0f / c);
    float diff_t0 = fabsf(result_t0[0] * result_t0[0] - (result_t0[1] * result_t0[1] + result_t0[2] * result_t0[2]) - 1.0f / c);
    float diff_t1 = fabsf(result_t1[0] * result_t1[0] - (result_t1[1] * result_t1[1] + result_t1[2] * result_t1[2]) - 1.0f / c);

    TEST_ASSERT(diff_u < 1e-4f, "input u violates Lorentz hyperboloid constraint");
    TEST_ASSERT(diff_v < 1e-4f, "input v violates Lorentz hyperboloid constraint");
    TEST_ASSERT(diff_t0 < 1e-3f, "t=0 output violates Lorentz hyperboloid constraint");
    TEST_ASSERT(diff_t1 < 1e-3f, "t=1 output violates Lorentz hyperboloid constraint");
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

// ============================================================================
// Klein Tests
// ============================================================================

bool test_klein_distance_same_point() {
    printf("Klein distance: same point ... ");
    
    float x[] = {0.1f, 0.2f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    klein_distance_cuda(d_out, d_x, d_x, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-3f, "Klein distance to self should be 0");
    printf("PASS\n");
    return true;
}

bool test_klein_layer_inside_ball() {
    printf("Klein layer: t=0.5 stays inside ball ... ");

    float x[] = {0.1f, 0.2f};
    float y[] = {-0.1f, 0.1f};
    float c = 1.0f;

    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    klein_layer_forward_cuda(d_out, d_x, d_y, c, 0.5f, 1, 2);
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(result[0] * result[0] + result[1] * result[1]);
    float radius = 1.0f / sqrtf(c);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    TEST_ASSERT(norm < radius - 1e-3f, "Klein layer output exceeds ball boundary");
    printf("PASS\n");
    return true;
}

// ============================================================================
// Möbius Tests
// ============================================================================

bool test_mobius_add_identity() {
    printf("Mobius add: identity u+0=u ... ");
    
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
    
    TEST_ASSERT_NEAR(result[0], u[0], EPSILON, "u+0: first component mismatch");
    TEST_ASSERT_NEAR(result[1], u[1], EPSILON, "u+0: second component mismatch");
    
    cudaFree(d_u);
    cudaFree(d_zero);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_mobius_scalar_zero() {
    printf("Mobius scalar: r=0 ... ");
    
    float u[] = {0.3f, 0.4f};
    float c = 1.0f;
    float r = 0.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], 0.0f, EPSILON, "r=0: first component not zero");
    TEST_ASSERT_NEAR(result[1], 0.0f, EPSILON, "r=0: second component not zero");
    
    cudaFree(d_u);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_mobius_scalar_identity() {
    printf("Mobius scalar: r=1 ... ");
    
    float u[] = {0.1f, 0.2f};
    float c = 1.0f;
    float r = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], u[0], 1e-3f, "r=1: first component mismatch");
    TEST_ASSERT_NEAR(result[1], u[1], 1e-3f, "r=1: second component mismatch");
    
    cudaFree(d_u);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_mobius_add_inside_ball() {
    printf("Mobius add: stays inside ball ... ");

    float u[] = {0.2f, 0.1f};
    float v[] = {0.1f, -0.1f};
    float c = 1.0f;

    float* d_u = to_gpu(u, 2);
    float* d_v = to_gpu(v, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    mobius_add_cuda(d_out, d_u, d_v, c, 1, 2);

    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(result[0] * result[0] + result[1] * result[1]);
    float radius = 1.0f;

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);

    TEST_ASSERT(norm < radius - 1e-3f, "Mobius add result exceeds ball boundary");
    printf("PASS\n");
    return true;
}

bool test_mobius_scalar_euclidean_limit() {
    printf("Mobius scalar: c=0 Euclidean limit ... ");

    float u[] = {0.3f, -0.4f};
    float c = 0.0f;
    float r = 2.0f;

    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);

    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_out);

    TEST_ASSERT_NEAR(result[0], r * u[0], 1e-6f, "c=0 mode mobius_scalar[0] mismatch");
    TEST_ASSERT_NEAR(result[1], r * u[1], 1e-6f, "c=0 mode mobius_scalar[1] mismatch");
    printf("PASS\n");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    printf("\n");
    printf("=======================================================\n");
    printf("        CUDA kernel unit tests\n");
    printf("=======================================================\n\n");
    
    int passed = 0;
    int total = 0;
    
    printf("Poincare Tests:\n");
    total++; if (test_poincare_distance_same_point()) passed++;
    total++; if (test_poincare_distance_origin()) passed++;
    total++; if (test_poincare_ball_layer_interpolation()) passed++;
    total++; if (test_poincare_distance_symmetry()) passed++;
    total++; if (test_poincare_triangle_inequality()) passed++;
    total++; if (test_poincare_ball_layer_inside_ball()) passed++;
    
    printf("\nLorentz Tests:\n");
    total++; if (test_lorentz_distance_same_point()) passed++;
    total++; if (test_lorentz_layer_interpolation()) passed++;
    
    printf("\nKlein Tests:\n");
    total++; if (test_klein_distance_same_point()) passed++;
    total++; if (test_klein_layer_inside_ball()) passed++;
    
    printf("\nMobius Tests:\n");
    total++; if (test_mobius_add_identity()) passed++;
    total++; if (test_mobius_scalar_zero()) passed++;
    total++; if (test_mobius_scalar_identity()) passed++;
    total++; if (test_mobius_add_inside_ball()) passed++;
    total++; if (test_mobius_scalar_euclidean_limit()) passed++;
    
    printf("\n=======================================================\n");
    printf("Result: %d/%d tests passed", passed, total);
    if (passed == total) {
        printf(" [OK]\n");
    } else {
        printf(" [FAIL]\n");
    }
    printf("=======================================================\n\n");
    
    return (passed == total) ? 0 : 1;
}

