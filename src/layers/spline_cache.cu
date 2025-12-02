
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

// Device function for cubic hermite basis
__device__ void cubic_hermite(
    float u,
    float* h00, float* h10, float* h01, float* h11
) {
    float u2 = u * u;
    float u3 = u2 * u;
    *h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
    *h10 = u3 - 2.0f * u2 + u;
    *h01 = -2.0f * u3 + 3.0f * u2;
    *h11 = u3 - u2;
}

// Kernel: Reconstruct states for a batch of timestamps
// control_points: [num_points, 2 * dim] (state concatenated with velocity)
// times: [num_points]
// target_times: [batch_size]
// output: [batch_size, dim]
// curvature: float
__global__ void spline_reconstruct_kernel(
    const float* control_points, // interleaved state/velocity or separate? Let's assume contiguous [state, velocity] per point
    const float* cp_times,
    int num_points,
    int dim,
    const float* target_times,
    float* output,
    int batch_size,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float t = target_times[idx];
    
    // 1. Binary Search for interval
    // Simple linear scan or binary search. Since num_points is likely small to moderate, 
    // binary search is better.
    
    int left = 0;
    int right = num_points - 1;
    int p0_idx = -1;

    if (num_points == 0) return;
    
    if (t <= cp_times[0]) {
        p0_idx = 0; // Clamp to start
        // Just copy state of p0
        for (int i = 0; i < dim; i++) {
             output[idx * dim + i] = control_points[0 * 2 * dim + i];
        }
        return;
    }
    if (t >= cp_times[num_points - 1]) {
        p0_idx = num_points - 1;
        // Copy last
        for (int i = 0; i < dim; i++) {
             output[idx * dim + i] = control_points[p0_idx * 2 * dim + i];
        }
        return;
    }

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (cp_times[mid] <= t) {
            p0_idx = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    // t is between p0_idx and p0_idx + 1
    int p1_idx = p0_idx + 1;
    if (p1_idx >= num_points) p1_idx = num_points - 1;

    float t0 = cp_times[p0_idx];
    float t1 = cp_times[p1_idx];
    float dt = t1 - t0;

    // Pointers to data
    // Layout assumption: point i stores [state (dim), velocity (dim)]
    const float* p0_state = &control_points[p0_idx * 2 * dim];
    const float* p0_vel = &control_points[p0_idx * 2 * dim + dim];
    const float* p1_state = &control_points[p1_idx * 2 * dim];
    const float* p1_vel = &control_points[p1_idx * 2 * dim + dim];

    if (dt < 1e-6f) {
        for (int i = 0; i < dim; i++) {
            output[idx * dim + i] = p0_state[i];
        }
        return;
    }

    float u = (t - t0) / dt;
    float h00, h10, h01, h11;
    cubic_hermite(u, &h00, &h10, &h01, &h11);

    float correction = 0.0f;
    if (abs(curvature) > 1e-6f) {
        correction = u * (1.0f - u) * curvature;
    }

    for (int i = 0; i < dim; i++) {
        float m0 = p0_vel[i] * dt;
        float m1 = p1_vel[i] * dt;
        
        float val = p0_state[i] * h00 + m0 * h10 + p1_state[i] * h01 + m1 * h11;
        
        // Apply correction
        val *= (1.0f + correction);
        
        output[idx * dim + i] = val;
    }
}

void launch_spline_reconstruct(
    const float* control_points,
    const float* cp_times,
    int num_points,
    int dim,
    const float* target_times,
    float* output,
    int batch_size,
    float curvature,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    spline_reconstruct_kernel<<<grid_size, block_size, 0, stream>>>(
        control_points,
        cp_times,
        num_points,
        dim,
        target_times,
        output,
        batch_size,
        curvature
    );
}

}

