#pragma once

// Common device helpers for Möbius operations shared by mobius.cu and poincare.cu
// We keep all math identical across kernels to avoid subtle numerical drift.

// Pointwise Möbius addition: out = u ⊕_c v for a single vector pair.
// Arguments:
// - u, v: input vectors (length dim)
// - out: output vector (length dim)
// - dim: spatial dimension
// - c: curvature parameter
// - min_den: minimum value to clamp the denominator for numerical stability
__device__ inline void mobius_add_point(
    const float* u,
    const float* v,
    float* out,
    int dim,
    float c,
    float min_den
) {
    float u2 = 0.0f;
    float v2 = 0.0f;
    float uv = 0.0f;

    for (int j = 0; j < dim; ++j) {
        float uj = u[j];
        float vj = v[j];
        u2 += uj * uj;
        v2 += vj * vj;
        uv += uj * vj;
    }

    float c2 = c * c;
    float den = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
    if (den < min_den) {
        den = min_den;
    }

    float coeff_u = (1.0f + 2.0f * c * uv + c * v2) / den;
    float coeff_v = (1.0f - c * u2) / den;

    for (int j = 0; j < dim; ++j) {
        out[j] = coeff_u * u[j] + coeff_v * v[j];
    }
}

// Pointwise Möbius scalar multiplication: out = r ⊗_c u for a single vector.
// Arguments:
// - u: input vector (length dim)
// - out: output vector (length dim)
// - dim: spatial dimension
// - c: curvature parameter
// - r: scalar multiplier
// - eps: small epsilon for handling very small norms / c ~ 0
// - boundary_eps: epsilon to keep arguments inside atanh / tanh domains
__device__ inline void mobius_scalar_point(
    const float* u,
    float* out,
    int dim,
    float c,
    float r,
    float eps,
    float boundary_eps
) {
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += u[j] * u[j];
    }

    // Very small vectors: fall back to simple scaling to keep gradients stable
    if (norm_sq < eps * eps) {
        for (int j = 0; j < dim; ++j) {
            out[j] = r * u[j];
        }
        return;
    }

    float norm = sqrtf(norm_sq);

    // c = 0: Euclidean case
    if (fabsf(c) < eps) {
        for (int j = 0; j < dim; ++j) {
            out[j] = r * u[j];
        }
        return;
    }

    float scale;
    if (c > 0.0f) {
        // Positive curvature
        float sqrt_c = sqrtf(c);
        float scn = fminf(sqrt_c * norm, 1.0f - boundary_eps);
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        scale = beta / (sqrt_c * norm);
    } else {
        // Negative curvature (compute with real-valued formula)
        float sqrt_abs_c = sqrtf(-c);
        float scn = sqrt_abs_c * norm;
        float alpha = atanf(scn);
        float beta = tanf(r * alpha);
        scale = beta / (sqrt_abs_c * norm);
    }

    for (int j = 0; j < dim; ++j) {
        out[j] = scale * u[j];
    }
}


