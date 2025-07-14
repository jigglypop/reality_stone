//! CUDA 리만 기하학 함수 구현
//!
//! GPU에서 실행되는 64개의 리만 기하학 함수들

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// 디바이스 함수: Bessel J0 근사
__device__ inline float bessel_j0_approx(float x) {
    if (fabsf(x) < 1e-6f) {
        return 1.0f;
    } else {
        float x2 = x * x;
        float x4 = x2 * x2;
        float x6 = x4 * x2;
        float x8 = x4 * x4;
        
        return 1.0f - x2 / 4.0f + x4 / 64.0f - x6 / 2304.0f + x8 / 147456.0f;
    }
}

// 디바이스 함수: Bessel J1 근사
__device__ inline float bessel_j1_approx(float x) {
    if (fabsf(x) < 1e-6f) {
        return x / 2.0f;
    } else {
        float x2 = x * x;
        float x3 = x2 * x;
        float x5 = x3 * x2;
        float x7 = x5 * x2;
        
        return x / 2.0f - x3 / 16.0f + x5 / 384.0f - x7 / 18432.0f;
    }
}

// 메인 리만 기하학 함수
__device__ float get_riemannian_function_device(uint8_t cat, uint8_t sub, uint8_t d, float r) {
    const float EPS = 1e-6f;
    
    // 비트 패킹으로 스위치 최적화
    uint16_t code = (cat << 4) | (sub << 2) | d;
    
    switch (code) {
        // ========== Category 0: Poincaré 기하학 ==========
        
        // 기본 함수족 (SUB=0)
        case 0x00: return tanhf(r) * 0.5f;                      // tanh(r)/2
        case 0x01: return -tanhf(r) * 0.5f;                     // -tanh(r)/2
        case 0x02: return 2.0f * tanhf(r * 0.25f);              // 2*tanh(r/4)
        case 0x03: { float t = tanhf(r * 0.5f); return t * t; } // tanh²(r/2)
        
        // 쌍곡 함수족 (SUB=1)
        case 0x04: return sinhf(r) / (1.0f + coshf(r));         // sinh(r)/(1+cosh(r))
        case 0x05: return (coshf(r) - 1.0f) / (1.0f + coshf(r)); // (cosh(r)-1)/(1+cosh(r))
        case 0x06: return tanhf(r);                             // tanh(r)
        case 0x07: return (fabsf(r) < EPS) ? 1.0f : sinhf(r) / r; // sinh(r)/r
        
        // 삼각 함수족 (SUB=2)
        case 0x08: return (fabsf(r) < EPS) ? 1.0f : sinf(r) / r; // sin(r)/r
        case 0x09: return cosf(r);                               // cos(r)
        case 0x0A: return (fabsf(r) < EPS) ? 0.5f : (1.0f - cosf(r)) / (r * r); // (1-cos(r))/r²
        case 0x0B: return sinf(r) * cosf(r);                     // sin(r)cos(r)
        
        // 지수/로그 함수족 (SUB=3)
        case 0x0C: return (fabsf(r) < EPS) ? 1.0f : (expf(r) - 1.0f) / r; // (e^r-1)/r
        case 0x0D: return expf(r) / (1.0f + expf(r));            // e^r/(1+e^r)
        case 0x0E: return (r > 0.0f) ? logf(r + 1.0f) / r : 1.0f; // ln(r+1)/r
        case 0x0F: return r / (1.0f + fabsf(r));                 // r/(1+|r|)
        
        // ========== Category 1: Lorentz 기하학 ==========
        
        // 기본 Lorentz (SUB=0)
        case 0x10: return sinhf(r);                              // sinh(r)
        case 0x11: return coshf(r) - 1.0f;                       // cosh(r)-1
        case 0x12: return tanhf(r);                              // tanh(r)
        case 0x13: return sinhf(r) / coshf(r);                   // sinh(r)/cosh(r)
        
        // 수정된 쌍곡 (SUB=1)
        case 0x14: return 2.0f * sinhf(r * 0.5f);                // 2*sinh(r/2)
        case 0x15: return coshf(r * 0.5f);                       // cosh(r/2)
        case 0x16: return expf(r) - 1.0f;                        // e^r-1
        case 0x17: return 1.0f - expf(-r);                       // 1-e^(-r)
        
        // 확장된 쌍곡 (SUB=2)
        case 0x18: return r * coshf(r);                          // r*cosh(r)
        case 0x19: return r * sinhf(r);                          // r*sinh(r)
        case 0x1A: return (fabsf(r) < EPS) ? 0.0f : (sinhf(r) - r) / (r * r); // (sinh(r)-r)/r²
        case 0x1B: return (fabsf(r) < EPS) ? 0.0f : (coshf(r) - 1.0f - r * r * 0.5f) / (r * r * r); // (cosh(r)-1-r²/2)/r³
        
        // 특수 Lorentz (SUB=3)
        case 0x1C: return sqrtf(coshf(r));                       // √cosh(r)
        case 0x1D: return (r >= 0.0f ? 1.0f : -1.0f) * sqrtf(fabsf(sinhf(r))); // sgn(r)√|sinh(r)|
        case 0x1E: return tanhf(r * r);                          // tanh(r²)
        case 0x1F: return (fabsf(r) < EPS) ? 1.0f : tanhf(r) / r; // tanh(r)/r
        
        // ========== Category 2: Klein 기하학 ==========
        
        // 기본 Klein (SUB=0)
        case 0x20: return r / (1.0f + r);                        // r/(1+r)
        case 0x21: return r / sqrtf(1.0f + r * r);               // r/√(1+r²)
        case 0x22: return r * r / (1.0f + r * r);                // r²/(1+r²)
        case 0x23: return 1.0f - 1.0f / (1.0f + r);              // 1-1/(1+r)
        
        // 투영 함수 (SUB=1)
        case 0x24: return 2.0f * r / (1.0f + r * r);             // 2r/(1+r²)
        case 0x25: return (1.0f - r * r) / (1.0f + r * r);       // (1-r²)/(1+r²)
        case 0x26: { float d = 1.0f + r * r; return 4.0f * r / (d * d); } // 4r/(1+r²)²
        case 0x27: return 2.0f * atanf(r) / 3.14159265359f;      // 2*atan(r)/π
        
        // 멱급수 근사 (SUB=2)
        case 0x28: return r / (1.0f + r + r * r * 0.5f);         // r/(1+r+r²/2)
        case 0x29: return r / (1.0f + r - r * r * 0.5f);         // r/(1+r-r²/2)
        case 0x2A: return r - r * r * r / 3.0f + r * r * r * r * r / 5.0f; // r-r³/3+r⁵/5
        case 0x2B: return r / (1.0f + r * r / 3.0f);             // r/(1+r²/3)
        
        // 변분 함수 (SUB=3)
        case 0x2C: return r / (1.0f + fabsf(r));                 // r/(1+|r|)
        case 0x2D: return (r >= 0.0f ? 1.0f : -1.0f) * sqrtf(fabsf(r)); // sgn(r)√|r|
        case 0x2E: return r * r * r / (1.0f + r * r);            // r³/(1+r²)
        case 0x2F: return r * expf(-r * r * 0.5f);               // r*exp(-r²/2)
        
        // ========== Category 3: 특수 함수 ==========
        
        // Bessel 유사 함수 (SUB=0)
        case 0x30: return 0.5f * (bessel_j0_approx(r) + bessel_j1_approx(r)); // (J₀(r)+J₁(r))/2
        case 0x31: return (fabsf(r) < EPS) ? 1.0f : (sinf(r) / r) * cosf(r * 0.5f); // sinc(r)*cos(r/2)
        case 0x32: return (fabsf(r) < EPS) ? 0.0f : 2.0f * bessel_j1_approx(r) / r; // 2J₁(r)/r
        case 0x33: return bessel_j0_approx(r) * cosf(r);         // J₀(r)*cos(r)
        
        // Gaussian 유사 함수 (SUB=1)
        case 0x34: return expf(-r * r * 0.5f);                   // exp(-r²/2)
        case 0x35: return r * expf(-r * r * 0.5f);               // r*exp(-r²/2)
        case 0x36: return (1.0f - r * r) * expf(-r * r * 0.5f);  // (1-r²)*exp(-r²/2)
        case 0x37: return 1.0f / sqrtf(1.0f + r * r);            // 1/√(1+r²)
        
        // 주기적 변조 (SUB=2)
        case 0x38: return cosf(r) * expf(-r * r * 0.25f);        // cos(r)*exp(-r²/4)
        case 0x39: return sinf(r) * expf(-r * r * 0.25f);        // sin(r)*exp(-r²/4)
        case 0x3A: return cosf(r * r);                           // cos(r²)
        case 0x3B: return sinf(r * r);                           // sin(r²)
        
        // 실험적 함수 (SUB=3)
        case 0x3C: return (fabsf(r) < EPS) ? 1.0f : tanf(r) / r; // tan(r)/r
        case 0x3D: return (fabsf(r) < EPS) ? 0.0f : sinf(r) * cosf(r) / r; // sin(r)cos(r)/r
        case 0x3E: return (fabsf(r) < EPS || fabsf(sinf(r)) < EPS) ? 0.5f : (1.0f - cosf(r)) / (r * sinf(r)); // (1-cos(r))/(r*sin(r))
        case 0x3F: return (r > 0.0f) ? logf(1.0f + r * r) / r : 2.0f; // ln(1+r²)/r
        
        // 기본값
        default: return tanhf(r) * 0.5f;
    }
}

// 호스트 인터페이스 (테스트용)
extern "C" {
    __global__ void test_riemannian_kernel(
        const uint8_t* cats,
        const uint8_t* subs,
        const uint8_t* ds,
        const float* rs,
        float* outputs,
        int n
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;
        
        outputs[tid] = get_riemannian_function_device(
            cats[tid], subs[tid], ds[tid], rs[tid]
        );
    }
} 