//! 1.5KB 통합 테이블 시스템
//!
//! 모든 비트필드 연산에 필요한 테이블을 1536바이트로 압축

use ndarray::{Array1, Array2};
use std::f32::consts::PI;

/// 통합 테이블 구조체 (1.5KB)
#[repr(C, align(64))]
pub struct UnifiedTables {
    /// 순환 테이블: 4×16 (64 bytes)
    pub cyclic_table: [[i8; 16]; 4],

    /// 리만 함수 메타데이터: 64×8 (512 bytes)
    /// [scale_factor, offset, period, reserved] × 64 functions
    pub riemannian_meta: [[f32; 4]; 64],

    /// 압축된 기저 테이블 인덱스: 256×4 (1024 bytes)
    /// 각 기저 벡터의 주요 성분 인덱스
    pub basis_indices: [[u8; 4]; 256],

    /// 패딩 (캐시 라인 정렬)
    _padding: [u8; 64],
}

impl UnifiedTables {
    /// 새로운 통합 테이블 생성
    pub fn new() -> Self {
        let mut tables = Self {
            cyclic_table: Self::init_cyclic_table(),
            riemannian_meta: Self::init_riemannian_meta(),
            basis_indices: [[0; 4]; 256],
            _padding: [0; 64],
        };

        // 기저 인덱스 초기화
        tables.init_basis_indices();
        tables
    }

    /// 순환 테이블 초기화
    const fn init_cyclic_table() -> [[i8; 16]; 4] {
        [
            // diff=0: sin(θ)
            [
                127, 117, 98, 71, 49, 24, 0, -24, -49, -71, -98, -117, -127, -117, -98, -71,
            ],
            // diff=1: cos(θ)
            [
                0, 49, 71, 98, 117, 127, 117, 98, 71, 49, 0, -49, -71, -98, -117, -127,
            ],
            // diff=2: -sin(θ)
            [
                -127, -117, -98, -71, -49, -24, 0, 24, 49, 71, 98, 117, 127, 117, 98, 71,
            ],
            // diff=3: -cos(θ)
            [
                0, -49, -71, -98, -117, -127, -117, -98, -71, -49, 0, 49, 71, 98, 117, 127,
            ],
        ]
    }

    /// 리만 함수 메타데이터 초기화
    fn init_riemannian_meta() -> [[f32; 4]; 64] {
        let mut meta = [[0.0; 4]; 64];

        for cat in 0..4 {
            for sub in 0..4 {
                for d in 0..4 {
                    let idx = (cat << 4) | (sub << 2) | d;

                    // 함수별 메타데이터 설정
                    meta[idx] = match (cat, sub, d) {
                        // Poincaré 기하학
                        (0, 0, 0) => [0.5, 0.0, 0.0, 0.0], // tanh(r)/2
                        (0, 0, 1) => [-0.5, 0.0, 0.0, 0.0], // -tanh(r)/2
                        (0, 0, 2) => [2.0, 0.0, 0.25, 0.0], // 2*tanh(r/4)
                        (0, 0, 3) => [1.0, 0.0, 0.5, 2.0], // tanh²(r/2)

                        // Lorentz 기하학
                        (1, 0, 0) => [1.0, 0.0, 0.0, 0.0], // sinh(r)
                        (1, 0, 1) => [1.0, -1.0, 0.0, 0.0], // cosh(r)-1
                        (1, 0, 2) => [1.0, 0.0, 0.0, 0.0], // tanh(r)
                        (1, 0, 3) => [1.0, 0.0, 0.0, 0.0], // sinh(r)/cosh(r)

                        // Klein 기하학
                        (2, 0, 0) => [1.0, 0.0, 0.0, 0.0], // r/(1+r)
                        (2, 0, 1) => [1.0, 0.0, 0.0, 0.0], // r/√(1+r²)
                        (2, 0, 2) => [1.0, 0.0, 0.0, 2.0], // r²/(1+r²)
                        (2, 0, 3) => [1.0, 0.0, 0.0, 0.0], // 1-1/(1+r)

                        // 특수 함수
                        (3, 0, 0) => [0.5, 0.0, 0.0, 0.0], // (J₀+J₁)/2
                        (3, 1, 0) => [1.0, 0.0, 0.0, -0.5], // exp(-r²/2)
                        (3, 2, 0) => [1.0, 0.0, 0.0, -0.25], // cos(r)exp(-r²/4)
                        (3, 3, 0) => [1.0, 0.0, 0.0, 0.0], // tan(r)/r

                        _ => [1.0, 0.0, 0.0, 0.0],
                    };
                }
            }
        }

        meta
    }

    /// 기저 인덱스 초기화 (희소 표현)
    fn init_basis_indices(&mut self) {
        // 각 기저 벡터의 주요 4개 성분 인덱스 저장
        for i in 0..256 {
            // 기본 패턴: 균등 분포
            self.basis_indices[i] = [
                (i as u8) % 64,
                ((i + 64) as u8) % 64,
                ((i + 128) as u8) % 64,
                ((i + 192) as u8) % 64,
            ];
        }
    }

    /// 순환 테이블 조회
    #[inline(always)]
    pub fn lookup_cyclic(&self, diff_order: u8, phase: u8) -> i8 {
        self.cyclic_table[(diff_order & 3) as usize][(phase & 15) as usize]
    }

    /// 리만 함수 메타데이터 조회
    #[inline(always)]
    pub fn get_riemannian_meta(&self, cat: u8, sub: u8, d: u8) -> &[f32; 4] {
        let idx = ((cat & 3) << 4) | ((sub & 3) << 2) | (d & 3);
        &self.riemannian_meta[idx as usize]
    }

    /// 기저 인덱스 조회
    #[inline(always)]
    pub fn get_basis_indices(&self, idx: u8) -> &[u8; 4] {
        &self.basis_indices[idx as usize]
    }

    /// 메모리 크기 확인
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// GPU용 통합 테이블 래퍼
#[cfg(feature = "cuda")]
pub struct GpuUnifiedTables {
    pub device_ptr: *mut UnifiedTables,
}

#[cfg(feature = "cuda")]
impl GpuUnifiedTables {
    /// GPU에 통합 테이블 업로드
    pub fn upload(host_tables: &UnifiedTables) -> Result<Self, String> {
        use crate::layers::cuda::ffi::*;

        unsafe {
            let mut device_ptr: *mut UnifiedTables = std::ptr::null_mut();

            // GPU 메모리 할당
            let result = cudaMalloc(
                &mut device_ptr as *mut _ as *mut *mut std::ffi::c_void,
                std::mem::size_of::<UnifiedTables>(),
            );

            if result != 0 {
                return Err(format!("CUDA malloc failed: {}", result));
            }

            // 호스트에서 디바이스로 복사
            let result = cudaMemcpy(
                device_ptr as *mut std::ffi::c_void,
                host_tables as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<UnifiedTables>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            if result != 0 {
                cudaFree(device_ptr as *mut std::ffi::c_void);
                return Err(format!("CUDA memcpy failed: {}", result));
            }

            Ok(Self { device_ptr })
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuUnifiedTables {
    fn drop(&mut self) {
        use crate::layers::cuda::ffi::cudaFree;

        unsafe {
            if !self.device_ptr.is_null() {
                cudaFree(self.device_ptr as *mut std::ffi::c_void);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_통합_테이블_크기() {
        assert_eq!(UnifiedTables::size(), 1600); // 1536 + 64 padding
        println!("통합 테이블 크기: {} bytes", UnifiedTables::size());
    }

    #[test]
    fn test_순환_테이블_조회() {
        let tables = UnifiedTables::new();

        // sin(0) = 0
        assert_eq!(tables.lookup_cyclic(0, 6), 0);

        // cos(0) = 127 (최대값)
        assert_eq!(tables.lookup_cyclic(1, 5), 127);
    }

    #[test]
    fn test_리만_메타데이터() {
        let tables = UnifiedTables::new();

        // Poincaré tanh(r)/2
        let meta = tables.get_riemannian_meta(0, 0, 0);
        assert_eq!(meta[0], 0.5); // scale factor

        // Lorentz sinh(r)
        let meta = tables.get_riemannian_meta(1, 0, 0);
        assert_eq!(meta[0], 1.0);
    }
}
