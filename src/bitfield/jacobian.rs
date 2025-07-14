//! # 비트 야코비안 모듈
//!
//! 비트 연산을 미분 가능한 대각 행렬 연산으로 해석하여
//! 그래디언트 기반 최적화를 가능하게 합니다.
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// CUDA FFI 바인딩
#[cfg(feature = "cuda")]
extern "C" {
    fn launch_bit_jacobian_kernel(
        codes: *const u32,
        x_values: *const f32,
        jacobian_diag: *mut f32,
        diff_order: u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_bit_mask_diagonal_kernel(
        mask: u32,
        diagonal: *mut f32,
        size: i32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_jacobian_transpose_kernel(
        grad_output: *const f32,
        jacobian_diag: *const f32,
        grad_input: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_hyperbolic_jacobian_kernel(
        func_types: *const u8,
        x_values: *const f32,
        jacobian_values: *mut f32,
        diff_order: u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
}
/// 비트 야코비안 계산을 위한 구조체
///
/// 비트 마스킹과 함수 선택을 대각 야코비안 행렬로 변환합니다.
pub struct BitJacobian {
    /// 2비트 순환 코드 → 야코비안 대각 원소
    /// [함수 인덱스][미분 차수] → 정규화된 도함수 값
    pub(crate) cyclic_derivatives: [[i8; 4]; 4],
    /// 스케일 팩터 (i8 → f32 변환용)
    pub(crate) scale_factor: f32,
    /// CUDA 사용 여부
    #[cfg(feature = "cuda")]
    pub use_cuda: bool,
}

impl BitJacobian {
    /// 새로운 BitJacobian 인스턴스 생성
    pub fn new() -> Self {
        // 2비트 순환 함수의 도함수 테이블
        // 00: sin(x)  → cos(x)   → -sin(x)  → -cos(x)
        // 01: cos(x)  → -sin(x)  → -cos(x)  → sin(x)
        // 10: -sin(x) → -cos(x)  → sin(x)   → cos(x)
        // 11: -cos(x) → sin(x)   → cos(x)   → -sin(x)

        let cyclic_derivatives = [
            [0, 127, 0, -127], // sin → cos → -sin → -cos
            [127, 0, -127, 0], // cos → -sin → -cos → sin
            [0, -127, 0, 127], // -sin → -cos → sin → cos
            [-127, 0, 127, 0], // -cos → sin → cos → -sin
        ];

        Self {
            cyclic_derivatives,
            scale_factor: 1.0 / 127.0,
            #[cfg(feature = "cuda")]
            use_cuda: false,
        }
    }

    /// CUDA 사용 설정
    #[cfg(feature = "cuda")]
    pub fn set_use_cuda(&mut self, use_cuda: bool) {
        self.use_cuda = use_cuda;
    }

    /// 비트필드 코드에서 함수 인덱스와 미분 차수 추출
    #[inline]
    fn decode_function_info(code: u32) -> (u8, u8) {
        // SUB 필드 (2비트): 함수 선택
        let func_idx = ((code >> 18) & 0x3) as u8;
        // D 필드 (1비트) + 내부 카운터로 미분 차수 결정
        let diff_order = ((code >> 8) & 0x1) as u8;
        (func_idx, diff_order)
    }

    /// 단일 비트필드 코드의 야코비안 대각 원소 계산
    pub fn compute_diagonal_element(&self, code: u32, x_value: f32, diff_order: u8) -> f32 {
        let (func_idx, base_diff) = Self::decode_function_info(code);

        // 전체 미분 차수 = 기본 차수 + 요청된 차수
        let total_diff = (base_diff + diff_order) % 4;

        // 위상 정보 반영
        let phase = ((code >> 24) & 0xFF) as f32 / 255.0 * 2.0 * std::f32::consts::PI;
        let x_with_phase = x_value + phase;

        // func_idx와 total_diff에 따라 실제 함수값 계산
        // func_idx: 0=sin, 1=cos, 2=-sin, 3=-cos
        // total_diff에 따라 미분 적용
        let func_value = match (func_idx, total_diff) {
            // sin 계열
            (0, 0) => x_with_phase.sin(),  // sin
            (0, 1) => x_with_phase.cos(),  // sin' = cos
            (0, 2) => -x_with_phase.sin(), // sin'' = -sin
            (0, 3) => -x_with_phase.cos(), // sin''' = -cos

            // cos 계열
            (1, 0) => x_with_phase.cos(),  // cos
            (1, 1) => -x_with_phase.sin(), // cos' = -sin
            (1, 2) => -x_with_phase.cos(), // cos'' = -cos
            (1, 3) => x_with_phase.sin(),  // cos''' = sin

            // -sin 계열
            (2, 0) => -x_with_phase.sin(), // -sin
            (2, 1) => -x_with_phase.cos(), // (-sin)' = -cos
            (2, 2) => x_with_phase.sin(),  // (-sin)'' = sin
            (2, 3) => x_with_phase.cos(),  // (-sin)''' = cos

            // -cos 계열
            (3, 0) => -x_with_phase.cos(), // -cos
            (3, 1) => x_with_phase.sin(),  // (-cos)' = sin
            (3, 2) => x_with_phase.cos(),  // (-cos)'' = cos
            (3, 3) => -x_with_phase.sin(), // (-cos)''' = -sin

            _ => 0.0,
        };

        // 진폭 적용
        let amp = ((code & 0xFF) as f32) / 255.0;
        let amp_fine = ((code >> 22) & 0x3) as f32 / 4.0;
        let total_amp = amp + amp_fine / 4.0;

        // 최종 야코비안 값
        func_value * total_amp
    }
    /// 비트필드 코드 배열의 전체 야코비안 대각 행렬 계산
    pub fn compute_jacobian(
        &self,
        codes: &Array1<u32>,
        x: &ArrayView1<f32>,
        diff_order: u8,
    ) -> Array1<f32> {
        let n = codes.len();
        assert_eq!(x.len(), n, "코드와 입력 차원이 일치해야 합니다");

        #[cfg(feature = "cuda")]
        if self.use_cuda {
            // GPU 경로
            let mut jacobian_diag = Array1::<f32>::zeros(n);
            unsafe {
                launch_bit_jacobian_kernel(
                    codes.as_ptr(),
                    x.as_ptr(),
                    jacobian_diag.as_mut_ptr(),
                    diff_order,
                    n as i32,
                    std::ptr::null_mut(), // stream (기본 스트림 사용)
                );
            }
            return jacobian_diag;
        }

        // CPU 경로
        let mut jacobian_diag = Array1::<f32>::zeros(n);
        for i in 0..n {
            jacobian_diag[i] = self.compute_diagonal_element(codes[i], x[i], diff_order);
        }

        jacobian_diag
    }

    /// 비트 마스킹을 대각 행렬로 변환
    pub fn bit_mask_to_diagonal(mask: u32, size: usize) -> Array1<f32> {
        let mut diag = Array1::<f32>::zeros(size);

        for i in 0..size.min(32) {
            if (mask >> i) & 1 == 1 {
                diag[i] = 1.0;
            }
        }

        diag
    }

    /// 역전파를 위한 야코비안 전치 적용
    pub fn apply_jacobian_transpose(
        &self,
        grad_output: &ArrayView1<f32>,
        codes: &Array1<u32>,
        x: &ArrayView1<f32>,
    ) -> Array1<f32> {
        // 야코비안이 대각 행렬이므로 전치해도 동일
        let jacobian_diag = self.compute_jacobian(codes, x, 0);

        // grad_input = J^T @ grad_output = diag(J) * grad_output
        &jacobian_diag * grad_output
    }
}

/// 쌍곡 함수용 비트 야코비안
pub struct HyperbolicBitJacobian {
    /// sinh/cosh의 2-주기 순환 테이블
    pub(crate) hyperbolic_table: [[i8; 2]; 2],
    pub(crate) scale_factor: f32,
}

impl HyperbolicBitJacobian {
    pub fn new() -> Self {
        // 쌍곡 함수의 미분 순환성
        // sinh'(x) = cosh(x), cosh'(x) = sinh(x)
        let hyperbolic_table = [
            [0, 127], // sinh → cosh
            [127, 0], // cosh → sinh
        ];

        Self {
            hyperbolic_table,
            scale_factor: 1.0 / 127.0,
        }
    }

    pub fn compute_hyperbolic_jacobian(&self, func_type: u8, x: f32, diff_order: u8) -> f32 {
        // 2-주기 순환
        let cycle_idx = diff_order % 2;

        // 실제 함수값 계산
        let func_value = match (func_type, cycle_idx) {
            (0, 0) => x.sinh(), // sinh
            (0, 1) => x.cosh(), // sinh'
            (1, 0) => x.cosh(), // cosh
            (1, 1) => x.sinh(), // cosh'
            _ => 0.0,
        };

        func_value
    }
}
