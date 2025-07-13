# 6. 비트필드 직접 추론 기능 구현 가이드

## 6.1 목표 및 개요

이 문서는 "리만 기하학 기반 비트필드 직접 추론(Riemannian Bitfield Direct Inference)" 모듈을 `reality_stone` Rust 프로젝트에 통합하기 위한 상세 구현 계획을 기술한다.

-   **최종 목표**: 기존의 `nn.Linear`와 유사한 역할을 하는 `BitfieldLinear` 레이어를 구현한다. 이 레이어는 가중치를 FP32 행렬로 저장하는 대신, 22비트 코드로 압축된 상태로 보관하며, 추론 시 완전한 행렬 복원 없이 압축 상태에서 직접 행렬 곱셈을 수행한다.
-   **기대 효과**:
    -   **모델 크기**: 수십~수백 배 감소.
    -   **추론 속도**: 3~15배 가속 (특히 메모리 대역폭 병목 해소).
-   **핵심 기술**:
    -   가중치 행 벡터를 푸앵카레 볼 위의 점 $(r, u)$로 표현.
    -   $(r, u)$를 이산적인 비트필드 `[cat|sub|idx|d|amp]`로 양자화 및 인코딩.
    -   내적의 선형성($x \cdot w = s(r) \cdot (x \cdot u)$)을 이용한 직접 추론 커널.

## 6.2 신규 파일 구조

`src/` 디렉토리 내에 다음 모듈들을 추가한다.

```
src/
├── bitfield.rs       # 비트 마스크 상수 및 인코딩/디코딩 유틸리티
├── basis.rs          # 단위 기저 벡터 테이블(B×n) 관리 모듈
├── hyper_ops.rs      # tanh, sin, cos 등 주기 함수 LUT
├── kernel.rs         # CPU 및 CUDA 추론 커널
├── lib.rs            # (수정) 새 모듈들을 통합하고 공개 API 노출
└── main.rs           # (수정 또는 신규) 데모 실행 파일
```

## 6.3 단계별 구현 계획

### 1단계: 비트필드 모듈 (`src/bitfield.rs`)

-   **역할**: 22비트 코드의 각 필드를 추출하기 위한 마스크 상수와 인라인 함수를 정의한다.
-   **구현 내용**:

```rust
// src/bitfield.rs
pub const CAT_MASK: u32 = 0b11 << 20;
pub const SUB_MASK: u32 = 0b11 << 18;
pub const IDX_MASK: u32 = 0xFF << 10;
pub const D_MASK:   u32 = 0b11 << 8;
pub const AMP_MASK: u32 = 0xFF;

#[inline(always)]
pub fn decode(code: u32) -> (u8, u8, u8, u8, u8) {
    let cat = ((code & CAT_MASK) >> 20) as u8;
    let sub = ((code & SUB_MASK) >> 18) as u8;
    let idx = ((code & IDX_MASK) >> 10) as u8;
    let d   = ((code & D_MASK) >> 8) as u8;
    let amp = (code & AMP_MASK) as u8;
    (cat, sub, idx, d, amp)
}
```

### 2단계: 기저 테이블 모듈 (`src/basis.rs`)

-   **역할**: 추론에 사용될 공유 단위 기저 벡터 테이블 $\{b_j\}$를 생성하고 관리한다.
-   **구현 내용**:

```rust
// src/basis.rs
use tch::{Tensor, Kind, Device};

/// B×n 크기의 단위 기저 벡터 테이블을 생성한다.
/// 실제로는 학습된 테이블을 파일에서 로드해야 하지만, 여기서는 랜덤으로 생성한다.
pub fn load_basis_table(b: i64, n: i64, device: Device) -> Tensor {
    let t = Tensor::randn(&[b, n], (Kind::Float, device));
    // 각 행을 L2-정규화하여 단위 벡터로 만든다.
    let t_norm = t.norm_scalaropt(2, &[1], true, Kind::Float);
    let basis = t / t_norm;
    // VRAM 절약을 위해 FP16으로 저장
    basis.to_kind(Kind::Half)
}
```

### 3단계: 하이퍼볼릭 연산 LUT (`src/hyper_ops.rs`)

-   **역할**: 비트필드의 `(cat, sub, d)` 코드에 따라 적절한 C∞ 기저 함수(또는 그 도함수)를 반환하는 조회 테이블(LUT) 역할을 한다.
-   **구현 내용**:

```rust
// src/hyper_ops.rs
use tch::Tensor;

/// (cat, sub, d) 코드와 반지름(r) 텐서를 받아 스케일링 함수 값을 반환한다.
pub fn lookup_and_apply(cat: u8, sub: u8, d: u8, r: &Tensor) -> Tensor {
    match (cat, sub) {
        // Poincaré Ball, 기본 스케일 함수
        (0, 0) => (0.5 * r).tanh(),
        // Poincaré Ball, C∞ 주기 함수
        (0, 1) => { // sinh/cosh family, period 2
            if d % 2 == 0 { r.sinh() } else { r.cosh() }
        },
        (0, 2) => { // sin/cos family, period 4
            match d % 4 {
                0 => r.sin(),
                1 => r.cos(),
                2 => -r.sin(),
                _ => -r.cos(),
            }
        },
        _ => unimplemented!("미구현 카테고리/서브루틴: ({}, {})", cat, sub),
    }
}
```

### 4단계: 직접 추론 커널 (`src/kernel.rs`)

-   **역할**: 핵심적인 "압축 상태 추론" 로직을 수행한다. CPU 참조 구현과 고성능 CUDA 구현 가이드를 포함한다.
-   **구현 내용 (CPU 참조)**:

```rust
// src/kernel.rs
use tch::{Tensor, Kind, Device, no_grad};
use crate::{bitfield, hyper_ops};

/// x: [batch, n], codes: [m], basis_table: [B, n] (FP16), delta: f32
pub fn gemm_hyper_bit_cpu(
    x: &Tensor,
    codes: &Tensor,
    basis_table: &Tensor,
    delta: f32,
) -> Tensor {
    let (batch_size, n) = (x.size()[0], x.size()[1]);
    let m = codes.size()[0];

    // 1. 전처리: dotB[k] = x · b_k => 결과: [batch_size, B]
    let basis_fp32 = basis_table.to_kind(Kind::Float);
    let dotb = x.matmul(&basis_fp32.tr());

    // 2. 메인 루프: m개의 행에 대해 스칼라 곱셈 수행
    let mut output_cols = Vec::with_capacity(m as usize);
    let codes_vec: Vec<u32> = codes.try_into().unwrap();

    no_grad(|| {
        for &code in &codes_vec {
            let (cat, sub, idx, d, amp) = bitfield::decode(code);
            let r_val = (amp as f32) * delta;
            let r_tensor = Tensor::from(r_val).to_device(x.device());

            // 3a. 스케일 계산: s_i = f(r_i)
            let scale_factor = hyper_ops::lookup_and_apply(cat, sub, d, &r_tensor);

            // 3b. y_i = s_i * (x · b_{idx_i})
            let dot_col = dotb.select(1, idx as i64); // [batch_size]
            let y_col = dot_col * scale_factor; // [batch_size]
            output_cols.push(y_col);
        }
    });

    Tensor::stack(&output_cols, 1) // [batch_size, m]
}
```
-   **CUDA 구현 가이드**:
    -   `dotB` 계산은 `cuBLAS`의 `gemm`을 활용한다.
    -   메인 루프는 커스텀 CUDA 커널로 작성한다.
    -   `dotB` 테이블을 **공유 메모리(shared memory)** 에 올려두어 L2 캐시보다 빠른 접근을 보장한다.
    -   각 스레드는 여러 `m` 인덱스를 담당하며, `codes` 버퍼에서 자신의 코드를 읽어온다.
    -   비트 언패킹(`&`, `>>`), 스케일 계산(`__tanhf`, `__expf`), 최종 곱셈을 모두 레지스터 상에서 수행하여 메모리 접근을 최소화한다.

### 5단계: 공개 API 통합 (`src/lib.rs` 수정)

-   **역할**: 구현된 모듈들을 사용하여 `BitfieldLinear`라는 사용자 친화적인 구조체를 만들고, 이를 라이브러리의 공개 API로 노출한다. (PyO3 바인딩은 추후 추가)

```rust
// src/lib.rs 에 추가될 내용
mod bitfield;
mod basis;
mod hyper_ops;
mod kernel;

use tch::{Tensor, Device};

pub struct BitfieldLinear {
    codes: Tensor,
    basis_table: Tensor,
    delta: f32,
    m: i64,
    n: i64,
}

impl BitfieldLinear {
    pub fn new(m: i64, n: i64, b: i64, r_max: f32, device: Device) -> Self {
        // 실제로는 학습된 codes와 basis_table을 파일에서 로드
        let codes = Tensor::randint(u32::MAX as i64, &[m], (Kind::Int64, device)).to_kind(Kind::Int);
        let basis_table = basis::load_basis_table(b, n, device);
        Self {
            codes,
            basis_table,
            delta: r_max / 255.0,
            m,
            n,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.size()[1], self.n, "Input dimension mismatch");
        kernel::gemm_hyper_bit_cpu(x, &self.codes, &self.basis_table, self.delta)
    }
}
```

### 6.4 학습 고려사항

이 구현은 **추론 전용**이다. 이 레이어를 학습시키려면 다음이 추가로 필요하다.
-   **방향(idx) 학습**: Gumbel-Softmax나 Straight-Through Estimator(STE)를 사용하여 이산적인 `idx` 선택에 대한 그래디언트를 근사해야 한다.
-   **크기(amp) 학습**: 양자화 과정에 대한 STE가 필요하다.
-   **정렬 손실(Alignment Loss)**: `u`가 기저 벡터 `b_idx`에 잘 정렬되도록 하는 추가적인 손실 항을 학습에 포함해야 한다.

이 가이드를 통해 `reality_stone` 프로젝트에 혁신적인 비트필드 직접 추론 기능을 체계적으로 탑재할 수 있을 것입니다. 