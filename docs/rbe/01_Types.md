# 1. 핵심 데이터 타입 (`src/types.rs`)

이 문서에서는 푸앵카레 레이어 라이브러리의 핵심을 이루는 데이터 구조체들을 설명합니다. 모든 기능은 이 타입들을 중심으로 구성됩니다.

---

### `Packed64`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64(pub u64);
```

-   **역할**: 푸앵카레 레이어 전체를 표현하는 **단일 64비트 시드(seed)** 입니다.
-   **구조**: `u64` 정수 하나를 튜플 구조체(tuple struct)로 감싸, 타입 안정성을 높이고 다른 `u64` 값과 혼동되지 않도록 합니다.
-   **핵심 아이디어**: 이 8바이트(64비트) 값 안에 행렬을 생성하는 데 필요한 모든 파라미터(반지름, 각도, 기저 함수 종류, 곡률 등)가 비트 단위로 압축되어 저장됩니다.

---

### `BasisFunction`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisFunction {
    SinCosh = 0,
    SinSinh = 1,
    // ... (기타 함수들)
}
```

-   **역할**: 가중치를 생성할 때 사용할 수학적 **기저 함수(basis function)**의 종류를 정의하는 열거형(enum)입니다.
-   **`#[repr(u8)]`**: 각 열거형 멤버가 메모리 상에서 `u8` 정수 값으로 표현되도록 지정합니다. 이를 통해 `Packed64` 시드 내의 4비트 공간과 효율적으로 매핑할 수 있습니다.
-   **종류**: 삼각함수, 쌍곡선 함수, 베셀 함수, 웨이블릿 등 다양한 함수를 조합하여 복잡한 패턴의 행렬을 생성할 수 있도록 합니다.

---

### `DecodedParams`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedParams {
    pub r: f32,           // 20 bits
    pub theta: f32,       // 24 bits
    pub basis_id: u8,     // 4 bits
    // ... (기타 파라미터들)
}
```

-   **역할**: `Packed64` 시드를 디코딩한 결과를 담는 구조체입니다.
-   **구조**: 64비트 시드에 압축되어 있던 각 필드(반지름 `r`, 각도 `theta` 등)를 사람이 읽기 쉬운 `f32`, `u8`, `bool` 등의 타입으로 변환하여 저장합니다.
-   **활용**: 가중치 생성 함수(`compute_weight`)는 이 `DecodedParams` 구조체를 인자로 받아, 각 파라미터를 조합하여 최종 가중치 값을 계산합니다.

---

### `PoincareMatrix`

```rust
pub struct PoincareMatrix {
    pub seed: Packed64,
    pub rows: usize,
    pub cols: usize,
}
```

-   **역할**: **단일 `Packed64` 시드**와 그 시드가 표현하는 행렬의 차원(`rows`, `cols`) 정보를 함께 관리하는 최상위 구조체입니다.
-   **핵심**: 이 구조체는 전체 가중치 행렬(예: 32x32 `f32` 행렬, 4KB)을 단 8바이트의 `seed`와 약간의 메타데이터로 표현합니다. 이것이 이 라이브러리의 핵심적인 압축 방식입니다.
-   **메서드**: `compress`와 `decompress` 메서드를 통해 실제 `f32` 행렬과 `PoincareMatrix` 간의 상호 변환을 수행합니다. 