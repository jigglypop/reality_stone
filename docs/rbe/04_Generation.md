# 4. 가중치 생성 (`src/generation.rs`)

이 문서에서는 디코딩된 파라미터(`DecodedParams`)를 사용하여, 행렬의 특정 위치 `(i, j)`에 해당하는 단일 가중치(`f32`) 값을 계산하는 `compute_weight` 함수의 핵심 로직을 설명합니다.

---

### `Packed64::compute_weight()`

```rust
impl Packed64 {
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // ... (구현)
    }
}
```

-   **역할**: `(i, j)` 좌표와 `DecodedParams`를 조합하여 최종 가중치 값을 생성합니다. 이 함수가 레이어의 복잡한 패턴을 만들어내는 심장부입니다.
-   **핵심 과정**:
    1.  **시드 디코딩**: 먼저 `self.decode()`를 호출하여 `DecodedParams`를 얻습니다.
    2.  **좌표 정규화**: `(i, j)` 행렬 인덱스를 `[-1, 1]` 범위의 데카르트 좌표 `(x, y)`로 변환합니다.
    3.  **로컬 극좌표 계산**: 정규화된 `(x, y)`를 다시 로컬 극좌표 `(r_local, theta_local)`로 변환합니다.
    4.  **파라미터 조합**:
        -   글로벌 `theta`와 로컬 `theta_local`, 그리고 `rot_code`에 의한 회전 각도를 모두 더해 최종 각도 `theta_final`을 계산합니다.
        -   `d_theta`, `d_r` 파라미터를 이용해 미분 연산을 적용하여 `angular_value`와 `radial_value`를 계산합니다.
    5.  **기저 함수 적용**: `basis_id`에 따라 선택된 기저 함수에 파라미터들을 적용하여 `basis_value`를 계산합니다.
    6.  **최종 가중치 계산**: `basis_value`에 야코비안(Jacobian) 보정치를 곱하여 최종 가중치 값을 반환합니다.

### 가중치 생성 공식의 직관적 이해

각 가중치는 다음과 같은 요소들의 복합적인 상호작용으로 결정됩니다.

`Weight(i, j) = Jacobian(c, r) * BasisFunction(Global(r, θ), Local(i, j), Derivatives, Rotation)`

-   **Global Parameters (`r`, `theta`)**: 시드에 저장된 전역 파라미터. 행렬 전체에 걸쳐 공유되는 기본 패턴(전체적인 크기, 위상)을 결정합니다.
-   **Local Coordinates (`i`, `j`)**: 각 가중치의 위치. 전역 패턴에 국소적인 변화(local variation)를 주어, 위치마다 다른 값을 갖게 합니다.
-   **Control Parameters (`basis_id`, `d_theta`, `d_r`, `rot_code`, `c`)**: 패턴의 형태를 질적으로 바꾸는 제어 파라미터입니다.
    -   `basis_id`: 사인파, 쌍곡선, 베셀 함수 등 패턴의 근본적인 모양을 결정합니다.
    -   `d_theta`, `d_r`: 미분을 통해 패턴을 더 복잡하고 날카롭게(high-frequency) 만듭니다.
    -   `rot_code`: 전체 패턴을 회전시킵니다.
    -   `c`: 공간의 곡률을 조절하여 패턴이 휘어지는 정도를 제어합니다.

이러한 계층적이고 조합적인 방식을 통해, 단 8바이트의 시드만으로도 매우 다양하고 복잡한 형태의 거대 행렬을 효율적으로 생성할 수 있습니다. 