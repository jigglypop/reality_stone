# 6. 행렬 압축 및 복원 (`src/matrix.rs`)

이 문서에서는 `PoincareMatrix` 구조체에 구현된 `compress`와 `decompress` 메서드의 작동 방식을 설명합니다. 이 두 함수는 라이브러리의 핵심적인 입출력 기능을 담당합니다.

---

### `PoincareMatrix::compress()`

```rust
impl PoincareMatrix {
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        // ... (구현)
    }
}
```

-   **역할**: 주어진 `f32` 행렬을 분석하여, 그 패턴을 가장 잘 표현하는 **최적의 `Packed64` 시드 하나**를 찾아냅니다. 이것이 '압축' 과정입니다.
-   **핵심 과정 (현재 구현)**:
    1.  **패턴 분석 (Heuristic)**:
        -   행렬 전체를 스캔하여 절대값이 가장 큰 요소의 위치 `(max_i, max_j)`를 찾습니다.
        -   이 위치를 기반으로 행렬의 주요 패턴이 가질 것으로 예상되는 `r`과 `theta`의 초기 추정치를 계산합니다.
    2.  **최적화 (Random Search)**:
        -   다양한 파라미터(`r`, `theta`, `basis_id` 등)를 무작위로 조합하여 1000개의 테스트 시드를 생성합니다.
        -   각 테스트 시드로부터 행렬의 일부(10x10)를 복원하고, 원본과의 오차(MSE)를 계산합니다.
        -   1000번의 시도 중 오차가 가장 작았던 **최고의 시드(`best_seed`)**를 최종 결과로 선택합니다.
-   **한계 및 개선점**: 현재의 랜덤 탐색 방식은 간단하지만, 항상 최적의 해를 찾는다는 보장은 없습니다. 더 정교한 압축을 위해서는 푸리에 변환(FFT)을 이용한 주파수 분석이나, 더 발전된 최적화 알고리즘(e.g., Simulated Annealing, Genetic Algorithm)을 도입할 수 있습니다.

---

### `PoincareMatrix::decompress()`

```rust
impl PoincareMatrix {
    pub fn decompress(&self) -> Vec<f32> {
        // ... (구현)
    }
}
```

-   **역할**: `PoincareMatrix`가 가진 **단일 `Packed64` 시드**를 사용하여, 완전한 `f32` 행렬 전체를 복원합니다. 이것이 '압축 해제' 과정입니다.
-   **핵심 과정**:
    1.  `rows` x `cols` 크기의 `Vec<f32>`를 0으로 초기화합니다.
    2.  이중 `for` 루프를 통해 모든 행렬 좌표 `(i, j)`를 순회합니다.
    3.  각 좌표마다 `self.seed.compute_weight(i, j, rows, cols)`를 호출하여 해당 위치의 가중치 값을 계산합니다.
    4.  계산된 값을 결과 행렬의 해당 위치에 채워 넣습니다.
-   **특징**: 이 과정은 결정론적(deterministic)입니다. 동일한 시드와 차원이 주어지면, 언제나 100% 동일한 행렬이 생성됩니다. 