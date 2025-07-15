# 5. 수학 함수 라이브러리 (`src/math.rs`)

이 문서에서는 `compute_weight` 함수 내부에서 사용되는 다양한 수학 헬퍼 함수들을 설명합니다. 이 함수들은 모두 순수 함수(pure function)로, 입력 값에만 의존하여 결과를 반환합니다.

---

### 회전 (`get_rotation_angle`)

-   **역할**: `rot_code` (0-15)를 입력받아, 미리 정의된 회전 각도(radian)를 반환합니다.
-   **예시**: `rot_code = 3`이면 `PI / 4.0` (45도)를 반환합니다.

---

### 각도/반지름 미분 (`apply_angular_derivative`, `apply_radial_derivative`)

-   **역할**: 주기 함수(삼각함수, 쌍곡선함수)의 미분 순환성을 구현합니다.
-   `apply_angular_derivative`:
    -   `d_theta` 값(0, 1, 2, 3)에 따라 `sin(θ)`의 0, 1, 2, 3차 도함수(`sin`, `cos`, `-sin`, `-cos`)를 계산합니다.
    -   `basis_id`를 통해 `sin` 기반인지 `cos` 기반인지 판단하여 시작점을 결정합니다.
-   `apply_radial_derivative`:
    -   `d_r` 값(`false`, `true`)에 따라 `sinh(r)`의 0, 1차 도함수(`sinh`, `cosh`)를 계산합니다.
    -   마찬가지로 `basis_id`를 통해 `sinh` 기반/`cosh` 기반을 결정합니다.

---

### 베셀 함수 (`bessel_j0`, `bessel_i0`, `bessel_k0`, `bessel_y0`)

-   **역할**: 파동이나 진동 패턴을 모델링하는 데 사용되는 특수 함수들입니다. 각 함수는 고유한 형태의 패턴을 생성합니다.
-   **구현**: 수치해석학적인 다항식 근사(polynomial approximation)나 점근적 확장(asymptotic expansion)을 사용하여 구현되었습니다. 입력 값의 범위에 따라 다른 공식을 사용하여 정확도를 높입니다.

---

### 기타 파형 함수

-   **`sech(x)`**: 쌍곡선 시컨트(secant) 함수. 종 모양(bell-shaped) 커브를 생성합니다. `1 / cosh(x)`와 같습니다.
-   **`triangle_wave(x)`**: 삼각파. 톱니파와 유사한 선형적인 주기 패턴을 생성합니다.
-   **`morlet_wavelet(r, theta, freq)`**: 모르레 웨이블릿. 가우시안 함수(종 모양)와 코사인파를 곱한 형태로, 특정 위치에 국소화된 파동 패턴을 생성하는 데 효과적입니다.

이러한 다양한 수학 함수들의 조합을 통해, 제한된 파라미터만으로도 풍부하고 복잡한 형태의 가중치 행렬을 생성할 수 있습니다. 