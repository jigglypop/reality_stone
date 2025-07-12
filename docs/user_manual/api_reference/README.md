# API 레퍼런스

Reality Stone의 모든 공개 API에 대한 상세한 문서입니다.

## 목차

### [레이어 (Layers)](./layers/)
하이퍼볼릭 공간에서 동작하는 신경망 레이어들
- **[Poincaré Ball Layer](./layers/poincare_ball.md)** - 포인카레 볼 모델 기반 레이어
- **[Lorentz Layer](./layers/lorentz.md)** - 로렌츠 모델 기반 레이어
- **[Klein Layer](./layers/klein.md)** - Klein 모델 기반 레이어

### [연산 (Operations)](./ops/)
하이퍼볼릭 공간에서의 기본 수학 연산들
- **[Mobius Operations](./ops/mobius_ops.md)** - 뫼비우스 변환 연산
- **[Lorentz Operations](./ops/lorentz_ops.md)** - 로렌츠 모델 연산
- **[Klein Operations](./ops/klein_ops.md)** - Klein 모델 연산

### [핵심 모듈 (Core)](./core/)
기본 유틸리티와 변환 함수들
- **[Utilities](./core/utilities.md)** - 공통 유틸리티 함수들

## 빠른 API 참조

### 주요 함수들

| 함수 | 설명 | 사용 예제 |
|------|------|-----------|
| `poincare_ball_layer()` | Poincaré Ball 레이어 | `rs.poincare_ball_layer(u, v, c=1.0, t=0.5)` |
| `lorentz_layer()` | Lorentz 레이어 | `rs.lorentz_layer(u, v, c=1.0, t=0.5)` |
| `klein_layer()` | Klein 레이어 | `rs.klein_layer(u, v, c=1.0, t=0.5)` |
| `poincare_add()` | Poincaré 덧셈 | `rs.poincare_add(x, y, c=1.0)` |
| `poincare_distance()` | Poincaré 거리 | `rs.poincare_distance(x, y, c=1.0)` |

### 클래스들

| 클래스 | 설명 | 사용 예제 |
|--------|------|-----------|
| `PoincareBallLayer` | Poincaré Ball 레이어 클래스 | `PoincareBallLayer.apply(u, v, c, t)` |
| `LorentzLayer` | Lorentz 레이어 클래스 | `LorentzLayer.apply(u, v, c, t)` |
| `KleinLayer` | Klein 레이어 클래스 | `KleinLayer.apply(u, v, c, t)` |
| `MobiusAdd` | Mobius 덧셈 클래스 | `MobiusAdd.apply(x, y, c)` |
| `MobiusScalarMul` | Mobius 스칼라 곱셈 클래스 | `MobiusScalarMul.apply(x, r, c)` |

## 매개변수 가이드

### 공통 매개변수

#### 곡률 매개변수 (c)
- **타입**: `float`
- **범위**: `c > 0`
- **설명**: 하이퍼볼릭 공간의 곡률을 제어합니다
- **일반적 값**: 
  - `1e-3`: 거의 평평한 공간 (시작점으로 권장)
  - `1e-2`: 보통 곡률
  - `1e-1`: 높은 곡률 (계층적 구조가 강할 때)

#### 보간 비율 (t)
- **타입**: `float`
- **범위**: `0.0 ≤ t ≤ 1.0`
- **설명**: 두 입력 간의 보간 비율을 제어합니다
- **값의 의미**:
  - `t = 0.0`: 첫 번째 입력만 사용
  - `t = 0.5`: 균등 혼합 (권장)
  - `t = 1.0`: 두 번째 입력만 사용

#### 텐서 형태
- **입력 텐서**: `torch.Tensor` 형태 `[batch_size, feature_dim]`
- **출력 텐서**: 입력과 동일한 형태
- **지원 데이터 타입**: `torch.float32`, `torch.float64`
- **지원 디바이스**: CPU, CUDA

## 주의사항

### 수치적 안정성
- 입력 텐서의 norm이 1에 가까우면 수치적 불안정성이 발생할 수 있습니다
- 입력 데이터를 적절히 스케일링하여 사용하세요 (예: `x * 0.1`)

### 메모리 사용량
- CUDA 버전은 더 많은 GPU 메모리를 사용할 수 있습니다
- 큰 배치 크기 사용 시 메모리 사용량을 모니터링하세요

### 그래디언트 클리핑
- 하이퍼볼릭 공간에서는 그래디언트가 폭발할 수 있습니다
- `torch.nn.utils.clip_grad_norm_()`을 사용하여 그래디언트를 제한하세요

## 상세 문서

각 모듈의 상세한 사용법과 예제는 해당 섹션을 참조하세요:

- **[레이어 문서](./layers/)**: 각 레이어의 수학적 배경과 구현 세부사항
- **[연산 문서](./ops/)**: 기본 연산들의 사용법과 성능 특성
- **[핵심 모듈 문서](./core/)**: 유틸리티 함수들의 활용법

## 관련 링크

- **[시작하기](../01_getting_started.md)**: 기본 사용법 학습
- **[예제 가이드](../03_examples.md)**: 실제 사용 예제들
- **[수학적 배경](../04_mathematical_background.md)**: 이론적 기초 