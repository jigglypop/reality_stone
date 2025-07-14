# Reality Stone 레이어 아키텍처 설명서

## 목차
1. [개요](#개요)
2. [레이어 구조](#레이어-구조)
3. [하이퍼볼릭 기하학 레이어](#하이퍼볼릭-기하학-레이어)
4. [압축 레이어](#압축-레이어)
5. [구현 상태](#구현-상태)
6. [인터페이스 분석](#인터페이스-분석)

## 개요

Reality Stone은 하이퍼볼릭 기하학과 고급 압축 기법을 활용한 신경망 레이어 라이브러리입니다. 각 레이어는 Python, Rust, CUDA 3단계 구조로 구현되어 있습니다.

## 레이어 구조

### 아키텍처 다이어그램
![레이어 아키텍처](위의 mermaid 다이어그램 참조)

### 계층별 역할
- **Python Layer**: PyTorch 통합, 사용자 인터페이스
- **Rust Core**: 핵심 수학 연산, 메모리 안전성
- **CUDA Kernels**: GPU 가속, 병렬 처리

## 하이퍼볼릭 기하학 레이어

### 1. PoincareBallLayer
**목적**: 푸앵카레 볼 모델에서의 신경망 연산

**주요 기능**:
- Möbius 덧셈/스칼라 곱
- 측지선 거리 계산
- 지수/로그 매핑
- 동적 곡률 지원

**구현 상태**:
```
Python: ✅ 완전 구현 (poincare.py)
Rust:   ✅ 완전 구현 (poincare.rs)
CUDA:   ✅ 완전 구현 (poincare.cu)
```

**특징**:
- 3가지 선형 레이어 변형 제공 (HyperbolicLinear, GeodesicLinear, EquivalentHyperbolicLinear)
- 동적 곡률 최적화 지원
- 다차원 텐서 지원

### 2. LorentzLayer
**목적**: 로렌츠 모델(하이퍼볼로이드)에서의 연산

**주요 기능**:
- 로렌츠 내적
- 로렌츠 거리
- 모델 간 변환 (Poincaré ↔ Lorentz)

**구현 상태**:
```
Python: ✅ 완전 구현 (lorentz.py)
Rust:   ✅ 완전 구현 (lorentz.rs)
CUDA:   ✅ 완전 구현 (lorentz.cu)
```

**특징**:
- 수치적으로 더 안정적
- 계층적 데이터에 적합

### 3. KleinLayer
**목적**: Klein 디스크 모델에서의 연산

**주요 기능**:
- Klein 모델 연산
- 모델 간 변환
- 효율적인 거리 계산

**구현 상태**:
```
Python: ✅ 완전 구현 (klein.py)
Rust:   ✅ 완전 구현 (klein.rs)
CUDA:   ✅ 완전 구현 (klein.cu)
```

**특징**:
- 선형 측지선
- 계산 효율성

## 압축 레이어

### 4. BitfieldLinear
**목적**: 가중치를 22비트로 압축하는 선형 레이어

**압축 구조**:
```rust
// 32비트 압축 코드
struct CompressedCode {
    phase: u8,      // 8비트 위상
    amplitude: u8,  // 8비트 진폭
    basis_idx: u8,  // 8비트 기저 인덱스
    metadata: u8,   // 8비트 메타데이터
}
```

**주요 기능**:
- 186배 압축률
- 직접 추론 (압축 해제 없이)
- INT8/INT16 최적화
- Tensor Core 지원
- 계층적 압축 (2.5비트까지)

**구현 상태**:
```
Python: ✅ 완전 구현 (bitfield.py)
Rust:   ✅ 완전 구현 (bitfield/mod.rs)
CUDA:   ✅ 완전 구현 (bitfield.cu)
```

**최적화 기법**:
1. **순환성 기반**: 삼각함수 미분의 순환성 활용
2. **INT8 양자화**: 기저 테이블과 잔차 INT8 변환
3. **계층적 압축**: 4비트/2.5비트 극한 압축
4. **학습 가능한 압축**: Gumbel-Softmax 기반

### 5. SplineLinear
**목적**: Catmull-Rom 스플라인을 이용한 가중치 압축

**주요 기능**:
- 제어점 기반 가중치 보간
- 적응적 압축률
- 부드러운 가중치 표현

**구현 상태**:
```
Python: ✅ 완전 구현 (spline.py)
Rust:   ⚠️ 부분 구현 (spline.rs - 바인딩만)
CUDA:   ❌ 미구현
```

**문제점**:
- 핵심 로직이 Python에만 구현
- GPU 가속 없음
- 성능 병목

## 구현 상태 요약

| 레이어 | Python | Rust Core | CUDA | 통합 테스트 |
|--------|--------|-----------|------|-------------|
| Poincaré | ✅ | ✅ | ✅ | ✅ |
| Lorentz | ✅ | ✅ | ✅ | ✅ |
| Klein | ✅ | ✅ | ✅ | ✅ |
| Bitfield | ✅ | ✅ | ✅ | ✅ |
| Spline | ✅ | ⚠️ | ❌ | ⚠️ |

## 인터페이스 분석

### Python 인터페이스 불일치
```python
# 함수형 인터페이스 (Poincaré, Lorentz, Klein)
output = poincare_add(x, y, c=1.0)

# 클래스형 인터페이스 (Bitfield, Spline)
layer = BitfieldLinear(in_features, out_features)
output = layer(input)
```

### Rust 바인딩 중복
```rust
// 모든 레이어에서 반복되는 패턴
#[pyfunction]
pub fn xxx_add<'py>(...) -> &'py PyArray2<f32> {
    // 거의 동일한 코드
}
```

### 메모리 관리 문제
- GPU ↔ CPU 전환 시 불필요한 복사
- 다차원 텐서 처리 비효율
- 메모리 풀링 없음

## 주요 문제점 정리

1. **코드 분산**: 바인딩이 6개 파일에 분산
2. **인터페이스 불일치**: 레이어별로 다른 API
3. **중복 코드**: 유사한 패턴이 반복
4. **불완전한 구현**: SplineLinear의 Rust/CUDA 미구현
5. **성능 병목**: 메모리 복사, 배치 처리 비효율

이러한 문제점들을 해결하기 위한 상세한 리팩토링 계획은 다음 문서를 참조하세요. 