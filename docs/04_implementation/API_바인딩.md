# Reality Stone Python API 바인딩 (Python API Bindings)

이 문서는 Rust에서 Python으로 바인딩된 모든 함수와 클래스를 정리합니다.

## 바인딩 구조

Reality Stone은 `_rust` 확장 모듈을 통해 Rust 구현을 Python에서 사용할 수 있습니다.

```python
from reality_stone import _rust
```

## 1. Mobius Operations (`_rust`)

### CPU 함수

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `mobius_add_cpu` | Mobius 덧셈 (CPU) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `mobius_scalar_cpu` | Mobius 스칼라 곱 (CPU) | `u: Array2<f32>`, `r: f32`, `c: f32` | `Array2<f32>` |
