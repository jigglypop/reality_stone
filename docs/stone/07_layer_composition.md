# 07. Layer Composition — Order‑Preserving Merge

- 합성: T_total = T_n … T_1 (순서 보존)
- 무손실 저장: 파라미터 ≈ d^2, 이론적 압축률 ≈ L
- 저손실 압축: SVD/FFT/구조화 행렬(ACDC/Monarch 등)
- 주의: 비교환성(순서 의존), 재분해 난이도↑ → 파인튜닝 전략 별도 설계

## 구현 세부(f32/f64)

- f32: 속도 최적, 온라인 전처리 기본 경로
- f64: 합성/적용을 머신 엡실론 수준으로 무손실화(≈1e−12~1e−14)
- 혼용: 합성(f64) + 적용(f32), 민감 구간만 f64 적용

## Compact 합성 API

- 키와 질량 스케줄만으로 합성:
  - key_i = master_key#i, mass_i = mass_base + i·mass_step
  - compose_layers_gravity_compact_f64(master_key, L, d, λ_min, λ_max, mass_base, mass_step)
  - 저장 스펙: O(L) + 메타 → 대규모 배포/버전관리 용이
