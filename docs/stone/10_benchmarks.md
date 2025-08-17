# 10. Benchmarks — Measures and Expected Gains

- 측정
  - 품질: nDCG@k, Recall@k
  - 지연: 평균/중앙/95p, 전처리·ANN·재랭크 분해
  - 메모리: 인덱스·캐시·모델(T_total)
  - 보안: 키 회수 시 복원 실패율/랭킹 붕괴율
- 예상(보수적)
  - 전처리 지연: 0.3–1.2 ms (d≈1k~1.5k)
  - P95 검색 지연 20–40% 개선(ANN/OPQ/IVF 결합)
  - 중복 스토리지 5–12× 절감(물리 분할 대비)

## 실측(참조, CPU)

- 합성(T_total, f64): d=1024, L=8 → 약 135,150.75 ms (1회 캐시 권장)
- 적용(apply_linear)
  - f64: batch=1 ≈ 4.02 ms, 64 ≈ 7.00 ms, 1024 ≈ 56.65 ms
  - f32: batch=1 ≈ 3.00 ms, 64 ≈ 6.02 ms, 256 ≈ 18.04 ms
- 무손실 검증(f64): 합성 vs 순차 ||diff|| ≈ 2.7e−12, 배치 max|diff| ≈ 2.0e−14, 랭킹 동일 100%

## 압축률(합성행렬 저장 대비)

- d=1024, L=8: T_l f32 전부 저장 대비 ≈ 254,200× (키+masses 스펙만 저장)
- gpt‑oss‑급(예: hidden_size=2880, layers=36): ≈ 1,788,000× (config 기반 추정)

## 스케일 예측(공식)

- 합성: O(L·d^3)
  - t_compose(d,L) ≈ t_ref · (L/L_ref) · (d/d_ref)^3
- 적용(matvec): O(d^2)
  - t_apply(d) ≈ t_ref · (d/d_ref)^2

참고값(예): ref(d=512,L=4)에서 합성≈5,094 ms일 때,
- gpt‑oss‑120B급(d=2880,L=36) 합성 ≈ 5,094 · (36/4) · (2880/512)^3 ≈ 8.16×10^6 ms (오프라인 1회)
- d=1024에서 batch=1024 적용 ≈ 56.65 ms → d=2880은 (2880/1024)^2 배 증가 ≈ 447 ms (CPU 추정)

## 운영 팁

- 합성은 오프라인/시작 시 1회, T_total 캐시
- 온라인 전처리는 f32, 민감 구간은 f64 혼용
- IVF/OPQ: (키,리스트)별 `L_k·centroid` TTL 캐시로 P95 20–40% 개선
- 구조화/저랭크(블록‑직교, ACDC/Monarch 등)로 O(d·r) 또는 O(d log d)로 추가 단축
