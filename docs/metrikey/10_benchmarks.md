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
