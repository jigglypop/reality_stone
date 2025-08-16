# MetriKey: Keyed Riemannian Metric with Order‑Preserving Layer Composition

본 문서는 “메트릭=키(SPD 메트릭)”와 “순서보존 리만 합성(레이어 단일화)” 기법을 Reality Stone에 적용·운영하는 방법을 장별로 정리합니다.

## 목차
- 00_overview.md — 개요와 문제정의, 기대효과
- 01_principles.md — 핵심 원칙(메트릭=키, 순서보존 합성, 하이브리드 검색)
- 02_math.md — 수학적 기초(SPD 메트릭, g‑직교, 지오데식, 푸시포워드)
- 03_architecture.md — 시스템 아키텍처(모듈·흐름·캐시)
- 04_pipeline.md — 색인/질의 파이프라인(성능/캐싱 전략)
- 05_security.md — 보안 모델(HSM/TEE, 세션 회전, 로깅)
- 06_integration_vector_db.md — 벡터DB 통합(ANN, OPQ/IVF/HNSW)
- 07_layer_composition.md — 레이어 합성·압축(무손실/저손실)
- 08_metric_learning.md — 메트릭 학습(손실/제약/안정화)
- 09_poc_plan.md — 2주 PoC 계획
- 10_benchmarks.md — 지표/측정/예상치
- 11_faq.md — 자주 묻는 질문

본 챕터들은 Reality Stone 코드베이스(리만 연산/압축)와 즉시 결합 가능하도록 작성되었습니다.


