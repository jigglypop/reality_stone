# 03. Architecture — Modules and Flow

## 모듈
- Metric Generator: 키→(Q_k, D_k)→G_k
- Preprocessor: z = L_k x (또는 R_k, g‑직교)
- ANN Index: HNSW/IVF/OPQ (유클리드/코사인)
- Reranker: 상위 M 재랭크(d_k 또는 지오데식)
- Layer Composer: T_total 합성/압축(무손실/저손실)
- Key Management: HSM/TEE, 세션 회전

## 흐름(질의)
1) user_ctx→key k
2) G_k, L_k 복원(캐시)
3) q' = L_k q, ANN 1차 후보
4) top-M만 d_k/지오데식 재랭크(옵션)

## 흐름(색인)
- 문서는 원본 임베딩 저장, 메타만 부착(키 미저장)
- IVF/OPQ 사용 시 센트로이드만 프레임 변환 캐시
