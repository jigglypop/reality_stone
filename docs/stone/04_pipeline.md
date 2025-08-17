# 04. Pipeline — Indexing and Query

## 색인(offline)
- x_i = f_θ(doc)
- 벡터DB 저장(메타: 정책/조직 태그)

## 질의(online)
- k→(Q_k, D_k)→G_k, L_k (캐시)
- q' = L_k q; ANN 검색(단일 인덱스)
- top-M 재랭크(d_k 또는 지오데식)

## 성능 팁
- (키, IVF-list)별 (L_k·centroid) TTL 캐시
- 블록-직교/저랭크로 전처리 O(dr) 또는 O(d log d)
- 수치안정: λ_min/λ_max 제약, 정규화

## Kubernetes에서의 배포 팁

- NodePort 서비스로 외부 접근(포트포워딩 금지 정책 준수)
- 레플리카=1, 리소스 요청 소량으로 시작 후 점진적 조정
- ANN엔진(Qdrant/Faiss 서버 등)과 앱 계층 분리, 키/메트릭은 TEE/HSM에 격리
