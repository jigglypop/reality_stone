# 06. Vector DB Integration — ANN and Frames

- 1차: ANN(
  - HNSW/IVF/OPQ, 유클리드/코사인
  - 쿼리만 프레임 전처리 q' = L_k q
- 2차: 상위 M 지오데식/마할라노비스 재랭크
- OPQ 호환: 블록-직교와 잘 맞음(서브스페이스 회전)
- IVF: 센트로이드만 프레임 변환·캐시

## 통합 시나리오(단일 인덱스, 권한별 랭킹)

- 입력: 원본 임베딩 x_i 를 단일 인덱스에 저장(복수 인덱스·복제 없음)
- 질의: 요청 컨텍스트→키 k, `G_k=L_k^T L_k`, `q' = L_k q` 전처리 → ANN(1차)
- 재랭크: top‑M에 한해 `d_k(x, y) = (x−y)^T G_k (x−y)`로 재랭크(2차, 옵션)
- 합성: 리만 레이어가 다수면 `T_total = T_n … T_1`, `q' = T_total q`로 1회 전처리

## 운영 규칙(보안/성능)

- 보안: 키/메트릭은 TEE/HSM 내 생성·적용, 세션 회전 `L̃_k = R_s L_k`(요청별)
- 노출 최소화: 변환 임베딩 디스크 상주 금지, TTL 캐시·메모리 한정
- 수치 안정: `λ_min, λ_max` 범위 고정, 정규화/스케일 관리
- 캐시: (key-hash, dim)→`L_k`/`T_total` 캐시, (key, list)→`L_k·centroid` 캐시(IVF/OPQ)

## Kubernetes(Docker Desktop)에서 Qdrant로 통합

사전: Docker Desktop의 Kubernetes가 활성화되어 있고, `helm`, `kubectl` 사용 가능하다고 가정합니다. 포트포워딩 금지 정책에 따라 Service는 NodePort를 사용합니다. 레플리카는 1.

1) Helm 저장소 등록/업데이트

```bash
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update
```

2) Qdrant 설치(NodePort, replica=1)

```bash
helm upgrade --install qdrant qdrant/qdrant \
  --namespace qdrant --create-namespace \
  --set replicaCount=1 \
  --set service.type=NodePort \
  --set resources.requests.cpu=100m \
  --set resources.requests.memory=256Mi
```

3) 서비스 확인 및 NodePort 파악

```bash
kubectl get svc -n qdrant qdrant -o wide
```

출력의 `PORT(S)`에서 `http`의 NodePort를 확인합니다(예: 30080→6333/TCP). Docker Desktop 환경에서는 보통 `127.0.0.1:<NodePort>`로 접근 가능합니다.

4) 컬렉션 생성(REST)

```bash
curl -s -X PUT "http://127.0.0.1:<NODEPORT>/collections/documents" \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {"size": 128, "distance": "Cosine"},
    "optimizers_config": {"default_segment_number": 1}
  }'
```

5) 임베딩 적재(REST)

```bash
curl -s -X PUT "http://127.0.0.1:<NODEPORT>/collections/documents/points?wait=true" \
  -H 'Content-Type: application/json' \
  -d '{
    "points": [
      {"id": 1, "vector": [0.1,0.2, ... 128차원 ...], "payload": {"dept": "AI"}},
      {"id": 2, "vector": [...], "payload": {"dept": "Finance"}}
    ]
  }'
```

6) 질의 전처리(L_k) + 검색(ANN 1차) + 재랭크(옵션)

- Rust에서 L_k 계산: `metric_factor_cholesky(spd_metric_from_key(k, d, λ_min, λ_max))`
- 쿼리 전처리: `q' = apply_linear(L_k, q[None,:])`
- ANN 질의(REST): `/collections/documents/points/search`로 q'
- top‑M 결과를 `mahalanobis_distance_sq_g(q, x, G_k)`로 재랭크(옵션)

예시(REST 검색):

```bash
curl -s -X POST "http://127.0.0.1:<NODEPORT>/collections/documents/points/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [ ... q_prime 128차원 ... ],
    "limit": 5
  }'
```

## Rust API 매핑(코어는 Rust)

- 메트릭/전처리
  - `spd_metric_from_key(key, dim, min_lambda, max_lambda)` → `G_k`
  - `metric_factor_cholesky(G_k)` → `L_k`
  - `apply_linear(L_k, queries)` → `q'`
  - 세션 회전: `rotate_metric_factor_block(session_key, L_k, global_dim)`
- 합성: `compose_layers_order_preserving([T_l])` → `T_total`
- 재랭크: `mahalanobis_distance_sq_g(x, y, G_k)` 또는 `*_sq_l`

## 성능/캐시 팁

- (key, list)별 `L_k·centroid` TTL 캐시(IVF/OPQ)
- 전처리 저비용화: 블록‑직교/저랭크 구조, O(d·r) 또는 O(d log d)
- 수치 안정: `λ_min/λ_max` 제약, 입력 정규화, f64 내부 계산(필요 시)
