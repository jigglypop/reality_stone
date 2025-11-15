"""Metric context routing 모듈 - Phase 3

docs/sentence_topic_architecture.md의 5장 L2: Metric Context Router 명세 준수
"""
import torch
from typing import List, Dict

try:
    import reality_stone.metrikey as metrikey
    HAS_METRIKEY = True
except ImportError:
    HAS_METRIKEY = False
    print("Warning: reality_stone.metrikey not available, using fallback")


class MetricContextRouter:
    """
    Metric context routing
    
    docs 명세:
    - metric key와 score로부터 SPD 메트릭 합성
    - Cholesky factorization으로 L_i 계산
    - eigenvalue 클램핑으로 안정성 확보
    """
    def __init__(
        self,
        d_head: int = 64,
        lambda_min: float = 0.1,
        lambda_max: float = 5.0,
        cache_size: int = 1000
    ):
        self.d_head = d_head
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.cache = {}  # 키 조합 캐싱
        self.cache_size = cache_size
    
    def __call__(
        self,
        metric_keys: List[str],
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Metric key와 score로부터 SPD Cholesky factor 생성
        
        Args:
            metric_keys: [B*T] 문장별 metric key 리스트
            scores: [B, T] 우선순위 점수
        
        Returns:
            L: [B, T, d_head, d_head] Cholesky factor
        """
        B, T = scores.shape
        L_list = []
        
        scores_flat = scores.flatten()
        
        for i, key in enumerate(metric_keys):
            score_val = scores_flat[i].item()
            
            # 캐시 확인
            cache_key = (key, round(score_val, 2))
            if cache_key in self.cache:
                L_list.append(self.cache[cache_key])
                continue
            
            # SPD 메트릭 생성
            if HAS_METRIKEY:
                try:
                    G = metrikey.metric_from_keys(
                        [key],
                        dim=self.d_head,
                        min_lambda=self.lambda_min,
                        max_lambda=self.lambda_max,
                        masses=[score_val]
                    )
                except Exception as e:
                    # 키가 없으면 identity 사용
                    print(f"Warning: metric key '{key}' not found, using identity. Error: {e}")
                    G = torch.eye(self.d_head)
            else:
                # Fallback: identity + 작은 랜덤 섭동
                G = torch.eye(self.d_head) + torch.randn(self.d_head, self.d_head) * 0.01 * score_val
                G = (G + G.T) / 2  # 대칭화
            
            # Eigenvalue 클램핑
            G = self._clamp_eigenvalues(G)
            
            # Cholesky factorization
            try:
                L = torch.linalg.cholesky(G)
            except RuntimeError:
                # 수치 불안정 시 regularization
                G = G + torch.eye(self.d_head) * 1e-6
                try:
                    L = torch.linalg.cholesky(G)
                except RuntimeError:
                    # 최후의 수단: identity
                    print(f"Warning: Cholesky failed for key '{key}', using identity")
                    L = torch.eye(self.d_head)
            
            # 캐시 저장
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = L.clone()
            
            L_list.append(L)
        
        return torch.stack(L_list).view(B, T, self.d_head, self.d_head)
    
    def _clamp_eigenvalues(self, G: torch.Tensor) -> torch.Tensor:
        """
        Eigenvalue 범위 제한
        
        docs 명세:
        - SPD 조건 유지 및 수치 안정성
        - eigenvalue를 [lambda_min, lambda_max] 범위로 클램프
        """
        try:
            eigvals, eigvecs = torch.linalg.eigh(G)
            eigvals = torch.clamp(eigvals, self.lambda_min, self.lambda_max)
            return eigvecs @ torch.diag(eigvals) @ eigvecs.T
        except RuntimeError:
            # eigenvalue 분해 실패 시 원본 반환
            return G

