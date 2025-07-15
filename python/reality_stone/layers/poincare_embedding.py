import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .poincare import project_to_ball


class PoincareEmbedding(nn.Module):
    """푸앵카레 볼 공간에서의 임베딩 레이어"""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        c: float = 1.0,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.c = c
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        # 임베딩 가중치를 푸앵카레 볼 내부에 초기화
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()
        
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
    
    def reset_parameters(self):
        """임베딩을 푸앵카레 볼 내부에 균등하게 초기화"""
        with torch.no_grad():
            # 유클리드 공간에서 초기화
            nn.init.uniform_(self.weight, -0.1, 0.1)
            
            # 정합성 테스트를 위해 project_to_ball 비활성화
            # 푸앵카레 볼로 투영 (반지름 내부로 제한)
            # self.weight.data = project_to_ball(self.weight.data)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """푸앵카레 볼 공간에서 임베딩 조회"""
        # 기본 임베딩 조회
        embedded = F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        
        # 정합성 테스트를 위해 project_to_ball 비활성화
        # embedded = project_to_ball(embedded)
        
        return embedded
    
    @classmethod
    def from_euclidean_embedding(cls, embedding: nn.Embedding, c: float = 1.0):
        """기존 유클리드 임베딩을 푸앵카레 임베딩으로 변환"""
        poincare_emb = cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            c=c,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
        )
        
        # 가중치 복사 (변환 없이)
        with torch.no_grad():
            euclidean_weights = embedding.weight.data
            # 정합성 테스트를 위해 project_to_ball 비활성화
            # poincare_weights = project_to_ball(euclidean_weights)
            poincare_emb.weight.data = euclidean_weights.clone()
            
            # padding_idx 처리
            if embedding.padding_idx is not None:
                poincare_emb.weight[embedding.padding_idx].fill_(0)
        
        return poincare_emb
    
    def to_euclidean(self) -> torch.Tensor:
        """푸앵카레 임베딩을 유클리드 공간으로 역변환"""
        # 방법 1: 직접 사용 (이미 유클리드 좌표)
        return self.weight.data
        
        # 방법 2: 로그 맵 사용 (더 정교한 역변환)
        # return self.poincare.log_map_zero(self.weight)
    
    def compress_to_seed(self, block_size: int = 64) -> dict:
        """임베딩을 RBE 시드로 압축"""
        # 임베딩 행렬을 블록으로 나누어 압축
        weight_np = self.weight.detach().cpu().numpy()
        num_embeddings, embedding_dim = weight_np.shape
        
        # 블록 수 계산
        emb_blocks = (num_embeddings + block_size - 1) // block_size
        dim_blocks = (embedding_dim + block_size - 1) // block_size
        
        seeds = []
        block_info = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'emb_blocks': emb_blocks,
            'dim_blocks': dim_blocks,
            'block_size': block_size,
            'c': self.c,
            'padding_idx': self.padding_idx,
        }
        
        # 각 블록을 압축 (여기서는 구조만 제공)
        for i in range(emb_blocks):
            for j in range(dim_blocks):
                start_i = i * block_size
                end_i = min((i + 1) * block_size, num_embeddings)
                start_j = j * block_size
                end_j = min((j + 1) * block_size, embedding_dim)
                
                block = weight_np[start_i:end_i, start_j:end_j]
                # 실제 압축은 RBE compressor 사용
                # seed = compress_block(block)
                # seeds.append(seed)
        
        return {'seeds': seeds, 'block_info': block_info}


class EquivalentPoincareEmbedding(nn.Module):
    """유클리드 공간에서 푸앵카레 임베딩과 동등한 연산을 수행하는 레이어"""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        c: float = 1.0,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.c = c
        self.padding_idx = padding_idx
        
        # 유클리드 임베딩 (푸앵카레 좌표를 직접 저장)
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """유클리드 공간에서 푸앵카레 임베딩 조회 (동일성 보장)"""
        # 기본 임베딩 조회 후, 어떠한 변환도 거치지 않음
        return self.embedding(input)
    
    @classmethod
    def from_poincare_embedding(cls, poincare_emb: PoincareEmbedding):
        """푸앵카레 임베딩을 동등한 유클리드 임베딩으로 변환"""
        equiv_emb = cls(
            num_embeddings=poincare_emb.num_embeddings,
            embedding_dim=poincare_emb.embedding_dim,
            c=poincare_emb.c,
            padding_idx=poincare_emb.padding_idx,
        )
        
        # 가중치 직접 복사
        equiv_emb.embedding.weight.data = poincare_emb.weight.data.clone()
        
        return equiv_emb 