import torch
import torch.nn as nn
from typing import Optional

from reality_stone.models.riemannian_aggregation import RiemannianAggregation
from reality_stone.layers.poincare import project_to_ball


class BottomUpEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        manifold: str = "poincare",
        c: float = 1e-3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        
        self.token_to_sentence = RiemannianAggregation(d_model=d_model, manifold=manifold, c=c, temperature=temperature)
        
        self.sentence_to_paragraph = RiemannianAggregation(d_model=d_model, manifold=manifold, c=c, temperature=temperature)
        
        self.poincare_proj = nn.Linear(d_model, d_head)
    
    def encode_tokens_to_sentences(
        self,
        token_embeddings: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, L, d = token_embeddings.shape
        
        sentence_list = []
        for t in range(T):
            tokens_t = token_embeddings[:, t, :, :]
            
            if metric_ctx is not None:
                metric_t = metric_ctx[:, t, :, :]
            else:
                metric_t = None
            
            sent_emb = self.token_to_sentence(children_states=tokens_t, metric_ctx=metric_t)
            
            sentence_list.append(sent_emb)
        
        sentence_embeddings = torch.stack(sentence_list, dim=1)
        
        return sentence_embeddings
    
    def encode_sentences_to_paragraph(
        self,
        sentence_embeddings: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = sentence_embeddings.shape
        
        paragraph_emb = self.sentence_to_paragraph(
            children_states=sentence_embeddings,
            metric_ctx=metric_ctx,
        )
        
        return paragraph_emb
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        sentence_metric: Optional[torch.Tensor] = None,
        paragraph_metric: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sentence_embeddings = self.encode_tokens_to_sentences(
            token_embeddings,
            metric_ctx=sentence_metric,
        )
        
        paragraph_embedding = self.encode_sentences_to_paragraph(
            sentence_embeddings,
            metric_ctx=paragraph_metric,
        )
        
        return sentence_embeddings, paragraph_embedding

