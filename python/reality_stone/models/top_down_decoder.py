import torch
import torch.nn as nn
from typing import Optional, Dict

class TopDownDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.vocab_size = vocab_size
        
        self.paragraph_to_sentences = nn.Linear(d_model, d_model)
        
        self.lexical_head = nn.Linear(d_model, vocab_size)
    
    def generate_sentence_structure(
        self,
        paragraph_embedding: torch.Tensor,
        num_sentences: int,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = paragraph_embedding.shape[0]
        device = paragraph_embedding.device
        
        para_exp = paragraph_embedding.unsqueeze(1).expand(B, num_sentences, self.d_model)
        
        pos_encoding = torch.arange(num_sentences, device=device).float()
        pos_encoding = pos_encoding.unsqueeze(0).unsqueeze(-1).expand(B, num_sentences, self.d_model)
        pos_encoding = pos_encoding / num_sentences
        
        sentence_embeddings = self.paragraph_to_sentences(para_exp + pos_encoding * 0.1)
        
        return sentence_embeddings
    
    def generate_tokens(
        self,
        sentence_embeddings: torch.Tensor,
        max_length: int,
        replacement_mask: Optional[torch.Tensor] = None,
        original_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = sentence_embeddings.shape
        device = sentence_embeddings.device
        
        logits = self.lexical_head(sentence_embeddings)
        
        logits_exp = logits.unsqueeze(2).expand(B, T, max_length, self.vocab_size)
        
        pred_tokens = torch.argmax(logits_exp, dim=-1)
        
        if original_tokens is not None and replacement_mask is not None:
            tokens = torch.where(
                replacement_mask.bool(),
                pred_tokens,
                original_tokens,
            )
        else:
            tokens = pred_tokens
        
        return tokens
    
    def forward(
        self,
        paragraph_embedding: torch.Tensor,
        num_sentences: int,
        max_length: int,
        paragraph_metric: Optional[torch.Tensor] = None,
        replacement_mask: Optional[torch.Tensor] = None,
        original_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        sentence_embeddings = self.generate_sentence_structure(
            paragraph_embedding,
            num_sentences,
            metric_ctx=paragraph_metric,
        )
        
        tokens = self.generate_tokens(
            sentence_embeddings,
            max_length,
            replacement_mask=replacement_mask,
            original_tokens=original_tokens,
        )
        
        return {
            "sentence_embeddings": sentence_embeddings,
            "tokens": tokens,
        }

