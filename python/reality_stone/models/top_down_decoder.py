import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict


class TopDownDecoder(nn.Module):
    def __init__(self, d_model: int, d_head: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.sentence_proj = nn.Linear(d_model, d_model)
        self.token_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        paragraph_embedding: Tensor,
        num_sentences: int,
        max_length: int,
        paragraph_metric: Optional[Tensor] = None,
        replacement_mask: Optional[Tensor] = None,
        original_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size = paragraph_embedding.shape[0]
        sent = self.sentence_proj(paragraph_embedding)
        sentence_embeddings = sent.unsqueeze(1).expand(batch_size, num_sentences, self.d_model)
        token_logits = self.token_proj(sentence_embeddings.view(batch_size * num_sentences, self.d_model))
        token_ids = token_logits.argmax(dim=-1)
        tokens = token_ids.view(batch_size, num_sentences, -1)
        seq_len = tokens.shape[2]
        if seq_len < max_length:
            pad_len = max_length - seq_len
            pad = torch.zeros(batch_size, num_sentences, pad_len, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=2)
        elif seq_len > max_length:
            tokens = tokens[:, :, :max_length]
        return {"sentence_embeddings": sentence_embeddings, "tokens": tokens}
