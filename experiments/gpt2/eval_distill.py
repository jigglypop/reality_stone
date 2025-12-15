from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class DistillEvalResult:
    logits_cos: float
    logits_rel_l2: float
    teacher_ppl: float
    rsulf_ppl: float

@torch.no_grad()
def eval_teacher_vs_rsulf(
    original_model,
    rs_lm,
    tokenizer,
    device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> DistillEvalResult:
    if isinstance(device, str):
        device = torch.device(device)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    t_out = original_model(input_ids=input_ids, use_cache=False)
    t_logits = t_out.logits
    s_logits = rs_lm(input_ids)
    if isinstance(s_logits, (tuple, list)):
        s_logits = s_logits[0]
    t_pred = t_logits[:, :-1, :]
    s_pred = s_logits[:, :-1, :]
    targets = input_ids[:, 1:]
    V = t_logits.size(-1)

    flat_targets = targets.reshape(-1)
    if attention_mask is not None:
        valid = attention_mask[:, 1:].reshape(-1).ne(0)
    else:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            valid = torch.ones_like(flat_targets, dtype=torch.bool)
        else:
            valid = flat_targets.ne(int(pad_id))
    if not valid.any():
        valid = torch.ones_like(valid, dtype=torch.bool)

    def _xent(logits: torch.Tensor) -> torch.Tensor:
        flat = logits.reshape(-1, V)
        return F.cross_entropy(flat[valid], flat_targets[valid])

    t_loss = _xent(t_pred)
    s_loss = _xent(s_pred)

    # Similarity on valid positions (same mask)
    t_flat = t_pred.reshape(-1, V)[valid]
    s_flat = s_pred.reshape(-1, V)[valid]
    cos = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
    rel = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)

    return DistillEvalResult(
        logits_cos=float(cos),
        logits_rel_l2=float(rel.item()),
        teacher_ppl=float(math.exp(float(t_loss.item()))),
        rsulf_ppl=float(math.exp(float(s_loss.item()))),
    )


