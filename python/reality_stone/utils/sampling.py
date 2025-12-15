import torch

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty is None or float(penalty) == 1.0:
        return logits
    out = logits.clone()
    bsz = generated_ids.size(0)
    for bi in range(bsz):
        seen = torch.unique(generated_ids[bi]).long()
        vals = out[bi, seen]
        out[bi, seen] = torch.where(vals < 0, vals * float(penalty), vals / float(penalty))
    return out

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    out = logits
    if top_k is not None and int(top_k) > 0 and int(top_k) < out.size(-1):
        k = int(top_k)
        topk_vals, topk_idx = torch.topk(out, k, dim=-1)
        masked = torch.full_like(out, float("-inf"))
        masked.scatter_(1, topk_idx, topk_vals)
        out = masked
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > float(top_p)
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        out = torch.full_like(out, float("-inf"))
        out.scatter_(1, sorted_idx, sorted_logits)
    return out


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    scores = logits
    scores = apply_repetition_penalty(scores, generated_ids, repetition_penalty)
    if temperature is not None and float(temperature) > 0 and float(temperature) != 1.0:
        scores = scores / float(max(temperature, 1e-8))
    scores = top_k_top_p_filter(scores, top_k=top_k, top_p=top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)

