"""Wake/NREM/REM refinement for standalone CE artifacts."""

from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

try:
    from .engine import CEEngine, DEFAULT_PROMPTS, state_partition_counts
    from .ce_ops import pq_build_codebook
    from .utils import safe_print
except ImportError:
    from reality_stone.clarus.engine import CEEngine, DEFAULT_PROMPTS, state_partition_counts
    from reality_stone.clarus.ce_ops import pq_build_codebook
    from reality_stone.clarus.utils import safe_print


@dataclass
class SleepBatch:
    state_x: torch.Tensor
    prev_x: torch.Tensor
    target_y: torch.Tensor
    soft_y: torch.Tensor
    hard_mask: torch.Tensor
    top1_hits: torch.Tensor
    top50_hits: torch.Tensor
    target_ids: torch.Tensor
    top10_hits: torch.Tensor | None = None
    risk_scores: torch.Tensor | None = None
    phase_risk_scores: torch.Tensor | None = None
    teacher_top_ids: torch.Tensor | None = None
    teacher_top_probs: torch.Tensor | None = None


@dataclass
class DecoderTokenHead:
    token_ids: torch.Tensor
    state_proj: torch.Tensor | None
    prev_proj: torch.Tensor | None
    bias: torch.Tensor | None
    scale: float = 1.0


@dataclass
class PromptReplayBuffer:
    capacity: int
    prompts: deque[str] = field(default_factory=deque)

    def add(self, prompt: str):
        if not prompt:
            return
        self.prompts.append(prompt)
        while len(self.prompts) > self.capacity:
            self.prompts.popleft()

    def extend(self, prompts: list[str]):
        for prompt in prompts:
            self.add(prompt)

    def items(self) -> list[str]:
        return list(self.prompts)


DEFAULT_CORPUS_DATASET = "lcw99/wikipedia-korean-20221001"
DEFAULT_CORPUS_SPLIT = "train"
DEFAULT_CORPUS_TEXT_COLUMN = "text"


def _split_corpus_documents(text: str) -> list[str]:
    docs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                docs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        docs.append(" ".join(current))
    if docs:
        return docs
    fallback = " ".join(part.strip() for part in text.split() if part.strip())
    return [fallback] if fallback else []


def _chunk_document(text: str, *, max_chars: int = 320, min_chars: int = 64) -> list[str]:
    cleaned = " ".join(part.strip() for part in text.split() if part.strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    segments = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
        if part.strip()
    ]
    if not segments:
        segments = [cleaned]

    chunks: list[str] = []
    current = ""
    for segment in segments:
        if len(segment) > max_chars:
            words = segment.split()
            partial = ""
            for word in words:
                candidate = word if not partial else f"{partial} {word}"
                if len(candidate) <= max_chars:
                    partial = candidate
                    continue
                if len(partial) >= min_chars:
                    chunks.append(partial)
                    partial = word
                else:
                    overflow = candidate[:max_chars].strip()
                    if overflow:
                        chunks.append(overflow)
                    partial = candidate[max_chars:].strip()
            if partial:
                if len(partial) >= min_chars or not chunks:
                    chunks.append(partial)
            current = ""
            continue

        candidate = segment if not current else f"{current} {segment}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = segment
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def load_corpus_documents(
    data_path: str | None = None,
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str = DEFAULT_CORPUS_SPLIT,
    text_column: str = DEFAULT_CORPUS_TEXT_COLUMN,
    doc_limit: int = 256,
    text_limit: int = 1_000_000,
) -> list[str]:
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing corpus file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs: list[str] = []
        for raw_doc in _split_corpus_documents(text[: int(text_limit)]):
            docs.extend(_chunk_document(raw_doc))
            if len(docs) >= max(int(doc_limit), 1):
                break
        return docs[: max(int(doc_limit), 1)]

    dataset_name = dataset_name or DEFAULT_CORPUS_DATASET
    try:
        datasets = importlib.import_module("datasets")
        ds = datasets.load_dataset(dataset_name, dataset_config, split=dataset_split)
    except Exception as exc:
        raise RuntimeError(
            "Pass --data or install the 'datasets' package to load an external Korean corpus."
        ) from exc

    docs: list[str] = []
    total_chars = 0
    for row in ds:
        text = row.get(text_column)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        chunks = _chunk_document(text)
        if not chunks:
            continue
        docs.extend(chunks)
        total_chars += sum(len(chunk) for chunk in chunks)
        if len(docs) >= max(int(doc_limit), 1) or total_chars >= int(text_limit):
            break
    if not docs:
        raise RuntimeError("Loaded corpus is empty.")
    return docs


def _content_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"[0-9A-Za-z가-힣]{2,}", text)}


def prioritize_documents_for_prompts(
    docs: list[str],
    prompts: list[str] | None,
) -> list[str]:
    if not docs or not prompts:
        return list(docs)

    prompt_weights: dict[str, int] = {}
    for prompt in prompts:
        for token in _content_terms(prompt):
            prompt_weights[token] = prompt_weights.get(token, 0) + 1

    if not prompt_weights:
        return list(docs)

    scored_docs: list[tuple[float, int, str]] = []
    for idx, doc in enumerate(docs):
        doc_tokens = _content_terms(doc)
        overlap = float(sum(prompt_weights.get(token, 0) for token in doc_tokens))
        scored_docs.append((overlap, idx, doc))

    prioritized = [
        doc
        for overlap, idx, doc in sorted(
            scored_docs,
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if overlap > 0.0
    ]
    if not prioritized:
        return list(docs)

    ordered: list[str] = []
    seen: set[str] = set()
    for doc in [*prioritized, *docs]:
        if doc in seen:
            continue
        seen.add(doc)
        ordered.append(doc)
    return ordered


def ridge_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.float()
    y = y.float()
    if weights is not None:
        w = weights.float().clamp_min(1e-6).sqrt().unsqueeze(1)
        x = x * w
        y = y * w
    xtx = x.T @ x
    xty = x.T @ y
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    return torch.linalg.solve(xtx + float(ridge) * eye, xty)


def fit_linear_with_bias(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x.float(), ones], dim=1)
    out = ridge_solve(x_aug, y.float(), ridge, weights=weights)
    return out[:-1], out[-1]


def batch_weights(batch: SleepBatch, rem_weight: float) -> torch.Tensor:
    weights = torch.ones(
        batch.target_y.shape[0],
        dtype=batch.target_y.dtype,
        device=batch.target_y.device,
    )
    if batch.hard_mask.numel() and rem_weight > 1.0:
        weights[batch.hard_mask.bool()] = float(rem_weight)
    return weights


def fit_decoder_from_batch(
    batch: SleepBatch,
    *,
    prev_scale: float,
    ridge: float,
    rem_weight: float = 1.0,
    rem_mix: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = batch.target_y.clone()
    weights = batch_weights(batch, rem_weight)

    if batch.hard_mask.numel() and rem_mix > 0.0:
        hard = batch.hard_mask.bool()
        y[hard] = (1.0 - rem_mix) * y[hard] + rem_mix * batch.soft_y[hard]

    feat = torch.cat([batch.state_x.float(), float(prev_scale) * batch.prev_x.float()], dim=1)
    proj, bias = fit_linear_with_bias(feat, y, ridge=ridge, weights=weights)
    d = batch.state_x.shape[1]
    state_proj = proj[:d]
    prev_proj = proj[d : 2 * d]
    pred = batch.state_x @ state_proj + float(prev_scale) * (batch.prev_x @ prev_proj) + bias
    denom = float(pred.pow(2).sum().item())
    if denom > 1e-8:
        scale = float((pred * y).sum().item() / denom)
        state_proj = state_proj * scale
        prev_proj = prev_proj * scale
        bias = bias * scale
    return state_proj, prev_proj, bias


def fit_token_head_from_batch(
    batch: SleepBatch,
    *,
    prev_scale: float,
    ridge: float,
    rem_weight: float = 1.0,
    max_vocab: int = 2048,
    scale: float = 1.0,
) -> DecoderTokenHead | None:
    if batch.teacher_top_ids is None or batch.teacher_top_probs is None:
        return None

    top_ids = batch.teacher_top_ids.long()
    top_probs = batch.teacher_top_probs.float()
    if top_ids.numel() == 0 or top_probs.numel() == 0:
        return None

    flat_ids = top_ids.reshape(-1)
    flat_probs = top_probs.reshape(-1)
    uniq_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    mass = torch.zeros(uniq_ids.shape[0], dtype=flat_probs.dtype, device=flat_probs.device)
    mass.scatter_add_(0, inverse, flat_probs)

    if max_vocab > 0 and uniq_ids.numel() > max_vocab:
        keep = torch.topk(mass, max_vocab).indices
        uniq_ids = uniq_ids.index_select(0, keep)
        uniq_ids, _ = torch.sort(uniq_ids)

    if uniq_ids.numel() == 0:
        return None

    token_map = {int(token_id): col for col, token_id in enumerate(uniq_ids.tolist())}
    y = torch.zeros(
        (top_ids.shape[0], uniq_ids.shape[0]),
        dtype=torch.float32,
        device=batch.state_x.device,
    )
    for row_idx, (row_ids, row_probs) in enumerate(zip(top_ids.tolist(), top_probs.tolist(), strict=False)):
        for token_id, prob in zip(row_ids, row_probs, strict=False):
            col_idx = token_map.get(int(token_id))
            if col_idx is not None:
                y[row_idx, col_idx] += float(prob)

    weights = batch_weights(batch, rem_weight).float()
    denom = weights.sum().clamp_min(1e-6)
    feat = torch.cat([batch.state_x.float(), float(prev_scale) * batch.prev_x.float()], dim=1)
    proj, bias = fit_linear_with_bias(feat, y, ridge=ridge, weights=weights)
    d = batch.state_x.shape[1]
    state_proj = proj[:d]
    prev_proj = proj[d : 2 * d]
    return DecoderTokenHead(
        token_ids=uniq_ids.long().cpu(),
        state_proj=state_proj.cpu(),
        prev_proj=prev_proj.cpu(),
        bias=bias.cpu(),
        scale=float(scale),
    )


def finetune_vocab_head_from_batch(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    lr: float = 1e-3,
    steps: int = 64,
    batch_size: int = 256,
    rem_weight: float = 1.0,
    weight_decay: float = 1e-4,
    soft_target_weight: float = 0.35,
) -> dict[str, float]:
    if batch.state_x.numel() == 0:
        return {"loss": 0.0, "top1_acc": 0.0, "top10_acc": 0.0, "steps": 0, "batch_size": 0}

    eng.ensure_vocab_head()
    assert eng.decoder_vocab_weight is not None

    state_x = batch.state_x.to(eng.device).float()
    prev_x = batch.prev_x.to(eng.device).float()
    target_ids = batch.target_ids.to(eng.device).long()
    sample_weights = batch_weights(batch, rem_weight).to(eng.device).float()
    teacher_top_ids = (
        None
        if batch.teacher_top_ids is None
        else batch.teacher_top_ids.to(eng.device).long()
    )
    teacher_top_probs = (
        None
        if batch.teacher_top_probs is None
        else batch.teacher_top_probs.to(eng.device).float()
    )
    soft_target_weight = min(max(float(soft_target_weight), 0.0), 1.0)

    weight = eng.decoder_vocab_weight.detach().clone().to(eng.device)
    bias = (
        torch.zeros(weight.shape[0], dtype=weight.dtype, device=eng.device)
        if eng.decoder_vocab_bias is None
        else eng.decoder_vocab_bias.detach().clone().to(eng.device)
    )
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    optimizer = torch.optim.AdamW([weight, bias], lr=float(lr), weight_decay=float(weight_decay))
    total = int(state_x.shape[0])
    batch_size = max(1, min(int(batch_size), total))
    steps = max(1, int(steps))

    last_loss = 0.0

    for step in range(steps):
        start = (step * batch_size) % total
        end = start + batch_size
        if end <= total:
            idx = torch.arange(start, end, device=eng.device)
        else:
            tail = torch.arange(start, total, device=eng.device)
            head = torch.arange(0, end - total, device=eng.device)
            idx = torch.cat([tail, head], dim=0)

        query = eng.decoder_query(state_x.index_select(0, idx), prev_x.index_select(0, idx))
        logits = F.linear(query, weight, bias)
        target = target_ids.index_select(0, idx)
        hard_loss = F.cross_entropy(logits, target, reduction="none")
        loss = hard_loss
        if (
            teacher_top_ids is not None
            and teacher_top_probs is not None
            and soft_target_weight > 0.0
        ):
            top_ids = teacher_top_ids.index_select(0, idx).clamp(0, logits.shape[1] - 1)
            top_probs = teacher_top_probs.index_select(0, idx)
            top_probs = top_probs / top_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
            student_top_logits = logits.gather(1, top_ids)
            soft_loss = -(top_probs * F.log_softmax(student_top_logits, dim=1)).sum(dim=1)
            loss = (1.0 - soft_target_weight) * hard_loss + soft_target_weight * soft_loss
        weights = sample_weights.index_select(0, idx)
        loss = (loss * weights).sum() / weights.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        query = eng.decoder_query(state_x, prev_x)
        logits = F.linear(query, weight, bias)
        top1 = float((logits.argmax(dim=-1) == target_ids).float().mean().item())
        top10_ids = torch.topk(logits, min(10, logits.shape[1]), dim=-1).indices
        top10 = float((top10_ids == target_ids.unsqueeze(1)).any(dim=1).float().mean().item())

    eng.apply_vocab_head(weight.detach().cpu(), bias=bias.detach().cpu(), scale=1.0)
    return {
        "loss": last_loss,
        "top1_acc": top1,
        "top10_acc": top10,
        "steps": float(steps),
        "batch_size": float(batch_size),
    }


def build_refresh_args(
    ce_args,
    *,
    steps: int,
    cb_topk: int,
    metric_rank: int,
    noise_scale: float,
):
    payload = vars(ce_args).copy()
    payload.update(
        steps=int(steps),
        cb_topk=int(cb_topk),
        metric_rank=int(metric_rank),
        noise_scale=float(noise_scale),
    )
    return argparse.Namespace(**payload)


def allocate_phase_sample_counts(
    total_samples: int,
    phase_profile: dict[str, float],
) -> dict[str, int]:
    total_samples = max(0, int(total_samples))
    if total_samples == 0:
        return {name: 0 for name in phase_profile}

    names = list(phase_profile)
    weights = [max(float(phase_profile[name]), 0.0) for name in names]
    weight_sum = sum(weights)
    if weight_sum <= 1e-8:
        base = total_samples // max(len(names), 1)
        counts = {name: base for name in names}
        for name in names[: total_samples - base * len(names)]:
            counts[name] += 1
        return counts

    raw = [total_samples * weight / weight_sum for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total_samples - sum(counts)
    order = sorted(
        range(len(names)),
        key=lambda idx: (raw[idx] - counts[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    positive = [idx for idx, weight in enumerate(weights) if weight > 0.0]
    if total_samples >= len(positive):
        missing = [idx for idx in positive if counts[idx] == 0]
        for receiver in missing:
            donors = sorted(
                (idx for idx in positive if counts[idx] > 1 and idx != receiver),
                key=lambda idx: (counts[idx] - raw[idx], counts[idx]),
                reverse=True,
            )
            if not donors:
                break
            donor = donors[0]
            counts[donor] -= 1
            counts[receiver] += 1

    return {name: int(count) for name, count in zip(names, counts, strict=False)}


def _build_sleep_batch(
    state_rows: list[torch.Tensor],
    prev_rows: list[torch.Tensor],
    target_rows: list[torch.Tensor],
    soft_rows: list[torch.Tensor],
    hard_rows: list[bool],
    top1_hits: list[bool],
    top10_hits: list[bool],
    top50_hits: list[bool],
    risk_rows: list[float],
    phase_risk_rows: list[float],
    target_ids: list[int],
    teacher_top_ids_rows: list[torch.Tensor],
    teacher_top_prob_rows: list[torch.Tensor],
) -> SleepBatch:
    return SleepBatch(
        state_x=torch.stack(state_rows, dim=0),
        prev_x=torch.stack(prev_rows, dim=0),
        target_y=torch.stack(target_rows, dim=0),
        soft_y=torch.stack(soft_rows, dim=0),
        hard_mask=torch.tensor(hard_rows, dtype=torch.bool),
        top1_hits=torch.tensor(top1_hits, dtype=torch.bool),
        top10_hits=torch.tensor(top10_hits, dtype=torch.bool),
        top50_hits=torch.tensor(top50_hits, dtype=torch.bool),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        risk_scores=torch.tensor(risk_rows, dtype=torch.float32),
        phase_risk_scores=torch.tensor(phase_risk_rows, dtype=torch.float32),
        teacher_top_ids=torch.stack(teacher_top_ids_rows, dim=0),
        teacher_top_probs=torch.stack(teacher_top_prob_rows, dim=0),
    )


def _mean_phase_grounding_risk(step_meta: dict[str, object]) -> float:
    risk = step_meta.get("phase_grounding_risk")
    if torch.is_tensor(risk) and risk.numel():
        return float(risk.float().mean().item())
    return 0.0


def _context_slice(full_ids: torch.Tensor, end_pos: int, window_tokens: int) -> torch.Tensor:
    start = max(0, int(end_pos) - max(int(window_tokens), 1))
    return full_ids[:, start:int(end_pos)]


def _target_distribution(
    eng: CEEngine,
    target_id: int,
    *,
    topk: int,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_emb = eng.token_embedding([target_id]).squeeze(0).detach()
    k = min(max(int(topk), 1), eng.vocab)
    if k <= 1:
        top_idx = torch.tensor([target_id], dtype=torch.long, device=eng.device)
        probs = torch.tensor([1.0], dtype=torch.float32, device=eng.device)
        return target_emb, top_idx.cpu(), probs.cpu(), target_emb

    if teacher_logits is not None:
        top_vals, top_idx_raw = torch.topk(teacher_logits.float(), min(k, teacher_logits.numel()))
        probs = F.softmax(top_vals, dim=0)
        ordered_ids = top_idx_raw.tolist()
        if target_id not in ordered_ids:
            ordered_ids = [target_id] + ordered_ids[:-1]
            probs_list = [0.5] + [p * 0.5 for p in probs.tolist()[:-1]]
            probs = torch.tensor(probs_list, dtype=torch.float32)
            probs = probs / probs.sum()
        top_idx = torch.tensor(ordered_ids, dtype=torch.long, device=eng.device)
        soft_target = (eng.token_embedding(ordered_ids).detach() * probs.unsqueeze(1).to(eng.device)).sum(0)
        return target_emb, top_idx.cpu(), probs.cpu(), soft_target

    soft_scores = eng.lexical_scores(target_emb)
    gather_k = min(max(k * 2, k + 1), soft_scores.numel())
    top_vals, top_idx = torch.topk(soft_scores, gather_k)
    neighbor_ids: list[int] = []
    neighbor_vals: list[float] = []
    seen = {int(target_id)}
    for token_id, score in zip(top_idx.tolist(), top_vals.tolist(), strict=False):
        token_int = int(token_id)
        if token_int in seen:
            continue
        seen.add(token_int)
        neighbor_ids.append(token_int)
        neighbor_vals.append(float(score))
        if len(neighbor_ids) >= k - 1:
            break

    ordered_ids = [int(target_id), *neighbor_ids]
    top_idx = torch.tensor(ordered_ids, dtype=torch.long, device=eng.device)
    probs = torch.zeros(top_idx.shape[0], dtype=torch.float32, device=eng.device)
    if top_idx.shape[0] == 1:
        probs[0] = 1.0
    else:
        target_mass = 0.85
        probs[0] = target_mass
        neighbor_scores = torch.tensor(neighbor_vals, dtype=torch.float32, device=eng.device)
        neighbor_probs = F.softmax(neighbor_scores, dim=0)
        probs[1:] = (1.0 - target_mass) * neighbor_probs
    soft_target = (probs.unsqueeze(1) * eng.token_embedding(top_idx).detach()).sum(dim=0)
    return target_emb, top_idx.detach().cpu(), probs.detach().cpu(), soft_target


def collect_sleep_batch(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    teacher_topk: int,
    refresh_interval: int = 0,
    refresh_steps: int = 48,
    refresh_cb_topk: int = 128,
    refresh_metric_rank: int = 0,
    refresh_noise_scale: float = 0.0,
    sample_budget: int | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> SleepBatch:
    if not prompts:
        raise ValueError("collect_sleep_batch requires at least one prompt")
    prompts = [
        prompt
        for prompt in prompts
        if eng.tok.encode(prompt, return_tensors="pt").shape[1] > 1
    ]
    if not prompts:
        raise ValueError("collect_sleep_batch requires corpus entries with at least two tokens")

    state_rows: list[torch.Tensor] = []
    prev_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    soft_rows: list[torch.Tensor] = []
    hard_rows: list[bool] = []
    top1_hits: list[bool] = []
    top10_hits: list[bool] = []
    top50_hits: list[bool] = []
    risk_rows: list[float] = []
    phase_risk_rows: list[float] = []
    target_ids: list[int] = []
    teacher_top_ids_rows: list[torch.Tensor] = []
    teacher_top_prob_rows: list[torch.Tensor] = []
    refresh_args = None
    if refresh_interval > 0:
        refresh_args = build_refresh_args(
            ce_args,
            steps=refresh_steps,
            cb_topk=refresh_cb_topk,
            metric_rank=refresh_metric_rank,
            noise_scale=refresh_noise_scale,
        )

    from tqdm import tqdm as _tqdm

    target_samples = len(prompts) * max(1, int(max_new_tokens))
    if sample_budget is not None:
        target_samples = max(1, int(sample_budget))

    _pbar = _tqdm(total=target_samples, desc="    collect", unit="tok", ncols=80)
    prompt_idx = 0
    _collect_t0 = time.time()
    _collect_timeout = max(300.0, target_samples * 2.0)
    while len(state_rows) < target_samples:
        if time.time() - _collect_t0 > _collect_timeout:
            break
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1
        full_ids = eng.tok.encode(prompt, return_tensors="pt").to(eng.device)
        if full_ids.shape[1] <= 1:
            continue
        cursor = min(max(int(seed_tokens), 1), full_ids.shape[1] - 1)
        while cursor < full_ids.shape[1] and len(state_rows) < target_samples:
            ids = _context_slice(full_ids, cursor, context_window)
            if ids.shape[1] == 0:
                cursor += max(1, int(max_new_tokens))
                continue
            ctx = eng.context_from_ids(ids)
            if eng.model is not None:
                with torch.no_grad():
                    teacher_out = eng.model(ids, output_hidden_states=True)
                    ce_hidden = teacher_out.hidden_states[-1][0, -1].float().detach()
                phi_state = ctx.phi.detach()
            else:
                relax_result = eng.relax_context(ctx, ce_args)
                ce_hidden = eng.ce_hidden(relax_result["m_star"]).detach()
                phi_state = relax_result["phi_updated"].detach()
            init_layer = ctx.best_layer
            history_ids = ids[0].tolist()
            prev_hidden = None
            prev_prev_hidden = None
            context_anchor = ce_hidden.detach().clone()
            max_stop = min(full_ids.shape[1], cursor + max(1, int(max_new_tokens)))

            for target_pos in range(cursor, max_stop):
                target_id = int(full_ids[0, target_pos].item())
                prev_id = int(ids[0, -1].item())
                prev_emb = eng.token_embedding([prev_id]).squeeze(0).detach()

                teacher_logits = None
                if eng.model is not None:
                    with torch.no_grad():
                        teacher_out = eng.model(ids)
                        teacher_logits = teacher_out.logits[0, -1].detach().cpu()

                target_emb, top_idx, probs, soft_target = _target_distribution(
                    eng,
                    target_id,
                    topk=teacher_topk,
                    teacher_logits=teacher_logits,
                )

                standalone_logits, step_meta = eng.standalone_logits(
                    ce_hidden,
                    prev_id,
                    temperature=1.0,
                    history_ids=history_ids,
                    prev_hidden=prev_hidden,
                    prev_prev_hidden=prev_prev_hidden,
                    context_anchor=context_anchor,
                    return_meta=True,
                )
                top_ids = torch.topk(standalone_logits, min(50, standalone_logits.numel())).indices.tolist()
                stand_top1 = int(top_ids[0])
                hit1 = stand_top1 == target_id
                hit10 = target_id in top_ids[:10]
                hit50 = target_id in top_ids

                state_rows.append(ce_hidden.cpu())
                prev_rows.append(prev_emb.cpu())
                _pbar.update(1)
                target_rows.append(target_emb.cpu())
                soft_rows.append(soft_target.cpu())
                hard_rows.append(not hit50)
                top1_hits.append(hit1)
                top10_hits.append(hit10)
                top50_hits.append(hit50)
                risk_rows.append(float(step_meta["curvature_risk_score"]))
                phase_risk_rows.append(_mean_phase_grounding_risk(step_meta))
                target_ids.append(target_id)
                teacher_top_ids_rows.append(top_idx)
                teacher_top_prob_rows.append(probs)

                if len(state_rows) >= target_samples:
                    break

                step_hidden = ce_hidden.detach().clone()
                prev_prev_hidden = prev_hidden
                prev_hidden = step_hidden
                next_token = torch.tensor([[target_id]], device=eng.device)
                ids = torch.cat([ids, next_token], dim=1)
                if ids.shape[1] > int(context_window):
                    ids = ids[:, -int(context_window) :]
                history_ids.append(target_id)
                if (
                    refresh_args is not None
                    and target_pos + 1 < max_stop
                    and (target_pos - cursor + 1) % refresh_interval == 0
                ):
                    refresh_ctx = eng.context_from_ids(
                        ids,
                        init_layer=init_layer,
                        phi=phi_state,
                        need_teacher=False,
                    )
                    refresh_result = eng.relax_context(refresh_ctx, refresh_args)
                    ce_hidden = eng.ce_hidden(refresh_result["m_star"]).detach()
                    phi_state = refresh_result["phi_updated"].detach()
                    init_layer = refresh_ctx.best_layer
            cursor = max_stop

    _pbar.close()
    return _build_sleep_batch(
        state_rows,
        prev_rows,
        target_rows,
        soft_rows,
        hard_rows,
        top1_hits,
        top10_hits,
        top50_hits,
        risk_rows,
        phase_risk_rows,
        target_ids,
        teacher_top_ids_rows,
        teacher_top_prob_rows,
    )


def batch_stats(batch: SleepBatch) -> dict[str, float]:
    top1 = batch.top1_hits.float().mean().item() if batch.top1_hits.numel() else 0.0
    top10 = batch.top10_hits.float().mean().item() if batch.top10_hits is not None and batch.top10_hits.numel() else 0.0
    top50 = batch.top50_hits.float().mean().item() if batch.top50_hits.numel() else 0.0
    hard = batch.hard_mask.float().mean().item() if batch.hard_mask.numel() else 0.0
    risk = batch.risk_scores.float().mean().item() if batch.risk_scores is not None and batch.risk_scores.numel() else 0.0
    phase_risk = (
        batch.phase_risk_scores.float().mean().item()
        if batch.phase_risk_scores is not None and batch.phase_risk_scores.numel()
        else 0.0
    )
    return {
        "top1_acc": top1,
        "top10_acc": top10,
        "top50_acc": top50,
        "hard_ratio": hard,
        "curvature_risk": risk,
        "phase_grounding_risk": phase_risk,
        "samples": int(batch.state_x.shape[0]),
    }


def classify_state_dimensions(
    batch: SleepBatch,
    *,
    active_ratio: float,
    struct_ratio: float,
) -> dict[str, object]:
    scores = batch.state_x.abs().float().mean(dim=0)
    dim = int(scores.numel())
    active_k, struct_only_k, _ = state_partition_counts(dim, active_ratio, struct_ratio)
    struct_k = min(dim, active_k + struct_only_k)
    active_idx = torch.topk(scores, active_k).indices
    struct_idx = torch.topk(scores, struct_k).indices
    active_mask = torch.zeros(dim, dtype=torch.bool)
    struct_mask = torch.zeros(dim, dtype=torch.bool)
    active_mask[active_idx] = True
    struct_mask[struct_idx] = True
    struct_mask |= active_mask
    background_mask = ~struct_mask
    return {
        "active_mask": active_mask,
        "struct_mask": struct_mask,
        "background_mask": background_mask,
        "active_ratio": active_mask.float().mean().item(),
        "struct_ratio": (struct_mask & ~active_mask).float().mean().item(),
        "background_ratio": background_mask.float().mean().item(),
    }


def _weighted_covariance(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    x = x.float()
    if weights is not None:
        scale = weights.float().clamp_min(1e-6).sqrt().unsqueeze(1)
        x = x * scale
    cov = x.T @ x
    return cov / max(int(x.shape[0]), 1)


def covariance_delta(batch: SleepBatch, *, emphasize_hard: float = 1.0) -> torch.Tensor:
    weights = batch_weights(batch, emphasize_hard)
    cov_state = _weighted_covariance(batch.state_x, weights=weights)
    cov_target = _weighted_covariance(batch.target_y, weights=weights)
    delta = cov_target - cov_state
    return 0.5 * (delta + delta.T)


def offdiag_density(mask: torch.Tensor) -> float:
    dim = int(mask.shape[0])
    if dim <= 1:
        return 0.0
    offdiag = mask.detach().bool().clone()
    offdiag.fill_diagonal_(False)
    return float(offdiag.sum().item()) / float(dim * (dim - 1))


def row_topk_mask(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    dim = int(matrix.shape[0])
    mask = torch.zeros_like(matrix, dtype=torch.bool)
    if dim <= 1 or float(keep_ratio) <= 0.0:
        return mask

    upper_i, upper_j = torch.triu_indices(dim, dim, offset=1, device=matrix.device)
    if upper_i.numel() == 0:
        return mask

    pair_scores = matrix.detach().abs()[upper_i, upper_j]
    keep_pairs = int(math.floor(float(keep_ratio) * float(upper_i.numel()) + 0.5))
    keep_pairs = max(1, min(int(upper_i.numel()), keep_pairs))
    top_idx = torch.topk(pair_scores, keep_pairs).indices
    keep_i = upper_i.index_select(0, top_idx)
    keep_j = upper_j.index_select(0, top_idx)
    mask[keep_i, keep_j] = True
    mask[keep_j, keep_i] = True
    return mask


def normalize_update(matrix: torch.Tensor) -> torch.Tensor:
    peak = float(matrix.abs().amax().item()) if matrix.numel() else 0.0
    if peak <= 1e-8:
        return torch.zeros_like(matrix)
    return matrix / peak


def smooth_weight_matrix(w: torch.Tensor, laplacian: torch.Tensor, eta: float) -> torch.Tensor:
    lap = laplacian.float()
    w = w.float()
    smoothed = w - float(eta) * (lap @ w + w @ lap) / 2.0
    return 0.5 * (smoothed + smoothed.T)


def apply_nrem_weight_update(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    smooth_eta: float = 0.02,
    plastic_lr: float = 0.01,
) -> dict[str, float]:
    partition = classify_state_dimensions(
        batch,
        active_ratio=eng.active_ratio,
        struct_ratio=eng.struct_ratio,
    )
    if eng.active_dim_mask is None or eng.struct_dim_mask is None:
        eng.apply_state_partition(partition["active_mask"], partition["struct_mask"])

    lap = eng.state_graph_laplacian().detach().cpu()
    delta = covariance_delta(batch, emphasize_hard=1.0)
    plastic_mask = row_topk_mask(delta, eng.active_ratio)
    update = normalize_update(delta * plastic_mask)
    base_w = eng.W.detach().cpu().float()
    candidate_w = smooth_weight_matrix(base_w, lap, smooth_eta)
    if update.abs().amax().item() > 0.0:
        candidate_w = candidate_w + float(plastic_lr) * update
    eng.apply_relax_matrix(candidate_w)
    return {
        "smooth_eta": float(smooth_eta),
        "plastic_lr": float(plastic_lr),
        "delta_norm": float(delta.norm().item()),
        "active_ratio": float(partition["active_ratio"]),
        "struct_ratio": float(partition["struct_ratio"]),
        "background_ratio": float(partition["background_ratio"]),
        "plastic_density": offdiag_density(plastic_mask),
        "w_offdiag_density_pct": eng.weight_density() * 100.0,
        "w_target_density_pct": eng.target_w_density * 100.0,
    }


def apply_rem_weight_update(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    rem_rank: int = 8,
    rem_lr: float = 0.005,
    noise_scale: float = 0.01,
) -> dict[str, float]:
    delta = covariance_delta(batch, emphasize_hard=2.5)
    selected_mask = row_topk_mask(delta, eng.active_ratio)
    residual_mask = ~selected_mask
    residual_mask.fill_diagonal_(False)
    residual = delta * residual_mask
    dim = int(residual.shape[0])
    if residual.abs().amax().item() <= 1e-8:
        return {
            "rem_rank": float(rem_rank),
            "rem_lr": float(rem_lr),
            "noise_scale": float(noise_scale),
            "residual_norm": 0.0,
            "residual_density": offdiag_density(residual_mask),
        }

    rank = max(1, min(int(rem_rank), dim))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    proj = torch.randn(dim, rank, generator=gen)
    remix = residual @ proj @ proj.T / float(rank)
    remix = 0.5 * (remix + remix.T)
    if noise_scale > 0.0:
        remix = remix + float(noise_scale) * residual.abs().mean().item() * torch.randn_like(remix, generator=gen)
    update = normalize_update(remix)
    candidate_w = eng.W.detach().cpu().float() + float(rem_lr) * update
    eng.apply_relax_matrix(candidate_w)
    return {
        "rem_rank": float(rank),
        "rem_lr": float(rem_lr),
        "noise_scale": float(noise_scale),
        "residual_norm": float(residual.norm().item()),
        "residual_density": offdiag_density(residual_mask),
        "w_offdiag_density_pct": eng.weight_density() * 100.0,
        "w_target_density_pct": eng.target_w_density * 100.0,
    }


def evaluate_guard_set(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> dict[str, float]:
    if not prompts:
        return {
            "top1_acc": 0.0,
            "top10_acc": 0.0,
            "top50_acc": 0.0,
            "curvature_risk": 0.0,
            "phase_grounding_risk": 0.0,
            "samples": 0,
        }

    refresh_args = None
    if refresh_interval > 0:
        refresh_args = build_refresh_args(
            ce_args,
            steps=refresh_steps,
            cb_topk=refresh_cb_topk,
            metric_rank=refresh_metric_rank,
            noise_scale=refresh_noise_scale,
        )

    from tqdm import tqdm as _tqdm

    top1 = 0
    top10 = 0
    top50 = 0
    total = 0
    curvature_risk = 0.0
    phase_grounding_risk = 0.0

    for prompt in _tqdm(prompts, desc="    guard", unit="doc", ncols=80):
        full_ids = eng.tok.encode(prompt, return_tensors="pt").to(eng.device)
        if full_ids.shape[1] <= 1:
            continue
        cursor = min(max(int(seed_tokens), 1), full_ids.shape[1] - 1)
        while cursor < full_ids.shape[1]:
            ids = _context_slice(full_ids, cursor, context_window)
            ctx = eng.context_from_ids(ids, prompt=prompt)
            relax_result = eng.relax_context(ctx, ce_args)
            ce_hidden = eng.ce_hidden(relax_result["m_star"]).detach()
            phi_state = relax_result["phi_updated"].detach()
            init_layer = ctx.best_layer
            history_ids = ids[0].tolist()
            prev_hidden = None
            prev_prev_hidden = None
            context_anchor = ce_hidden.detach().clone()
            max_stop = min(full_ids.shape[1], cursor + max_new_tokens)

            for target_pos in range(cursor, max_stop):
                prev_id = int(ids[0, -1].item())
                target_id = int(full_ids[0, target_pos].item())
                logits, step_meta = eng.standalone_logits(
                    ce_hidden,
                    prev_id,
                    temperature=1.0,
                    history_ids=history_ids,
                    prev_hidden=prev_hidden,
                    prev_prev_hidden=prev_prev_hidden,
                    context_anchor=context_anchor,
                    return_meta=True,
                )
                top_ids = torch.topk(logits, min(50, logits.numel())).indices.tolist()
                top1 += int(top_ids[0] == target_id)
                top10 += int(target_id in top_ids[:10])
                top50 += int(target_id in top_ids)
                curvature_risk += float(step_meta["curvature_risk_score"])
                phase_grounding_risk += _mean_phase_grounding_risk(step_meta)
                total += 1

                step_hidden = ce_hidden.detach().clone()
                prev_prev_hidden = prev_hidden
                prev_hidden = step_hidden
                ids = torch.cat([ids, torch.tensor([[target_id]], device=eng.device)], dim=1)
                if ids.shape[1] > int(context_window):
                    ids = ids[:, -int(context_window) :]
                history_ids.append(target_id)
                if (
                    refresh_args is not None
                    and target_pos + 1 < max_stop
                    and (target_pos - cursor + 1) % refresh_interval == 0
                ):
                    refresh_ctx = eng.context_from_ids(
                        ids,
                        init_layer=init_layer,
                        phi=phi_state,
                        need_teacher=False,
                    )
                    refresh_result = eng.relax_context(refresh_ctx, refresh_args)
                    ce_hidden = eng.ce_hidden(refresh_result["m_star"]).detach()
                    phi_state = refresh_result["phi_updated"].detach()
                    init_layer = refresh_ctx.best_layer
            cursor = max_stop

    return {
        "top1_acc": top1 / max(total, 1),
        "top10_acc": top10 / max(total, 1),
        "top50_acc": top50 / max(total, 1),
        "curvature_risk": curvature_risk / max(total, 1),
        "phase_grounding_risk": phase_grounding_risk / max(total, 1),
        "samples": total,
    }


def should_accept_guard_update(
    before: dict[str, float],
    after: dict[str, float],
    *,
    min_top10_delta: float = 0.0,
    min_top50_delta: float = 0.0,
    max_top10_drop: float = 0.0,
    max_top50_drop: float = 0.0,
    max_phase_grounding_risk_increase: float | None = None,
) -> bool:
    top10_delta = float(after["top10_acc"]) - float(before["top10_acc"])
    top50_delta = float(after["top50_acc"]) - float(before["top50_acc"])
    phase_risk_delta = float(after.get("phase_grounding_risk", 0.0)) - float(
        before.get("phase_grounding_risk", 0.0)
    )
    if top10_delta < -float(max_top10_drop):
        return False
    if top50_delta < -float(max_top50_drop):
        return False
    if (
        max_phase_grounding_risk_increase is not None
        and phase_risk_delta > float(max_phase_grounding_risk_increase)
    ):
        return False
    return (
        top10_delta >= float(min_top10_delta)
        and top50_delta >= float(min_top50_delta)
    )


def run_guarded_microsleep_step(
    eng: CEEngine,
    buffer: PromptReplayBuffer,
    prompt: str,
    guard_prompts: list[str],
    ce_args,
    *,
    step_index: int,
    sleep_every: int,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
    guard_min_top10_delta: float = 0.0,
    guard_min_top50_delta: float = 0.0,
    guard_max_top10_drop: float = 0.0,
    guard_max_top50_drop: float = 0.0,
    guard_max_phase_grounding_risk_increase: float | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> dict[str, object] | None:
    buffer.add(prompt)
    if sleep_every <= 0 or step_index % int(sleep_every) != 0:
        return None

    train_prompts = buffer.items()
    snapshot = eng.decoder_snapshot()
    before_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    sleep_report = run_sleep_cycle(
        eng,
        train_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        ridge=ridge,
        rem_weight=rem_weight,
        rem_mix=rem_mix,
        token_head_max_vocab=token_head_max_vocab,
        token_head_scale=token_head_scale,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        refresh_pq=refresh_pq,
        pq_subdim=pq_subdim,
        pq_bits=pq_bits,
        pq_iters=pq_iters,
        pq_batch_size=pq_batch_size,
        pq_sample_size=pq_sample_size,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    after_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    accepted_update = should_accept_guard_update(
        before_guard,
        after_guard,
        min_top10_delta=guard_min_top10_delta,
        min_top50_delta=guard_min_top50_delta,
        max_top10_drop=guard_max_top10_drop,
        max_top50_drop=guard_max_top50_drop,
        max_phase_grounding_risk_increase=guard_max_phase_grounding_risk_increase,
    )
    if not accepted_update:
        eng.restore_decoder_snapshot(snapshot)

    effective_guard = after_guard if accepted_update else before_guard
    return {
        "step": step_index,
        "buffer_size": len(train_prompts),
        "train_prompts": train_prompts,
        "accepted": accepted_update,
        "sleep_report": sleep_report,
        "guard_before": before_guard,
        "guard_after": after_guard,
        "guard_effective": effective_guard,
        "guard_delta": {
            "top10_acc": after_guard["top10_acc"] - before_guard["top10_acc"],
            "top50_acc": after_guard["top50_acc"] - before_guard["top50_acc"],
        },
    }


def maybe_refresh_pq(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    subdim: int,
    bits: int,
    iters: int,
    batch_size: int,
    sample_size: int,
):
    if eng.emb is None:
        return None

    emb = eng.emb.detach().cpu().float()
    freq = torch.bincount(batch.target_ids.cpu(), minlength=emb.shape[0]).float()
    hot = torch.nonzero(freq > 0, as_tuple=False).squeeze(1)
    if hot.numel() == 0:
        return None

    hot_emb = emb.index_select(0, hot)
    pool = torch.cat([emb, hot_emb, hot_emb], dim=0)
    pq = pq_build_codebook(
        pool,
        subdim=subdim,
        bits=bits,
        iters=iters,
        batch_size=batch_size,
        sample_size=min(sample_size, pool.shape[0]),
        seed=0,
    )
    centroids = pq["centroids"].cpu()

    codes = torch.empty((emb.shape[0], centroids.shape[0]), dtype=torch.uint8)
    for sub_idx in range(centroids.shape[0]):
        start = sub_idx * subdim
        stop = start + subdim
        dist = torch.cdist(emb[:, start:stop], centroids[sub_idx].float())
        codes[:, sub_idx] = dist.argmin(dim=1).to(torch.uint8)

    eng.pq_centroids = centroids.to(eng.device)
    eng.pq_codes = codes.to(eng.device)
    eng.data["pq_centroids"] = centroids
    eng.data["pq_codes"] = codes
    return {
        "pq_centroids_mb": centroids.numel() * centroids.element_size() / 1024 / 1024,
        "pq_codes_mb": codes.numel() * codes.element_size() / 1024 / 1024,
    }


def run_sleep_cycle(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
    guard_prompts: list[str] | None = None,
    guard_min_top10_delta: float = 0.0,
    guard_min_top50_delta: float = 0.0,
    guard_max_top10_drop: float = 0.0,
    guard_max_top50_drop: float = 0.0,
    guard_max_phase_grounding_risk_increase: float | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
    vocab_finetune_lr: float = 1e-3,
    vocab_finetune_steps: int = 64,
    vocab_finetune_batch_size: int = 256,
    vocab_finetune_soft_target_weight: float = 0.35,
) -> dict[str, object]:
    guard_snapshot = None
    guard_before = None
    if guard_prompts:
        guard_snapshot = eng.decoder_snapshot()
        guard_before = evaluate_guard_set(
            eng,
            guard_prompts,
            ce_args,
            max_new_tokens=max_new_tokens,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            context_window=context_window,
            seed_tokens=seed_tokens,
        )

    phase_profile = {
        "wake": float(eng.wake_ratio),
        "nrem": float(eng.nrem_ratio),
        "rem": float(eng.rem_ratio),
    }
    base_phase_samples = max(1, len(prompts) * max(1, int(max_new_tokens)))
    total_cycle_samples = max(len(phase_profile), base_phase_samples * len(phase_profile))
    phase_budget = allocate_phase_sample_counts(total_cycle_samples, phase_profile)
    sleep_total = max(phase_profile["nrem"] + phase_profile["rem"], 1e-8)
    phase_sleep_split = {
        "nrem": phase_profile["nrem"] / sleep_total,
        "rem": phase_profile["rem"] / sleep_total,
    }
    cycle_prompts = prioritize_documents_for_prompts(prompts, guard_prompts)
    wake = collect_sleep_batch(
        eng,
        cycle_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        sample_budget=phase_budget["wake"],
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    wake_stats = batch_stats(wake)
    nrem_weight_stats = apply_nrem_weight_update(
        eng,
        wake,
        smooth_eta=0.02 * phase_profile["nrem"],
        plastic_lr=0.01 * phase_profile["nrem"],
    )

    state_nrem, prev_nrem, bias_nrem = fit_decoder_from_batch(
        wake,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
    )
    eng.apply_decoder_refine(prev_nrem.cpu(), state_nrem.cpu(), query_bias=bias_nrem.cpu())
    vocab_nrem = finetune_vocab_head_from_batch(
        eng,
        wake,
        lr=vocab_finetune_lr,
        steps=vocab_finetune_steps,
        batch_size=vocab_finetune_batch_size,
        rem_weight=1.0,
        soft_target_weight=vocab_finetune_soft_target_weight,
    )
    token_nrem = fit_token_head_from_batch(
        wake,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        max_vocab=token_head_max_vocab,
        scale=token_head_scale,
    )
    if token_nrem is not None:
        eng.apply_token_head(
            token_nrem.token_ids,
            state_proj=token_nrem.state_proj,
            prev_proj=token_nrem.prev_proj,
            bias=token_nrem.bias,
            scale=token_nrem.scale,
        )

    nrem = collect_sleep_batch(
        eng,
        cycle_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        sample_budget=phase_budget["nrem"],
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    nrem_stats = batch_stats(nrem)

    rem_snapshot = eng.decoder_snapshot()
    rem_weight_stats = apply_rem_weight_update(
        eng,
        nrem,
        rem_lr=0.005 * phase_profile["rem"],
        noise_scale=max(float(refresh_noise_scale), 0.01) * phase_profile["rem"],
    )
    state_rem, prev_rem, bias_rem = fit_decoder_from_batch(
        nrem,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        rem_weight=rem_weight,
        rem_mix=rem_mix,
    )
    eng.apply_decoder_refine(prev_rem.cpu(), state_rem.cpu(), query_bias=bias_rem.cpu())
    vocab_rem = finetune_vocab_head_from_batch(
        eng,
        nrem,
        lr=vocab_finetune_lr,
        steps=vocab_finetune_steps,
        batch_size=vocab_finetune_batch_size,
        rem_weight=rem_weight,
        soft_target_weight=vocab_finetune_soft_target_weight,
    )
    token_rem = fit_token_head_from_batch(
        nrem,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        rem_weight=rem_weight,
        max_vocab=token_head_max_vocab,
        scale=token_head_scale,
    )
    if token_rem is not None:
        eng.apply_token_head(
            token_rem.token_ids,
            state_proj=token_rem.state_proj,
            prev_proj=token_rem.prev_proj,
            bias=token_rem.bias,
            scale=token_rem.scale,
        )

    pq_stats = None
    if refresh_pq:
        pq_stats = maybe_refresh_pq(
            eng,
            nrem,
            subdim=pq_subdim,
            bits=pq_bits,
            iters=pq_iters,
            batch_size=pq_batch_size,
            sample_size=pq_sample_size,
        )

    rem = collect_sleep_batch(
        eng,
        cycle_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        sample_budget=phase_budget["rem"],
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    rem_stats = batch_stats(rem)
    rem_accepted = True
    if (
        rem_stats["top50_acc"] < nrem_stats["top50_acc"]
        and rem_stats["top1_acc"] < nrem_stats["top1_acc"]
    ):
        eng.restore_decoder_snapshot(rem_snapshot)
        rem = collect_sleep_batch(
            eng,
            cycle_prompts,
            ce_args,
            max_new_tokens=max_new_tokens,
            teacher_topk=teacher_topk,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            sample_budget=phase_budget["rem"],
            context_window=context_window,
            seed_tokens=seed_tokens,
        )
        rem_stats = batch_stats(rem)
        rem_accepted = False

    guard_after = None
    guard_effective = None
    guard_accepted = None
    if guard_prompts:
        guard_after = evaluate_guard_set(
            eng,
            guard_prompts,
            ce_args,
            max_new_tokens=max_new_tokens,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            context_window=context_window,
            seed_tokens=seed_tokens,
        )
        guard_accepted = should_accept_guard_update(
            guard_before,
            guard_after,
            min_top10_delta=guard_min_top10_delta,
            min_top50_delta=guard_min_top50_delta,
            max_top10_drop=guard_max_top10_drop,
            max_top50_drop=guard_max_top50_drop,
            max_phase_grounding_risk_increase=guard_max_phase_grounding_risk_increase,
        )
        if not guard_accepted and guard_snapshot is not None:
            eng.restore_decoder_snapshot(guard_snapshot)
            guard_effective = dict(guard_before)
        else:
            guard_effective = dict(guard_after)

    cycle_applied = guard_accepted is None or guard_accepted
    return {
        "phase_profile": phase_profile,
        "phase_sleep_split": phase_sleep_split,
        "phase_budget": {
            phase: {
                "samples": int(phase_budget[phase]),
                "ratio": float(phase_budget[phase]) / float(total_cycle_samples),
            }
            for phase in phase_profile
        },
        "phase_total_samples": int(total_cycle_samples),
        "wake": wake_stats,
        "nrem": nrem_stats,
        "rem": rem_stats,
        "nrem_weight": nrem_weight_stats if cycle_applied else None,
        "nrem_vocab_head": vocab_nrem if cycle_applied else None,
        "rem_weight": {**rem_weight_stats, "accepted": rem_accepted} if cycle_applied else None,
        "rem_vocab_head": vocab_rem if cycle_applied else None,
        "pq": pq_stats if cycle_applied else None,
        "token_head_vocab": (0 if token_rem is None else int(token_rem.token_ids.numel())) if cycle_applied else 0,
        "guard_before": guard_before,
        "guard_after": guard_after,
        "guard_effective": guard_effective,
        "guard_accepted": guard_accepted,
    }


def run_guarded_microsleep_session(
    eng: CEEngine,
    incoming_prompts: list[str],
    guard_prompts: list[str],
    ce_args,
    *,
    sleep_every: int,
    replay_capacity: int,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
    guard_min_top10_delta: float = 0.0,
    guard_min_top50_delta: float = 0.0,
    guard_max_top10_drop: float = 0.0,
    guard_max_top50_drop: float = 0.0,
    guard_max_phase_grounding_risk_increase: float | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> dict[str, object]:
    buffer = PromptReplayBuffer(capacity=max(1, int(replay_capacity)))
    events: list[dict[str, object]] = []
    accepted = 0
    rejected = 0
    initial_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )

    for idx, prompt in enumerate(incoming_prompts, start=1):
        event = run_guarded_microsleep_step(
            eng,
            buffer,
            prompt,
            guard_prompts,
            ce_args,
            step_index=idx,
            sleep_every=sleep_every,
            max_new_tokens=max_new_tokens,
            teacher_topk=teacher_topk,
            ridge=ridge,
            rem_weight=rem_weight,
            rem_mix=rem_mix,
            token_head_max_vocab=token_head_max_vocab,
            token_head_scale=token_head_scale,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            refresh_pq=refresh_pq,
            pq_subdim=pq_subdim,
            pq_bits=pq_bits,
            pq_iters=pq_iters,
            pq_batch_size=pq_batch_size,
            pq_sample_size=pq_sample_size,
            guard_min_top10_delta=guard_min_top10_delta,
            guard_min_top50_delta=guard_min_top50_delta,
            guard_max_top10_drop=guard_max_top10_drop,
            guard_max_top50_drop=guard_max_top50_drop,
            guard_max_phase_grounding_risk_increase=guard_max_phase_grounding_risk_increase,
            context_window=context_window,
            seed_tokens=seed_tokens,
        )
        if event is None:
            continue
        if event["accepted"]:
            accepted += 1
        else:
            rejected += 1
        events.append(event)

    final_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    return {
        "initial_guard": initial_guard,
        "final_guard": final_guard,
        "accepted": accepted,
        "rejected": rejected,
        "events": events,
        "buffer_size": len(buffer.items()),
    }


def build_prompts(args) -> list[str]:
    prompts = list(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)
    if args.prompt:
        prompts = [args.prompt] + prompts
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt and prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def main():
    ap = argparse.ArgumentParser(description="Sleep refinement for standalone CE artifacts")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--dataset-split", default=DEFAULT_CORPUS_SPLIT)
    ap.add_argument("--dataset-text-column", default=DEFAULT_CORPUS_TEXT_COLUMN)
    ap.add_argument("--doc-limit", type=int, default=256)
    ap.add_argument("--text-limit", type=int, default=1_000_000)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--online-prompts", nargs="*", default=None)
    ap.add_argument("--guard-prompts", nargs="*", default=None)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--sleep-every", type=int, default=4)
    ap.add_argument("--replay-capacity", type=int, default=16)
    ap.add_argument("--label-topk", "--teacher-topk", dest="teacher_topk", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--rem-weight", type=float, default=2.5)
    ap.add_argument("--rem-mix", type=float, default=0.35)
    ap.add_argument("--token-head-max-vocab", type=int, default=2048)
    ap.add_argument("--token-head-scale", type=float, default=1.0)
    ap.add_argument("--guard-min-top10-delta", type=float, default=0.0)
    ap.add_argument("--guard-min-top50-delta", type=float, default=0.0)
    ap.add_argument("--guard-max-top10-drop", type=float, default=0.0)
    ap.add_argument("--guard-max-top50-drop", type=float, default=0.0)
    ap.add_argument("--guard-max-phase-grounding-risk-increase", type=float, default=None)
    ap.add_argument("--refresh-interval", type=int, default=1)
    ap.add_argument("--refresh-steps", type=int, default=48)
    ap.add_argument("--refresh-cb-topk", type=int, default=128)
    ap.add_argument("--refresh-metric-rank", type=int, default=0)
    ap.add_argument("--refresh-noise-scale", type=float, default=0.0)
    ap.add_argument("--context-window", type=int, default=64)
    ap.add_argument("--seed-tokens", type=int, default=8)
    ap.add_argument("--refresh-pq", action="store_true")
    ap.add_argument("--pq-subdim", type=int, default=64)
    ap.add_argument("--pq-bits", type=int, default=8)
    ap.add_argument("--pq-iters", type=int, default=8)
    ap.add_argument("--pq-batch-size", type=int, default=4096)
    ap.add_argument("--pq-sample-size", type=int, default=16384)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    eng = CEEngine(args.engine, device=args.device, backend=args.backend)

    ce_args = argparse.Namespace(
        dt=args.dt,
        cb_weight=None,
        cb_topk=args.cb_topk,
        beta=args.beta,
        steps=args.steps,
        backend=args.backend,
        metric_rank=args.metric_rank,
        lambda0=args.lambda0,
        lambda_phi=args.lambda_phi,
        lambda_var=args.lambda_var,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    safe_print("=== CE Sleep Refinement ===")
    safe_print(f"  engine={args.engine}")
    safe_print(f"  model_source={eng.model_source}")

    if args.online_prompts:
        online_prompts = [prompt for prompt in args.online_prompts if prompt]
        guard_prompts = [prompt for prompt in (args.guard_prompts or list(DEFAULT_PROMPTS)) if prompt]
        safe_print(
            f"  online_prompts={len(online_prompts)}  guard_prompts={len(guard_prompts)}  "
            f"sleep_every={args.sleep_every}"
        )
        session = run_guarded_microsleep_session(
            eng,
            online_prompts,
            guard_prompts,
            ce_args,
            sleep_every=args.sleep_every,
            replay_capacity=args.replay_capacity,
            max_new_tokens=args.tokens,
            teacher_topk=args.teacher_topk,
            ridge=args.ridge,
            rem_weight=args.rem_weight,
            rem_mix=args.rem_mix,
            token_head_max_vocab=args.token_head_max_vocab,
            token_head_scale=args.token_head_scale,
            refresh_interval=args.refresh_interval,
            refresh_steps=args.refresh_steps,
            refresh_cb_topk=args.refresh_cb_topk,
            refresh_metric_rank=args.refresh_metric_rank,
            refresh_noise_scale=args.refresh_noise_scale,
            refresh_pq=args.refresh_pq,
            pq_subdim=args.pq_subdim,
            pq_bits=args.pq_bits,
            pq_iters=args.pq_iters,
            pq_batch_size=args.pq_batch_size,
            pq_sample_size=args.pq_sample_size,
            guard_min_top10_delta=args.guard_min_top10_delta,
            guard_min_top50_delta=args.guard_min_top50_delta,
            guard_max_top10_drop=args.guard_max_top10_drop,
            guard_max_top50_drop=args.guard_max_top50_drop,
            guard_max_phase_grounding_risk_increase=args.guard_max_phase_grounding_risk_increase,
            context_window=args.context_window,
            seed_tokens=args.seed_tokens,
        )
        safe_print(
            f"  microsleep: accepted={session['accepted']}  rejected={session['rejected']}  "
            f"final_guard_top50={session['final_guard']['top50_acc']:.3f}"
        )
        reports = session["events"]
        result_payload = {
            "engine": args.engine,
            "mode": "guarded_microsleep",
            "online_prompts": online_prompts,
            "guard_prompts": guard_prompts,
            "tokens": args.tokens,
            "session": session,
        }
    else:
        if args.data or args.dataset:
            prompts = load_corpus_documents(
                args.data,
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                text_column=args.dataset_text_column,
                doc_limit=args.doc_limit,
                text_limit=args.text_limit,
            )
            safe_print(
                f"  corpus_docs={len(prompts)}  cycles={args.cycles}  tokens={args.tokens}  "
                f"context_window={args.context_window}"
            )
        else:
            prompts = build_prompts(args)
            safe_print(f"  prompts={len(prompts)}  cycles={args.cycles}  tokens={args.tokens}")
        reports = []
        for cycle in range(1, args.cycles + 1):
            report = run_sleep_cycle(
                eng,
                prompts,
                ce_args,
                max_new_tokens=args.tokens,
                teacher_topk=args.teacher_topk,
                ridge=args.ridge,
                rem_weight=args.rem_weight,
                rem_mix=args.rem_mix,
                token_head_max_vocab=args.token_head_max_vocab,
                token_head_scale=args.token_head_scale,
                refresh_interval=args.refresh_interval,
                refresh_steps=args.refresh_steps,
                refresh_cb_topk=args.refresh_cb_topk,
                refresh_metric_rank=args.refresh_metric_rank,
                refresh_noise_scale=args.refresh_noise_scale,
                refresh_pq=args.refresh_pq,
                pq_subdim=args.pq_subdim,
                pq_bits=args.pq_bits,
                pq_iters=args.pq_iters,
                pq_batch_size=args.pq_batch_size,
                pq_sample_size=args.pq_sample_size,
                context_window=args.context_window,
                seed_tokens=args.seed_tokens,
            )
            reports.append(report)
            safe_print(
                f"  cycle {cycle}: "
                f"wake top50={report['wake']['top50_acc']:.3f} -> "
                f"nrem {report['nrem']['top50_acc']:.3f} -> "
                f"rem {report['rem']['top50_acc']:.3f}  "
                f"token_vocab={report['token_head_vocab']}"
            )
        result_payload = {
            "engine": args.engine,
            "mode": "sleep_cycle",
            "prompts": prompts,
            "cycles": args.cycles,
            "reports": reports,
        }

    out_path = args.output or args.engine
    eng.save_runtime_artifact(out_path)
    result_path = os.path.join(os.path.dirname(out_path), "sleep_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    safe_print(f"  saved_engine={out_path}")
    safe_print(f"  saved_report={result_path}")


if __name__ == "__main__":
    main()
