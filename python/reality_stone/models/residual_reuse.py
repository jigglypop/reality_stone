"""Residual reuse helpers for GPT-2 style transformer MLP blocks.

The wrapper is intentionally conservative: it only reuses the MLP residual in
eval/no-grad token decoding paths where ``seq_len == 1``. Attention still runs,
so KV-cache semantics remain owned by the original model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ResidualReuseStats:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    disabled: int = 0
    audit_hits: int = 0
    audit_rel_error_sum: float = 0.0
    audit_rel_error_max: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else float(self.hits) / float(total)

    @property
    def audit_rel_error_mean(self) -> float:
        return 0.0 if self.audit_hits == 0 else self.audit_rel_error_sum / float(self.audit_hits)


class ResidualReuseMLP(nn.Module):
    """Cache and reuse repeated GPT-2 MLP residuals.

    GPT-2 blocks compute ``x = x + mlp(ln_2(x))``. During autoregressive
    decoding many near-identical local states can recur, especially under
    repetition or stable syntax. This module caches the MLP output for a compact
    normalized signature of ``ln_2(x)`` and reuses it when cosine similarity is
    above ``similarity_threshold``.
    """

    def __init__(
        self,
        mlp: nn.Module,
        *,
        similarity_threshold: float = 0.999,
        distance_tolerance: float | None = None,
        match_metric: str = "cosine",
        audit: bool = False,
        audit_return_cached: bool = True,
        max_entries: int = 128,
        signature_dim: int | None = 128,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.similarity_threshold = float(similarity_threshold)
        self.distance_tolerance = None if distance_tolerance is None else float(distance_tolerance)
        self.match_metric = str(match_metric).lower().strip()
        self.audit = bool(audit)
        self.audit_return_cached = bool(audit_return_cached)
        self.max_entries = int(max(1, max_entries))
        self.signature_dim = None if signature_dim is None else int(max(1, signature_dim))
        self.enabled = bool(enabled)
        self.stats = ResidualReuseStats()
        self.register_buffer("_cache_signatures", torch.empty(0), persistent=False)
        self.register_buffer("_cache_values", torch.empty(0), persistent=False)

    def reset_cache(self) -> None:
        device = self._cache_signatures.device
        self._cache_signatures = torch.empty(0, device=device)
        self._cache_values = torch.empty(0, device=device)
        self.stats = ResidualReuseStats()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.stats.calls += 1
        if not self._can_reuse(hidden_states):
            self.stats.disabled += int(hidden_states.shape[0])
            return self.mlp(hidden_states)

        bsz, seq_len, dim = hidden_states.shape
        flat = hidden_states.reshape(bsz, dim)
        signatures = self._signature(flat)
        hit_mask, hit_values = self._lookup(signatures)

        if self.audit and bool(hit_mask.any().item()):
            true_hit = self.mlp(hidden_states[hit_mask]).detach()
            cached_hit = hit_values[hit_mask].to(dtype=true_hit.dtype).reshape_as(true_hit)
            rel = torch.linalg.vector_norm(
                (true_hit - cached_hit).reshape(true_hit.shape[0], -1), dim=-1
            ) / torch.linalg.vector_norm(true_hit.reshape(true_hit.shape[0], -1), dim=-1).clamp_min(1e-8)
            rel_cpu = rel.detach().float().cpu()
            self.stats.audit_hits += int(rel_cpu.numel())
            self.stats.audit_rel_error_sum += float(rel_cpu.sum().item())
            self.stats.audit_rel_error_max = max(
                self.stats.audit_rel_error_max,
                float(rel_cpu.max().item()) if rel_cpu.numel() else 0.0,
            )
            if not self.audit_return_cached:
                hit_values = hit_values.clone()
                hit_values[hit_mask] = true_hit.reshape(-1, true_hit.shape[-1]).to(hit_values.dtype)

        if bool(hit_mask.all().item()):
            self.stats.hits += int(bsz)
            return hit_values.to(dtype=hidden_states.dtype).reshape(bsz, seq_len, dim)

        out = torch.empty_like(hidden_states)
        if bool(hit_mask.any().item()):
            out[hit_mask] = hit_values[hit_mask].to(dtype=hidden_states.dtype).reshape(-1, seq_len, dim)
            self.stats.hits += int(hit_mask.sum().item())

        miss_mask = ~hit_mask
        miss_out = self.mlp(hidden_states[miss_mask])
        out[miss_mask] = miss_out
        self.stats.misses += int(miss_mask.sum().item())
        self._append(signatures[miss_mask], miss_out.detach().reshape(-1, dim))
        return out

    def _can_reuse(self, hidden_states: torch.Tensor) -> bool:
        return (
            self.enabled
            and not self.training
            and not torch.is_grad_enabled()
            and hidden_states.dim() == 3
            and int(hidden_states.shape[1]) == 1
        )

    def _signature(self, flat: torch.Tensor) -> torch.Tensor:
        sig = flat.detach().to(dtype=torch.float32)
        if self.signature_dim is not None and sig.shape[-1] > self.signature_dim:
            sig = sig[..., : self.signature_dim]
        if self.match_metric == "cosine":
            sig = torch.nn.functional.normalize(sig, p=2, dim=-1, eps=1e-8)
        return sig

    def _lookup(self, signatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = signatures.shape[0]
        dim = self.mlp.c_proj.nf if hasattr(self.mlp, "c_proj") and hasattr(self.mlp.c_proj, "nf") else 0
        if self._cache_signatures.numel() == 0:
            values = torch.empty(bsz, max(dim, 1), device=signatures.device, dtype=torch.float32)
            return torch.zeros(bsz, device=signatures.device, dtype=torch.bool), values

        if self.match_metric == "relative_l2":
            diff = signatures[:, None, :] - self._cache_signatures[None, :, :]
            dist = torch.linalg.vector_norm(diff, dim=-1)
            denom = torch.linalg.vector_norm(signatures, dim=-1, keepdim=True).clamp_min(1e-8)
            rel = dist / denom
            best_rel, best_idx = rel.min(dim=-1)
            tol = self.distance_tolerance
            if tol is None:
                tol = max(0.0, 1.0 - self.similarity_threshold)
            hit_mask = best_rel <= float(tol)
        else:
            scores = signatures @ self._cache_signatures.T
            best_scores, best_idx = scores.max(dim=-1)
            hit_mask = best_scores >= self.similarity_threshold
        values = self._cache_values.index_select(0, best_idx)
        return hit_mask, values

    def _append(self, signatures: torch.Tensor, values: torch.Tensor) -> None:
        if signatures.numel() == 0:
            return
        sig = signatures.detach().to(device=values.device, dtype=torch.float32)
        val = values.detach().to(dtype=torch.float32)
        if self._cache_signatures.numel() == 0:
            self._cache_signatures = sig
            self._cache_values = val
        else:
            self._cache_signatures = torch.cat([self._cache_signatures.to(sig.device), sig], dim=0)
            self._cache_values = torch.cat([self._cache_values.to(val.device), val], dim=0)
        if self._cache_signatures.shape[0] > self.max_entries:
            self._cache_signatures = self._cache_signatures[-self.max_entries :]
            self._cache_values = self._cache_values[-self.max_entries :]


def install_gpt2_residual_reuse(
    model: nn.Module,
    *,
    similarity_threshold: float = 0.999,
    distance_tolerance: float | None = None,
    match_metric: str = "cosine",
    audit: bool = False,
    audit_return_cached: bool = True,
    max_entries: int = 128,
    signature_dim: int | None = 128,
    enabled: bool = True,
) -> list[ResidualReuseMLP]:
    """Wrap GPT-2 ``transformer.h[*].mlp`` modules with residual reuse.

    Returns the installed wrappers so callers can inspect hit-rate stats or
    reset caches between prompts.
    """

    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise ValueError("expected a GPT-2 style model with model.transformer.h")

    wrappers: list[ResidualReuseMLP] = []
    for block in blocks:
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue
        if isinstance(mlp, ResidualReuseMLP):
            wrapper = mlp
            wrapper.enabled = bool(enabled)
            wrapper.similarity_threshold = float(similarity_threshold)
            wrapper.distance_tolerance = (
                None if distance_tolerance is None else float(distance_tolerance)
            )
            wrapper.match_metric = str(match_metric).lower().strip()
            wrapper.audit = bool(audit)
            wrapper.audit_return_cached = bool(audit_return_cached)
            wrapper.max_entries = int(max(1, max_entries))
            wrapper.signature_dim = None if signature_dim is None else int(max(1, signature_dim))
        else:
            wrapper = ResidualReuseMLP(
                mlp,
                similarity_threshold=similarity_threshold,
                distance_tolerance=distance_tolerance,
                match_metric=match_metric,
                audit=audit,
                audit_return_cached=audit_return_cached,
                max_entries=max_entries,
                signature_dim=signature_dim,
                enabled=enabled,
            )
            wrapper.train(bool(mlp.training))
            block.mlp = wrapper
        wrappers.append(wrapper)
    return wrappers


def reset_residual_reuse(model_or_wrappers: nn.Module | list[ResidualReuseMLP]) -> None:
    if isinstance(model_or_wrappers, list):
        wrappers = model_or_wrappers
    else:
        wrappers = [
            module for module in model_or_wrappers.modules() if isinstance(module, ResidualReuseMLP)
        ]
    for wrapper in wrappers:
        wrapper.reset_cache()


def residual_reuse_report(wrappers: list[ResidualReuseMLP]) -> dict[str, Any]:
    hits = sum(w.stats.hits for w in wrappers)
    misses = sum(w.stats.misses for w in wrappers)
    disabled = sum(w.stats.disabled for w in wrappers)
    total = hits + misses
    return {
        "layers": len(wrappers),
        "hits": hits,
        "misses": misses,
        "disabled": disabled,
        "hit_rate": 0.0 if total == 0 else float(hits) / float(total),
        "audit_hits": sum(w.stats.audit_hits for w in wrappers),
        "audit_rel_error_mean": (
            0.0
            if sum(w.stats.audit_hits for w in wrappers) == 0
            else sum(w.stats.audit_rel_error_sum for w in wrappers)
            / float(sum(w.stats.audit_hits for w in wrappers))
        ),
        "audit_rel_error_max": max((w.stats.audit_rel_error_max for w in wrappers), default=0.0),
    }
