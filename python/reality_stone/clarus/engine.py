"""CE-AGI runtime aligned with the Equation spec.

Runtime mode is standalone-only and must remain teacher-free.
Any artifact carrying a teacher clone is rejected at load time.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .ce_ops import (
        build_metric_basis as ce_build_metric_basis,
        pack_sparse as ce_pack_sparse,
        pq_reconstruct_tokens,
        pq_scores,
        relax_packed as ce_relax_packed,
    )
    from .constants import BYPASS
    from .research import phase_grounding_suppression
    from .utils import safe_print, normalize_vector, resolve_device
except ImportError:
    from reality_stone.clarus.ce_ops import (
        build_metric_basis as ce_build_metric_basis,
        pack_sparse as ce_pack_sparse,
        pq_reconstruct_tokens,
        pq_scores,
        relax_packed as ce_relax_packed,
    )
    from reality_stone.clarus.constants import BYPASS
    from reality_stone.clarus.research import phase_grounding_suppression
    from reality_stone.clarus.utils import safe_print, normalize_vector, resolve_device

import re as _re

_REPEATED_CHAR = _re.compile(r"(.)\1{4,}")
_REPEATED_WORD = _re.compile(r"((?:\S{2,}\s*){1,3}?)(?:\s*\1){3,}")
_MULTI_SPACE = _re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = _re.compile(r"\n{3,}")


def postprocess_output(text: str) -> str:
    """Normalize generated text from the standalone CE decoder path.

    Collapses character-level and token-level repetition, trims whitespace,
    and limits newlines so outputs from different decoders look consistent.
    """
    text = text.strip()
    text = _REPEATED_CHAR.sub(lambda m: m.group(1) * 3, text)
    text = _REPEATED_WORD.sub(lambda m: m.group(1), text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text


DEFAULT_PROMPTS = (
    "인공지능의 미래는",
    "오늘 날씨가",
    "한국어로 대답해줘",
)


def _rounded_count(total: int, ratio: float) -> int:
    return int(math.floor(max(float(ratio), 0.0) * float(total) + 0.5))


def update_phi(phi: torch.Tensor, m_star: torch.Tensor, phi_var: torch.Tensor | None = None) -> torch.Tensor:
    v = normalize_vector(m_star)
    if phi_var is not None and phi_var.numel() == phi.numel():
        var_mean = phi_var.mean().clamp(min=1e-8)
        alpha = (BYPASS * var_mean / (var_mean + 1.0)).item()
    else:
        alpha = BYPASS
    return (1 - alpha) * phi + alpha * v


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_optional(value) -> str:
    value = _optional_float(value)
    return "n/a" if value is None else f"{value:.4f}"


def state_partition_counts(dim: int, active_ratio: float, struct_ratio: float) -> tuple[int, int, int]:
    dim = max(0, int(dim))
    if dim == 0:
        return 0, 0, 0

    active_ratio = max(float(active_ratio), 0.0)
    struct_ratio = max(float(struct_ratio), 0.0)
    active_k = min(dim, max(1, _rounded_count(dim, active_ratio)))
    remaining = max(dim - active_k, 0)
    if remaining == 0:
        return active_k, 0, 0

    background_ratio = max(0.0, 1.0 - active_ratio - struct_ratio)
    non_active_total = struct_ratio + background_ratio
    if non_active_total <= 1e-8:
        return active_k, remaining, 0

    struct_only_ratio = struct_ratio / non_active_total
    struct_only_k = min(remaining, max(0, _rounded_count(remaining, struct_only_ratio)))
    background_k = remaining - struct_only_k
    return active_k, struct_only_k, background_k


@dataclass
class PromptContext:
    prompt: str
    prompt_ids: torch.Tensor
    h_true: torch.Tensor | None
    m0: torch.Tensor
    phi: torch.Tensor
    best_layer: int
    layer_scores: dict[int, float]


class CEEngine:
    def __init__(self, path: str, device: str = "cpu", backend: str = "torch"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.data = data
        self.device = resolve_device(device)
        self.backend = backend

        self.model_name = data["model_name"]
        self.d = int(data["d"])
        self.vocab = int(data["vocab"])
        self.tau = float(data["tau"])
        self.portal = float(data["portal"])
        self.bypass = float(data["bypass"])
        self.t_wake = float(data["t_wake"])
        self.n_layer = int(data["n_layer"])
        self.h_norm_ref = float(data.get("hidden_norm_ref", 50.0))
        self.decoder_prev_scale = float(data.get("decoder_prev_scale", 0.35))
        self.decoder_prev_proj = None
        if data.get("decoder_prev_proj") is not None:
            self.decoder_prev_proj = data["decoder_prev_proj"].float().to(self.device)
        self.decoder_state_proj = None
        if data.get("decoder_state_proj") is not None:
            self.decoder_state_proj = data["decoder_state_proj"].float().to(self.device)
        self.decoder_query_bias = None
        if data.get("decoder_query_bias") is not None:
            self.decoder_query_bias = data["decoder_query_bias"].float().to(self.device)
        self.decoder_vocab_weight = None
        if data.get("decoder_vocab_weight") is not None:
            self.decoder_vocab_weight = data["decoder_vocab_weight"].float().to(self.device)
        self.decoder_vocab_bias = None
        if data.get("decoder_vocab_bias") is not None:
            self.decoder_vocab_bias = data["decoder_vocab_bias"].float().to(self.device)
        self.decoder_vocab_scale = float(data.get("decoder_vocab_scale", 1.0))
        self.decoder_token_ids = None
        if data.get("decoder_token_ids") is not None:
            self.decoder_token_ids = data["decoder_token_ids"].long().to(self.device)
        self.decoder_token_state_proj = None
        if data.get("decoder_token_state_proj") is not None:
            self.decoder_token_state_proj = data["decoder_token_state_proj"].float().to(self.device)
        self.decoder_token_prev_proj = None
        if data.get("decoder_token_prev_proj") is not None:
            self.decoder_token_prev_proj = data["decoder_token_prev_proj"].float().to(self.device)
        self.decoder_token_bias = None
        if data.get("decoder_token_bias") is not None:
            self.decoder_token_bias = data["decoder_token_bias"].float().to(self.device)
        self.decoder_token_scale = float(data.get("decoder_token_scale", 1.0))
        self.decoder_query_blend = float(data.get("decoder_query_blend", 0.7))
        self.decoder_candidate_ratio = float(data.get("decoder_candidate_ratio", 0.04865))
        self.curvature_alpha = float(data.get("curvature_alpha", 1.5))
        self.curvature_lambda = float(data.get("curvature_lambda", 1.25))
        self.curvature_steepness = float(data.get("curvature_steepness", 8.0))
        self.curvature_eval_topk = int(data.get("curvature_eval_topk", 256))
        self.phase_grounding_lambda = float(data.get("phase_grounding_lambda", 0.25))
        self.data["phase_grounding_lambda"] = float(self.phase_grounding_lambda)
        self.repeat_window = int(data.get("repeat_window", 16))
        self.repeat_ngram = int(data.get("repeat_ngram", 3))
        self._terminal_ids_cache = None
        self.active_ratio = float(data.get("active_ratio", 0.0487))
        self.struct_ratio = float(data.get("struct_ratio", 0.2623))
        self.wake_ratio = float(data.get("wake_ratio", 0.6891))
        self.nrem_ratio = float(data.get("nrem_ratio", 0.2623))
        self.rem_ratio = float(data.get("rem_ratio", 0.0487))
        self.target_w_density = float(data.get("target_w_density", 0.0316))
        self.sparsity_radius = float(data.get("r_c", math.pi))
        self.active_dim_mask = None
        if data.get("active_dim_mask") is not None:
            self.active_dim_mask = data["active_dim_mask"].bool().to(self.device)
        self.struct_dim_mask = None
        if data.get("struct_dim_mask") is not None:
            self.struct_dim_mask = data["struct_dim_mask"].bool().to(self.device)
        self.background_dim_mask = None
        if data.get("background_dim_mask") is not None:
            self.background_dim_mask = data["background_dim_mask"].bool().to(self.device)
        self._state_graph_laplacian = None
        self._state_coords = None

        self.W = data["W"].float().to(self.device)
        self.W_pack = self._load_w_pack(data)
        self._dense_relax_w = None
        if self.W_pack[0].numel() == self.W.numel():
            self._dense_relax_w = self.W
        emb_weight = data.get("emb_weight")
        self.emb = emb_weight.float().to(self.device) if emb_weight is not None else None
        # Vocab pruning (V1): emb stores top-K rows in compact id space [0, K).
        # vocab_id_map maps a global tokenizer id -> compact id (or -1 if pruned).
        # kept_token_ids[k] is the global id of compact row k. unk_emb is the
        # fallback embedding used when a global id has been pruned.
        self.kept_token_ids = None
        self.vocab_id_map = None
        self.unk_emb = None
        if data.get("kept_token_ids") is not None and data.get("vocab_id_map") is not None:
            self.kept_token_ids = data["kept_token_ids"].long().to(self.device)
            self.vocab_id_map = data["vocab_id_map"].long().to(self.device)
            if data.get("pruned_unk_emb") is not None:
                self.unk_emb = data["pruned_unk_emb"].float().to(self.device)
            elif self.emb is not None:
                self.unk_emb = self.emb.mean(dim=0)
        pos_weight = data.get("pos_weight")
        self.pos = pos_weight.float().to(self.device) if pos_weight is not None else None
        self.ln_w = data["ln_f_weight"].float().to(self.device)
        self.ln_b = data["ln_f_bias"].float().to(self.device)
        self.context_first_proj = None
        if data.get("context_first_proj") is not None:
            self.context_first_proj = data["context_first_proj"].float().to(self.device)
        self.context_prev_proj = None
        if data.get("context_prev_proj") is not None:
            self.context_prev_proj = data["context_prev_proj"].float().to(self.device)
        self.context_last_proj = None
        if data.get("context_last_proj") is not None:
            self.context_last_proj = data["context_last_proj"].float().to(self.device)
        self.context_mean_proj = None
        if data.get("context_mean_proj") is not None:
            self.context_mean_proj = data["context_mean_proj"].float().to(self.device)
        self.context_decay_proj = None
        if data.get("context_decay_proj") is not None:
            self.context_decay_proj = data["context_decay_proj"].float().to(self.device)
        self.context_phi_proj = None
        if data.get("context_phi_proj") is not None:
            self.context_phi_proj = data["context_phi_proj"].float().to(self.device)
        self.context_len_proj = None
        if data.get("context_len_proj") is not None:
            self.context_len_proj = data["context_len_proj"].float().to(self.device)
        self.context_bias = None
        if data.get("context_bias") is not None:
            self.context_bias = data["context_bias"].float().to(self.device)
        self.allow_pretrained_fallback = bool(data.get("allow_pretrained_fallback", True))
        self.pq_centroids = None
        self.pq_codes = None
        if data.get("pq_centroids") is not None and data.get("pq_codes") is not None:
            self.pq_centroids = data["pq_centroids"].to(self.device)
            self.pq_codes = data["pq_codes"].to(self.device)

        self._stored_eigvecs = data.get("W_eigvecs")
        if self._stored_eigvecs is not None:
            self._stored_eigvecs = self._stored_eigvecs.float()
        self._eigvec_cache: dict[int, torch.Tensor] = {}
        if self.target_w_density > 0.0:
            self.apply_relax_matrix(self.W.detach().cpu())

        self.model = None
        self.tok = None
        self.pad_token_id = data.get("pad_token_id")
        self.eos_token_id = data.get("eos_token_id")
        self.model_memory_bytes = 0
        self._load_model()
        if self.active_dim_mask is None or self.struct_dim_mask is None:
            seed = None
            if self.decoder_state_proj is not None:
                seed = self.decoder_state_proj.abs().mean(dim=1)
            elif self.W is not None:
                seed = self.W.abs().mean(dim=1)
            if seed is not None:
                active_mask, struct_mask, _ = self.state_partition(seed, use_stored=False)
                self.apply_state_partition(active_mask, struct_mask)
        self._compress_runtime_projections()

    def _load_w_pack(self, data):
        values = data.get("W_values")
        col_idx = data.get("W_col_idx")
        row_ptr = data.get("W_row_ptr")
        if values is None or col_idx is None or row_ptr is None:
            values, col_idx, row_ptr = ce_pack_sparse(data["W"].float(), backend="torch")
        return (
            values.to(self.device),
            col_idx.to(self.device),
            row_ptr.to(self.device),
        )

    def _load_model(self):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        clone_config = self.data.get("clone_config")
        clone_state = self.data.get("clone_state")
        clone_kind = self.data.get("clone_kind")
        tokenizer_json = self.data.get("tokenizer_json")
        tokenizer_specials = self.data.get("tokenizer_specials") or {}

        if tokenizer_json is not None:
            backend_tok = Tokenizer.from_str(tokenizer_json)
            tok_kwargs = {k: v for k, v in tokenizer_specials.items() if v is not None}
            self.tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, **tok_kwargs)
        else:
            raise RuntimeError("runtime artifact must embed tokenizer_json and may not fall back to pretrained assets")

        if clone_config is not None or clone_state is not None or clone_kind is not None:
            raise RuntimeError("teacher-bearing artifact is forbidden in runtime mode; rebuild as runtime-only")
        if self.allow_pretrained_fallback:
            raise RuntimeError("pretrained fallback is forbidden in runtime mode")

        self.model = None
        self.model_source = "runtime"
        self.model_memory_bytes = 0
        if self.eos_token_id is None:
            self.eos_token_id = self.tok.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tok.pad_token_id
        self.inject_layer = self.n_layer // 2

    def _build_state_graph_laplacian(self) -> torch.Tensor:
        coords = self.state_coords()
        dist = torch.cdist(coords, coords)
        adj = (dist > 0) & (dist <= 1.01)
        deg = adj.sum(dim=1).float()
        lap = -adj.float()
        lap[torch.arange(self.d, device=self.device), torch.arange(self.d, device=self.device)] = deg
        return lap

    def _build_state_coords(self) -> torch.Tensor:
        side = int(math.ceil(self.d ** (1.0 / 3.0)))
        idx = torch.arange(self.d, device=self.device, dtype=torch.long)
        x = idx // (side * side)
        y = (idx // side) % side
        z = idx % side
        return torch.stack([x, y, z], dim=1).float()

    def state_coords(self) -> torch.Tensor:
        if self._state_coords is None:
            self._state_coords = self._build_state_coords()
        return self._state_coords

    def state_graph_laplacian(self) -> torch.Tensor:
        if self._state_graph_laplacian is None:
            self._state_graph_laplacian = self._build_state_graph_laplacian()
        return self._state_graph_laplacian

    def weight_density(self, w: torch.Tensor | None = None) -> float:
        mat = self.W if w is None else w
        dim = int(mat.shape[0])
        if dim <= 1:
            return 0.0
        off = mat.detach().clone()
        off.fill_diagonal_(0)
        return float((off != 0).sum().item()) / float(dim * (dim - 1))

    def resparsify_relax_matrix(
        self,
        w: torch.Tensor,
        *,
        target_density: float | None = None,
        radius: float | None = None,
    ) -> torch.Tensor:
        target = self.target_w_density if target_density is None else float(target_density)
        if target <= 0.0 or target >= 1.0:
            return 0.5 * (w + w.T)

        w_cpu = 0.5 * (w.detach().cpu().float() + w.detach().cpu().float().T)
        dim = int(w_cpu.shape[0])
        diag = torch.diag(w_cpu).clone()
        radius_val = self.sparsity_radius if radius is None else float(radius)
        coords = self.state_coords().detach().cpu()
        dist = torch.cdist(coords, coords)
        candidate = (dist > 0) & (dist <= radius_val)
        upper = torch.triu(candidate, diagonal=1)
        pair_idx = upper.nonzero(as_tuple=False)
        if pair_idx.numel() == 0:
            sparse = torch.diag(diag)
            return sparse

        keep_edges = max(1, int(round(target * dim * (dim - 1) / 2)))
        keep_edges = min(keep_edges, int(pair_idx.shape[0]))
        pair_scores = w_cpu[pair_idx[:, 0], pair_idx[:, 1]].abs()
        top_idx = torch.topk(pair_scores, keep_edges).indices
        chosen = pair_idx.index_select(0, top_idx)
        mask = torch.zeros_like(w_cpu, dtype=torch.bool)
        mask[chosen[:, 0], chosen[:, 1]] = True
        mask = mask | mask.T
        sparse = torch.zeros_like(w_cpu)
        sparse[mask] = w_cpu[mask]
        sparse[torch.arange(dim), torch.arange(dim)] = diag
        return sparse

    def apply_relax_matrix(self, w: torch.Tensor):
        w_cpu = w.detach().cpu().float()
        w_sym = self.resparsify_relax_matrix(0.5 * (w_cpu + w_cpu.T))
        eigvals = torch.linalg.eigvalsh(w_sym)
        lam_max = float(eigvals[-1].item())
        if lam_max >= -1e-4:
            shift = lam_max + 1e-3
            w_sym = w_sym - shift * torch.eye(w_sym.shape[0], dtype=w_sym.dtype)
        self.data["W"] = w_sym
        values, col_idx, row_ptr = ce_pack_sparse(w_sym, backend="torch")
        self.data["W_values"] = values.cpu()
        self.data["W_col_idx"] = col_idx.cpu()
        self.data["W_row_ptr"] = row_ptr.cpu()
        self.data["W_eigvecs"] = None
        self.W = w_sym.to(self.device)
        self.W_pack = (
            values.to(self.device),
            col_idx.to(self.device),
            row_ptr.to(self.device),
        )
        self._dense_relax_w = self.W if values.numel() == self.W.numel() else None
        self._stored_eigvecs = None
        self._eigvec_cache.clear()

    def build_brain_runtime(
        self,
        *,
        active_ratio: float | None = None,
        backend: str | None = None,
    ):
        from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig

        runtime_cfg = BrainRuntimeConfig(
            dim=self.d,
            active_ratio=self.active_ratio if active_ratio is None else float(active_ratio),
        )
        runtime_backend = "auto" if backend is None else backend
        return BrainRuntime(
            self.W.detach().cpu(),
            config=runtime_cfg,
            backend=runtime_backend,
            device=self.device,
        )

    def apply_state_partition(
        self,
        active_mask: torch.Tensor,
        struct_mask: torch.Tensor,
    ):
        active_mask = active_mask.bool().detach().cpu()
        struct_mask = struct_mask.bool().detach().cpu()
        if active_mask.shape != struct_mask.shape:
            raise ValueError("state partition masks must share shape")
        struct_mask = struct_mask | active_mask
        background_mask = ~(struct_mask)
        self.data["active_dim_mask"] = active_mask
        self.data["struct_dim_mask"] = struct_mask
        self.data["background_dim_mask"] = background_mask
        self.data["active_ratio"] = float(self.active_ratio)
        self.data["struct_ratio"] = float(self.struct_ratio)
        self.active_dim_mask = active_mask.to(self.device)
        self.struct_dim_mask = struct_mask.to(self.device)
        self.background_dim_mask = background_mask.to(self.device)

    def active_indices(self) -> torch.Tensor | None:
        if self.active_dim_mask is None:
            return None
        idx = torch.nonzero(self.active_dim_mask, as_tuple=False).squeeze(1)
        return idx if idx.numel() else None

    def struct_indices(self) -> torch.Tensor | None:
        if self.struct_dim_mask is None:
            return self.active_indices()
        idx = torch.nonzero(self.struct_dim_mask, as_tuple=False).squeeze(1)
        return idx if idx.numel() else None

    def _projection_indices(self) -> torch.Tensor | None:
        idx = self.struct_indices()
        return idx if idx is not None else self.active_indices()

    def _compress_state_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:
        if proj is None:
            return None
        proj_idx = self._projection_indices()
        if proj_idx is None:
            return proj.float().to(self.device)
        if proj.ndim == 2 and proj.shape[0] == self.d and proj.shape[1] == self.d:
            idx = proj_idx.to(proj.device)
            return proj.index_select(0, idx).index_select(1, idx).float().to(self.device)
        return proj.float().to(self.device)

    def _compress_prev_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:
        if proj is None:
            return None
        proj_idx = self._projection_indices()
        if proj_idx is None:
            return proj.float().to(self.device)
        if proj.ndim == 2 and proj.shape[0] == self.d and proj.shape[1] == self.d:
            idx = proj_idx.to(proj.device)
            return proj.index_select(1, idx).float().to(self.device)
        return proj.float().to(self.device)

    def _compress_token_state_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:
        if proj is None:
            return None
        proj_idx = self._projection_indices()
        if proj_idx is None:
            return proj.float().to(self.device)
        if proj.ndim == 2 and proj.shape[0] == self.d:
            idx = proj_idx.to(proj.device)
            return proj.index_select(0, idx).float().to(self.device)
        return proj.float().to(self.device)

    def _compress_runtime_projections(self):
        if self.decoder_state_proj is not None:
            if self.decoder_state_proj.shape[0] == self.d and self.decoder_state_proj.shape[1] == self.d:
                self.decoder_state_proj = self.decoder_state_proj.float().to(self.device)
            else:
                self.decoder_state_proj = self._compress_state_proj(self.decoder_state_proj.detach().cpu())
                self.data["decoder_state_proj"] = self.decoder_state_proj.detach().cpu()
        if self.decoder_prev_proj is not None:
            if self.decoder_prev_proj.shape[0] == self.d and self.decoder_prev_proj.shape[1] == self.d:
                self.decoder_prev_proj = self.decoder_prev_proj.float().to(self.device)
            else:
                self.decoder_prev_proj = self._compress_prev_proj(self.decoder_prev_proj.detach().cpu())
                self.data["decoder_prev_proj"] = self.decoder_prev_proj.detach().cpu()
        if self.decoder_token_state_proj is not None:
            self.decoder_token_state_proj = self._compress_token_state_proj(self.decoder_token_state_proj.detach().cpu())
            self.data["decoder_token_state_proj"] = self.decoder_token_state_proj.detach().cpu()

    def state_partition(
        self,
        x: torch.Tensor,
        *,
        use_stored: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_stored and self.active_dim_mask is not None and self.struct_dim_mask is not None:
            active_mask = self.active_dim_mask
            struct_mask = self.struct_dim_mask
            background_mask = (
                self.background_dim_mask
                if self.background_dim_mask is not None
                else ~struct_mask
            )
            return active_mask, struct_mask, background_mask

        dim = x.numel()
        active_k, struct_only_k, _ = state_partition_counts(dim, self.active_ratio, self.struct_ratio)
        struct_k = min(dim, active_k + struct_only_k)
        scores = x.detach().abs()
        active_idx = torch.topk(scores, active_k).indices
        struct_idx = torch.topk(scores, struct_k).indices
        active_mask = torch.zeros(dim, dtype=torch.bool, device=x.device)
        struct_mask = torch.zeros(dim, dtype=torch.bool, device=x.device)
        active_mask[active_idx] = True
        struct_mask[struct_idx] = True
        struct_mask = struct_mask | active_mask
        background_mask = ~struct_mask
        return active_mask, struct_mask, background_mask

    def masked_state(
        self,
        x: torch.Tensor,
        *,
        include_struct: bool = False,
        use_stored: bool = True,
    ) -> torch.Tensor:
        active_mask, struct_mask, _ = self.state_partition(x, use_stored=use_stored)
        mask = struct_mask if include_struct else active_mask
        return x * mask.to(dtype=x.dtype)

    def _project_state_query(self, state_hidden: torch.Tensor) -> torch.Tensor:
        proj_idx = self._projection_indices()
        if (
            self.decoder_state_proj is not None
            and proj_idx is not None
            and self.decoder_state_proj.ndim == 2
            and self.decoder_state_proj.shape[0] == proj_idx.numel()
            and self.decoder_state_proj.shape[1] == proj_idx.numel()
        ):
            state_in = state_hidden.index_select(-1, proj_idx)
            state_out = state_in @ self.decoder_state_proj
            query = torch.zeros_like(state_hidden)
            query[..., proj_idx] = state_out
            return query
        if self.decoder_state_proj is not None:
            return state_hidden @ self.decoder_state_proj
        return state_hidden

    def _project_prev_query(self, prev_emb: torch.Tensor) -> torch.Tensor:
        proj_idx = self._projection_indices()
        if (
            self.decoder_prev_proj is not None
            and proj_idx is not None
            and self.decoder_prev_proj.ndim == 2
            and self.decoder_prev_proj.shape[0] == self.d
            and self.decoder_prev_proj.shape[1] == proj_idx.numel()
        ):
            prev_out = prev_emb @ self.decoder_prev_proj
            query = torch.zeros_like(prev_emb)
            query[..., proj_idx] = prev_out
            return query
        if self.decoder_prev_proj is not None:
            return prev_emb @ self.decoder_prev_proj
        return prev_emb

    def _get_w_eigvecs(self, metric_rank: int) -> torch.Tensor | None:
        hess_rank = min(metric_rank // 2, 8)
        if hess_rank <= 0:
            return None
        if self._stored_eigvecs is not None and self._stored_eigvecs.shape[0] >= hess_rank:
            return self._stored_eigvecs[:hess_rank].to(self.device)
        if hess_rank not in self._eigvec_cache:
            _, eigvecs = torch.linalg.eigh(self.W.cpu())
            self._eigvec_cache[hess_rank] = eigvecs[:, :hess_rank].T.contiguous().to(self.device)
        return self._eigvec_cache[hess_rank]

    def memory_usage(self) -> dict[str, float]:
        values, col_idx, row_ptr = self.W_pack
        w_dense_bytes = self.W.numel() * self.W.element_size()
        w_packed_bytes = (
            values.numel() * values.element_size()
            + col_idx.numel() * col_idx.element_size()
            + row_ptr.numel() * row_ptr.element_size()
        )
        w_layers_bytes = sum(
            w.numel() * w.element_size() for w in self.data.get("W_layers", [])
        )
        emb_bytes = 0 if self.emb is None else self.emb.numel() * self.emb.element_size()
        pos_bytes = 0 if self.pos is None else self.pos.numel() * self.pos.element_size()
        pq_bytes = 0
        if self.pq_centroids is not None and self.pq_codes is not None:
            pq_bytes = (
                self.pq_centroids.numel() * self.pq_centroids.element_size()
                + self.pq_codes.numel() * self.pq_codes.element_size()
            )
        clone_artifact_bytes = 0
        clone_state = self.data.get("clone_state")
        if clone_state is not None:
            clone_artifact_bytes = sum(
                value.numel() * value.element_size() for value in clone_state.values()
            )
        prev_proj_bytes = (
            0 if self.decoder_prev_proj is None
            else self.decoder_prev_proj.numel() * self.decoder_prev_proj.element_size()
        )
        state_proj_bytes = (
            0 if self.decoder_state_proj is None
            else self.decoder_state_proj.numel() * self.decoder_state_proj.element_size()
        )
        query_bias_bytes = (
            0 if self.decoder_query_bias is None
            else self.decoder_query_bias.numel() * self.decoder_query_bias.element_size()
        )
        vocab_head_bytes = 0
        for tensor in (self.decoder_vocab_weight, self.decoder_vocab_bias):
            if tensor is not None:
                vocab_head_bytes += tensor.numel() * tensor.element_size()
        context_bytes = 0
        for tensor in (
            self.context_first_proj,
            self.context_prev_proj,
            self.context_last_proj,
            self.context_mean_proj,
            self.context_decay_proj,
            self.context_phi_proj,
            self.context_len_proj,
            self.context_bias,
        ):
            if tensor is not None:
                context_bytes += tensor.numel() * tensor.element_size()
        token_head_bytes = 0
        for tensor in (
            self.decoder_token_ids,
            self.decoder_token_state_proj,
            self.decoder_token_prev_proj,
            self.decoder_token_bias,
        ):
            if tensor is not None:
                token_head_bytes += tensor.numel() * tensor.element_size()
        partition_bytes = 0
        for tensor in (
            self.active_dim_mask,
            self.struct_dim_mask,
            self.background_dim_mask,
        ):
            if tensor is not None:
                partition_bytes += tensor.numel() * tensor.element_size()
        ln_bytes = (self.ln_w.numel() + self.ln_b.numel()) * self.ln_w.element_size()
        runtime_core = (
            w_packed_bytes
            + ln_bytes
            + context_bytes
            + prev_proj_bytes
            + state_proj_bytes
            + query_bias_bytes
            + vocab_head_bytes
            + token_head_bytes
            + partition_bytes
        )
        runtime_total = runtime_core + emb_bytes + pos_bytes + pq_bytes
        file_core = (
            w_dense_bytes
            + w_layers_bytes
            + ln_bytes
            + context_bytes
            + prev_proj_bytes
            + state_proj_bytes
            + query_bias_bytes
            + vocab_head_bytes
            + token_head_bytes
            + partition_bytes
        )
        file_total = file_core + emb_bytes + pos_bytes + pq_bytes + clone_artifact_bytes
        return {
            "W_dense_MB": w_dense_bytes / 1024 / 1024,
            "W_packed_MB": w_packed_bytes / 1024 / 1024,
            "W_layers_MB": w_layers_bytes / 1024 / 1024,
            "W_target_density_pct": self.target_w_density * 100.0,
            "W_offdiag_density_pct": self.weight_density() * 100.0,
            "Embedding_MB": emb_bytes / 1024 / 1024,
            "Positional_MB": pos_bytes / 1024 / 1024,
            "PQ_MB": pq_bytes / 1024 / 1024,
            "CloneArtifact_MB": clone_artifact_bytes / 1024 / 1024,
            "ContextProj_MB": context_bytes / 1024 / 1024,
            "PrevProj_MB": prev_proj_bytes / 1024 / 1024,
            "StateProj_MB": state_proj_bytes / 1024 / 1024,
            "QueryBias_MB": query_bias_bytes / 1024 / 1024,
            "VocabHead_MB": vocab_head_bytes / 1024 / 1024,
            "TokenHead_MB": token_head_bytes / 1024 / 1024,
            "StateMask_KB": partition_bytes / 1024,
            "ln_f_KB": ln_bytes / 1024,
            "runtime_core_MB": runtime_core / 1024 / 1024,
            "runtime_total_MB": runtime_total / 1024 / 1024,
            "file_total_MB": file_total / 1024 / 1024,
            "model_MB": self.model_memory_bytes / 1024 / 1024,
        }

    def save_artifact(self, path: str):
        torch.save(self.data, path)

    def save_runtime_artifact(self, path: str):
        runtime = dict(self.data)
        for key in ("clone_state", "clone_config", "clone_kind"):
            runtime.pop(key, None)
        runtime["allow_pretrained_fallback"] = False
        torch.save(runtime, path)

    def has_standalone_lexicon(self) -> bool:
        return self.emb is not None or (
            self.pq_centroids is not None and self.pq_codes is not None
        )

    def prompt_embeddings(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        token_ids = prompt_ids.to(device=self.device, dtype=torch.long).view(-1)
        emb = self.token_embedding(token_ids).view(prompt_ids.shape[1], self.d)
        if self.pos is not None:
            pos_idx = torch.arange(prompt_ids.shape[1], device=self.device, dtype=torch.long)
            pos_idx = pos_idx.clamp_max(self.pos.shape[0] - 1)
            emb = emb + self.pos.index_select(0, pos_idx)
        return emb

    def runtime_prompt_state(
        self,
        prompt_ids: torch.Tensor,
        *,
        phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb_seq = self.prompt_embeddings(prompt_ids)
        first_emb = emb_seq[0]
        prev_emb = emb_seq[-2] if emb_seq.shape[0] > 1 else emb_seq[-1]
        last_emb = emb_seq[-1]
        mean_emb = emb_seq.mean(dim=0)
        weights = torch.arange(1, emb_seq.shape[0] + 1, device=self.device, dtype=emb_seq.dtype).unsqueeze(1)
        decay_emb = (emb_seq * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
        phi_base = (
            torch.zeros_like(last_emb)
            if emb_seq.shape[0] <= 1
            else normalize_vector(emb_seq[:-1].mean(dim=0) - emb_seq[-1])
        )
        len_ratio = float(min(emb_seq.shape[0], 0 if self.pos is None else self.pos.shape[0]) or emb_seq.shape[0])
        if self.pos is not None and self.pos.shape[0] > 0:
            len_ratio /= float(self.pos.shape[0])
        else:
            len_ratio = 1.0
        state = torch.zeros_like(last_emb) if self.context_bias is None else self.context_bias.clone()
        if self.context_first_proj is not None:
            state = state + first_emb @ self.context_first_proj
        if self.context_prev_proj is not None:
            state = state + prev_emb @ self.context_prev_proj
        if self.context_last_proj is not None:
            state = state + last_emb @ self.context_last_proj
        else:
            state = state + last_emb
        if self.context_mean_proj is not None:
            state = state + mean_emb @ self.context_mean_proj
        if self.context_decay_proj is not None:
            state = state + decay_emb @ self.context_decay_proj
        if self.context_phi_proj is not None:
            state = state + phi_base @ self.context_phi_proj
        if self.context_len_proj is not None:
            state = state + len_ratio * self.context_len_proj
        return state, (phi_base if phi is None else phi.detach().float().to(self.device))

    def token_embedding(self, token_ids: int | list[int] | torch.Tensor) -> torch.Tensor:
        if self.emb is not None:
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
            if self.vocab_id_map is None:
                return self.emb.index_select(0, token_ids)
            compact_ids = self.vocab_id_map.index_select(0, token_ids)
            kept = compact_ids >= 0
            safe = compact_ids.clamp_min(0)
            out = self.emb.index_select(0, safe)
            if not bool(kept.all().item()):
                fallback = self.unk_emb if self.unk_emb is not None else torch.zeros_like(out[0])
                out = torch.where(kept.unsqueeze(1), out, fallback.unsqueeze(0).expand_as(out))
            return out
        if self.pq_centroids is not None and self.pq_codes is not None:
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
            return pq_reconstruct_tokens(
                self.pq_centroids,
                self.pq_codes,
                token_ids,
            ).to(self.device)
        raise RuntimeError("No lexical memory is available for token embedding lookup")

    def lexical_scores(self, query: torch.Tensor) -> torch.Tensor:
        if self.emb is not None:
            scores = self.emb @ query
            if self.kept_token_ids is None:
                return scores
            full = torch.full((self.vocab,), float("-inf"), dtype=scores.dtype, device=scores.device)
            full.index_copy_(0, self.kept_token_ids, scores)
            return full
        if self.pq_centroids is not None and self.pq_codes is not None:
            return pq_scores(query, self.pq_centroids, self.pq_codes)
        raise RuntimeError("No lexical memory is available for scoring")

    @staticmethod
    def _rescale_to_reference(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if query.ndim == 1:
            q_norm = float(query.norm().item())
            r_norm = float(reference.norm().item())
            if not math.isfinite(q_norm) or not math.isfinite(r_norm) or q_norm <= 1e-8 or r_norm <= 1e-8:
                return query
            return query * (r_norm / q_norm)
        q_norm = query.norm(dim=-1, keepdim=True)
        r_norm = reference.norm(dim=-1, keepdim=True)
        valid = torch.isfinite(q_norm) & torch.isfinite(r_norm) & (q_norm > 1e-8) & (r_norm > 1e-8)
        scale = torch.ones_like(q_norm)
        scale = torch.where(valid, r_norm / q_norm.clamp_min(1e-8), scale)
        return query * scale

    def decoder_query(self, state_hidden: torch.Tensor, prev_emb: torch.Tensor) -> torch.Tensor:
        state_query = self._project_state_query(state_hidden.float())
        prev_query = self._project_prev_query(prev_emb.float())
        query = state_query + float(self.decoder_prev_scale) * prev_query
        if self.decoder_query_bias is not None:
            query = query + self.decoder_query_bias

        blend = float(self.decoder_query_blend)
        if blend <= 0.0:
            query = state_hidden.float()
        elif blend < 1.0:
            query = blend * query + (1.0 - blend) * state_hidden.float()

        query = self._rescale_to_reference(query, state_hidden.float())
        finite_mask = torch.isfinite(query).all(dim=-1) if query.ndim > 1 else torch.isfinite(query).all()
        if query.ndim == 1:
            if not bool(finite_mask.item()) or query.abs().sum().item() <= 1e-8:
                return state_hidden.float()
            return query
        fallback = state_hidden.float()
        valid_rows = finite_mask & (query.abs().sum(dim=-1) > 1e-8)
        return torch.where(valid_rows.unsqueeze(-1), query, fallback)

    @staticmethod
    def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
        scores = logits.float()
        if scores.numel() <= 1:
            return scores.clone()
        mean = scores.mean()
        std = scores.std(unbiased=False).clamp_min(1e-6)
        return (scores - mean) / std

    def _merge_candidate_ids(self, *groups: torch.Tensor | None) -> torch.Tensor:
        merged: list[int] = []
        seen: set[int] = set()
        for group in groups:
            if group is None or group.numel() == 0:
                continue
            for token_id in group.reshape(-1).tolist():
                token_int = int(token_id)
                if token_int < 0 or token_int >= self.vocab or token_int in seen:
                    continue
                seen.add(token_int)
                merged.append(token_int)
        if not merged:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.tensor(merged, dtype=torch.long, device=self.device)

    def _sentence_terminal_ids(self) -> torch.Tensor:
        if self._terminal_ids_cache is not None:
            return self._terminal_ids_cache
        terminal_ids: list[int] = []
        if self.eos_token_id is not None:
            terminal_ids.append(int(self.eos_token_id))
        if self.tok is not None:
            for token_text in (".", "!", "?", "다", "요", "죠", "네", "까", "니다"):
                try:
                    token_ids = self.tok.encode(token_text, add_special_tokens=False)
                except TypeError:
                    token_ids = self.tok.encode(token_text)
                if len(token_ids) == 1:
                    terminal_ids.append(int(token_ids[0]))
        if terminal_ids:
            cache_src = torch.tensor(terminal_ids, dtype=torch.long, device=self.device)
            self._terminal_ids_cache = self._merge_candidate_ids(cache_src)
        else:
            self._terminal_ids_cache = torch.empty(0, dtype=torch.long, device=self.device)
        return self._terminal_ids_cache

    def _sentence_close_bonus(self, candidate_ids: torch.Tensor, *, generated_len: int) -> torch.Tensor:
        bonus = torch.zeros(candidate_ids.shape[0], dtype=torch.float32, device=self.device)
        if generated_len < 10 or candidate_ids.numel() == 0:
            return bonus
        terminal_ids = self._sentence_terminal_ids()
        if terminal_ids.numel() == 0:
            return bonus
        terminal_mask = (candidate_ids.unsqueeze(1) == terminal_ids.unsqueeze(0)).any(dim=1)
        if not terminal_mask.any():
            return bonus
        close_bonus = min(1.5, 0.35 + 0.08 * float(generated_len - 10))
        bonus[terminal_mask] = close_bonus
        if self.eos_token_id is not None:
            bonus[candidate_ids == int(self.eos_token_id)] += 0.15
        return bonus

    def _paper_candidate_count(self, vocab_size: int, top_k: int) -> int:
        ratio = min(max(float(self.decoder_candidate_ratio), 1e-6), 1.0)
        target = int(math.ceil(ratio * float(vocab_size)))
        target = max(target, int(top_k), 1)
        return min(int(vocab_size), target)

    def ensure_vocab_head(self):
        if self.decoder_vocab_weight is not None:
            if self.decoder_vocab_bias is None:
                self.decoder_vocab_bias = torch.zeros(
                    self.decoder_vocab_weight.shape[0],
                    dtype=self.decoder_vocab_weight.dtype,
                    device=self.device,
                )
                self.data["decoder_vocab_bias"] = self.decoder_vocab_bias.detach().cpu()
            self.data["decoder_vocab_scale"] = float(self.decoder_vocab_scale)
            return
        if self.emb is None:
            raise RuntimeError("cloned vocab head requires full embedding weights")
        weight = self.emb.detach().cpu().clone()
        bias = torch.zeros(weight.shape[0], dtype=weight.dtype)
        self.apply_vocab_head(weight, bias=bias, scale=1.0)

    def vocab_logits(self, query: torch.Tensor) -> torch.Tensor:
        if self.decoder_vocab_weight is not None:
            bias = None if self.decoder_vocab_bias is None else self.decoder_vocab_bias.float()
            logits = F.linear(query.float(), self.decoder_vocab_weight.float(), bias)
            return float(self.decoder_vocab_scale) * logits
        return self.lexical_scores(query.float())

    def _ngram_repeat_scores(self, history_ids: list[int] | None, candidate_ids: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(candidate_ids.shape[0], dtype=torch.float32, device=self.device)
        if not history_ids:
            return scores
        ngram = max(int(self.repeat_ngram), 2)
        if len(history_ids) < ngram - 1:
            return scores
        prefix = tuple(int(x) for x in history_ids[-(ngram - 1):])
        seen = {
            tuple(int(x) for x in history_ids[idx : idx + ngram])
            for idx in range(max(len(history_ids) - ngram + 1, 0))
        }
        if not seen:
            return scores
        values = [1.0 if (*prefix, int(token_id)) in seen else 0.0 for token_id in candidate_ids.tolist()]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _curvature_adjust_logits(
        self,
        candidate_ids: torch.Tensor,
        candidate_logits: torch.Tensor,
        *,
        ce_hidden: torch.Tensor,
        prev_hidden: torch.Tensor | None = None,
        prev_prev_hidden: torch.Tensor | None = None,
        history_ids: list[int] | None = None,
        context_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if candidate_ids.numel() == 0:
            return candidate_logits, {
                "candidate_ids": candidate_ids,
                "combined_risk": torch.empty(0, device=self.device),
                "phase_grounding_risk": torch.empty(0, device=self.device),
                "suppression": torch.empty(0, device=self.device),
                "threshold": None,
                "curvature_risk_score": 0.0,
                "phase_grounding_risk_score": 0.0,
                "suppressed_count": 0,
            }

        candidate_emb = self.token_embedding(candidate_ids).float()
        current_hidden = ce_hidden.float()
        step_next = candidate_emb - current_hidden.unsqueeze(0)

        k1 = torch.zeros(candidate_ids.shape[0], dtype=torch.float32, device=self.device)
        if prev_hidden is not None:
            prev_step = (current_hidden - prev_hidden.float()).unsqueeze(0).expand_as(step_next)
            k1 = 1.0 - F.cosine_similarity(prev_step, step_next, dim=1, eps=1e-6)
            k1 = k1.clamp_min(0.0)

        k2 = torch.zeros_like(k1)
        if prev_hidden is not None and prev_prev_hidden is not None:
            accel_prev = (current_hidden - 2.0 * prev_hidden.float() + prev_prev_hidden.float()).unsqueeze(0)
            accel_prev = accel_prev.expand_as(step_next)
            accel_next = candidate_emb - 2.0 * current_hidden.unsqueeze(0) + prev_hidden.float().unsqueeze(0)
            k2 = 1.0 - F.cosine_similarity(accel_prev, accel_next, dim=1, eps=1e-6)
            k2 = k2.clamp_min(0.0)

        lap = self.state_graph_laplacian().to(step_next.device, dtype=step_next.dtype)
        lbo = (step_next @ lap).pow(2).mean(dim=1)
        lbo = lbo / lbo.mean().clamp_min(1e-6)

        context_break = torch.zeros_like(k1)
        phase_reference = current_hidden
        if context_anchor is not None:
            anchor = context_anchor.float().unsqueeze(0).expand_as(candidate_emb)
            context_break = 1.0 - F.cosine_similarity(candidate_emb, anchor, dim=1, eps=1e-6)
            context_break = context_break.clamp_min(0.0)
            phase_reference = context_anchor.float()

        repeat = torch.zeros_like(k1)
        if history_ids:
            recent = history_ids[-max(int(self.repeat_window), 1) :]
            if recent:
                recent_ids = torch.tensor(recent, dtype=torch.long, device=self.device)
                repeat = (candidate_ids.unsqueeze(1) == recent_ids.unsqueeze(0)).float().sum(dim=1)
                repeat = repeat / float(max(len(recent), 1))
            repeat = repeat + 2.0 * self._ngram_repeat_scores(history_ids, candidate_ids)

        combined = k1 + 0.5 * k2 + 0.3 * lbo + 0.25 * context_break + 1.5 * repeat
        risk_mean = combined.mean()
        risk_std = combined.std(unbiased=False)
        threshold = risk_mean + float(self.curvature_alpha) * risk_std
        excess = (combined - threshold).clamp_min(0.0)
        gate = torch.sigmoid(float(self.curvature_steepness) * (combined - threshold))
        suppression = float(self.curvature_lambda) * gate * excess
        phase_risk = torch.zeros_like(suppression)
        if float(self.phase_grounding_lambda) > 0.0:
            _, phase_risk = phase_grounding_suppression(
                candidate_logits,
                candidate_emb,
                phase_reference,
                strength=0.0,
            )
            suppression = suppression + float(self.phase_grounding_lambda) * phase_risk
        adjusted = candidate_logits - suppression
        return adjusted, {
            "candidate_ids": candidate_ids,
            "combined_risk": combined,
            "phase_grounding_risk": phase_risk,
            "suppression": suppression,
            "threshold": float(threshold.item()),
            "curvature_risk_score": float((combined >= threshold).float().mean().item()),
            "phase_grounding_risk_score": float((phase_risk > 0.5).float().mean().item()),
            "suppressed_count": int((suppression > 1e-3).sum().item()),
        }

    def build_runtime_codebook(self, m_ref: torch.Tensor, top_k: int) -> torch.Tensor:
        if self.has_standalone_lexicon():
            query = self.masked_state(m_ref, include_struct=True)
            if query.abs().sum().item() <= 1e-8:
                query = m_ref
            scores = self.lexical_scores(query)
            top_ids = torch.topk(scores, min(top_k, scores.numel())).indices
            return self.token_embedding(top_ids)
        raise RuntimeError("legacy teacher-dependent codebook path was removed from reality_stone.clarus")

    def ce_hidden(self, m_star: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(m_star, (self.d,), self.ln_w, self.ln_b)

    def teacher_embedding(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("teacher embedding is unavailable for a runtime-only artifact")
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
        return self.model.transformer.wte.weight.index_select(0, token_ids)

    def teacher_hidden_and_logits(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("teacher path is disabled in runtime-only mode")

    def teacher_next_logits(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("teacher path is disabled in runtime-only mode")

    def decoder_token_correction(
        self,
        ce_hidden: torch.Tensor,
        prev_emb: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.decoder_token_ids is None:
            return None
        correction = None
        if self.decoder_token_state_proj is not None:
            proj_idx = self._projection_indices()
            if (
                proj_idx is not None
                and self.decoder_token_state_proj.ndim == 2
                and self.decoder_token_state_proj.shape[0] == proj_idx.numel()
            ):
                correction = ce_hidden.index_select(0, proj_idx) @ self.decoder_token_state_proj
            else:
                correction = ce_hidden @ self.decoder_token_state_proj
        if self.decoder_token_prev_proj is not None:
            prev_piece = self.decoder_prev_scale * (prev_emb @ self.decoder_token_prev_proj)
            correction = prev_piece if correction is None else correction + prev_piece
        if self.decoder_token_bias is not None:
            correction = self.decoder_token_bias if correction is None else correction + self.decoder_token_bias
        if correction is None:
            return None
        return self.decoder_token_scale * correction

    def standalone_logits(
        self,
        ce_hidden: torch.Tensor,
        prev_id: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        repeat_ids: list[int] | None = None,
        repeat_penalty: float = 3.0,
        history_ids: list[int] | None = None,
        prev_hidden: torch.Tensor | None = None,
        prev_prev_hidden: torch.Tensor | None = None,
        context_anchor: torch.Tensor | None = None,
        generated_len: int = 0,
        return_meta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        prev_emb = self.token_embedding([prev_id]).squeeze(0)
        state_hidden = ce_hidden.float()
        query = self.decoder_query(state_hidden, prev_emb)
        logits = self.vocab_logits(query)
        correction = self.decoder_token_correction(state_hidden, prev_emb)
        if correction is not None and self.decoder_token_ids is not None and self.decoder_token_ids.numel():
            logits = logits.clone()
            logits[self.decoder_token_ids] += correction

        candidate_k = self._paper_candidate_count(logits.numel(), top_k)
        candidate_top_ids = torch.topk(logits, min(candidate_k, logits.numel())).indices
        candidate_ids = self._merge_candidate_ids(candidate_top_ids, self._sentence_terminal_ids())
        if candidate_ids.numel() == 0:
            logits = logits.new_full(logits.shape, float("-inf"))
            if not return_meta:
                return logits
            return logits, {
                "candidate_ids": torch.empty(0, dtype=torch.long, device=self.device),
                "combined_risk": torch.empty(0, dtype=torch.float32, device=self.device),
                "phase_grounding_risk": torch.empty(0, dtype=torch.float32, device=self.device),
                "suppression": torch.empty(0, dtype=torch.float32, device=self.device),
                "threshold": None,
                "curvature_risk_score": 0.0,
                "phase_grounding_risk_score": 0.0,
                "suppressed_count": 0,
                "candidate_count": 0,
                "eval_candidate_count": 0,
            }

        candidate_logits = logits.index_select(0, candidate_ids)
        if repeat_ids:
            repeat_set = {int(token_id) for token_id in repeat_ids}
            if repeat_set:
                repeat_mask = torch.tensor(
                    [int(token_id) in repeat_set for token_id in candidate_ids.tolist()],
                    dtype=torch.bool,
                    device=self.device,
                )
                candidate_logits = candidate_logits.clone()
                candidate_logits[repeat_mask] -= repeat_penalty

        eval_k = min(
            int(candidate_ids.numel()),
            max(int(top_k) * 2, min(int(self.curvature_eval_topk), 96), 1),
        )
        eval_rank = torch.topk(candidate_logits, eval_k).indices
        eval_ids = candidate_ids.index_select(0, eval_rank)
        eval_logits = candidate_logits.index_select(0, eval_rank)
        adjusted_eval_logits, curvature_meta = self._curvature_adjust_logits(
            eval_ids,
            eval_logits,
            ce_hidden=ce_hidden,
            prev_hidden=prev_hidden,
            prev_prev_hidden=prev_prev_hidden,
            history_ids=history_ids,
            context_anchor=context_anchor,
        )
        candidate_logits = candidate_logits.clone()
        candidate_logits.index_copy_(0, eval_rank, adjusted_eval_logits)
        candidate_logits = candidate_logits + self._sentence_close_bonus(
            candidate_ids,
            generated_len=generated_len,
        )
        logits = logits.new_full(logits.shape, float("-inf"))
        logits[candidate_ids] = candidate_logits
        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits = logits.clone()
            logits[logits < v[-1]] = float("-inf")
        if not return_meta:
            return logits
        curvature_meta["candidate_count"] = int(candidate_ids.numel())
        curvature_meta["eval_candidate_count"] = int(eval_k)
        return logits, curvature_meta

    def apply_vocab_head(
        self,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        scale: float | None = None,
    ):
        self.decoder_vocab_weight = weight.float().to(self.device)
        if bias is None:
            bias = torch.zeros(self.decoder_vocab_weight.shape[0], dtype=self.decoder_vocab_weight.dtype)
        self.decoder_vocab_bias = bias.float().to(self.device)
        if scale is not None:
            self.decoder_vocab_scale = float(scale)
        self.data["decoder_vocab_weight"] = self.decoder_vocab_weight.detach().cpu()
        self.data["decoder_vocab_bias"] = self.decoder_vocab_bias.detach().cpu()
        self.data["decoder_vocab_scale"] = float(self.decoder_vocab_scale)

    def apply_decoder_refine(
        self,
        prev_proj: torch.Tensor,
        state_proj: torch.Tensor,
        *,
        query_bias: torch.Tensor | None = None,
    ):
        self.decoder_prev_proj = self._compress_prev_proj(prev_proj.detach().cpu())
        self.decoder_state_proj = self._compress_state_proj(state_proj.detach().cpu())
        self.data["decoder_prev_proj"] = self.decoder_prev_proj.detach().cpu()
        self.data["decoder_state_proj"] = self.decoder_state_proj.detach().cpu()
        if query_bias is not None:
            self.decoder_query_bias = query_bias.detach().float().to(self.device)
            self.data["decoder_query_bias"] = self.decoder_query_bias.detach().cpu()

    def apply_token_head(
        self,
        token_ids: torch.Tensor,
        *,
        state_proj: torch.Tensor | None = None,
        prev_proj: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        scale: float | None = None,
    ):
        self.decoder_token_ids = token_ids.long().to(self.device)
        self.decoder_token_state_proj = None if state_proj is None else self._compress_token_state_proj(state_proj.detach().cpu())
        self.decoder_token_prev_proj = None if prev_proj is None else prev_proj.float().to(self.device)
        self.decoder_token_bias = None if bias is None else bias.float().to(self.device)
        if scale is not None:
            self.decoder_token_scale = float(scale)
        self.data["decoder_token_ids"] = token_ids.detach().cpu().long()
        self.data["decoder_token_state_proj"] = None if self.decoder_token_state_proj is None else self.decoder_token_state_proj.detach().cpu()
        self.data["decoder_token_prev_proj"] = None if prev_proj is None else prev_proj.detach().cpu()
        self.data["decoder_token_bias"] = None if bias is None else bias.detach().cpu()
        self.data["decoder_token_scale"] = float(self.decoder_token_scale)

    def decoder_snapshot(self) -> dict[str, torch.Tensor | float | None]:
        def clone_cpu(value):
            return None if value is None else value.detach().cpu().clone()

        return {
            "decoder_prev_proj": clone_cpu(self.decoder_prev_proj),
            "decoder_state_proj": clone_cpu(self.decoder_state_proj),
            "decoder_query_bias": clone_cpu(self.decoder_query_bias),
            "decoder_vocab_weight": clone_cpu(self.decoder_vocab_weight),
            "decoder_vocab_bias": clone_cpu(self.decoder_vocab_bias),
            "decoder_vocab_scale": float(self.decoder_vocab_scale),
            "decoder_token_ids": clone_cpu(self.decoder_token_ids),
            "decoder_token_state_proj": clone_cpu(self.decoder_token_state_proj),
            "decoder_token_prev_proj": clone_cpu(self.decoder_token_prev_proj),
            "decoder_token_bias": clone_cpu(self.decoder_token_bias),
            "decoder_token_scale": float(self.decoder_token_scale),
            "pq_centroids": clone_cpu(self.pq_centroids),
            "pq_codes": clone_cpu(self.pq_codes),
            "W": clone_cpu(self.W),
            "active_dim_mask": clone_cpu(self.active_dim_mask),
            "struct_dim_mask": clone_cpu(self.struct_dim_mask),
            "background_dim_mask": clone_cpu(self.background_dim_mask),
        }

    def restore_decoder_snapshot(self, snapshot: dict[str, torch.Tensor | float | None]):
        def load_tensor(name: str):
            value = snapshot.get(name)
            return None if value is None else value.to(self.device)

        w_tensor = snapshot.get("W")
        if w_tensor is not None:
            self.apply_relax_matrix(w_tensor)
        self.decoder_prev_proj = load_tensor("decoder_prev_proj")
        self.decoder_state_proj = load_tensor("decoder_state_proj")
        self.decoder_query_bias = load_tensor("decoder_query_bias")
        self.decoder_vocab_weight = load_tensor("decoder_vocab_weight")
        self.decoder_vocab_bias = load_tensor("decoder_vocab_bias")
        self.decoder_vocab_scale = float(snapshot.get("decoder_vocab_scale", self.decoder_vocab_scale))
        self.decoder_token_ids = load_tensor("decoder_token_ids")
        self.decoder_token_state_proj = load_tensor("decoder_token_state_proj")
        self.decoder_token_prev_proj = load_tensor("decoder_token_prev_proj")
        self.decoder_token_bias = load_tensor("decoder_token_bias")
        self.decoder_token_scale = float(snapshot.get("decoder_token_scale", self.decoder_token_scale))
        self.pq_centroids = load_tensor("pq_centroids")
        self.pq_codes = load_tensor("pq_codes")
        active_mask = snapshot.get("active_dim_mask")
        struct_mask = snapshot.get("struct_dim_mask")
        if active_mask is not None and struct_mask is not None:
            self.apply_state_partition(active_mask, struct_mask)
        else:
            self.active_dim_mask = None
            self.struct_dim_mask = None
            self.background_dim_mask = None

        for key in (
            "decoder_prev_proj",
            "decoder_state_proj",
            "decoder_query_bias",
            "decoder_vocab_weight",
            "decoder_vocab_bias",
            "decoder_token_ids",
            "decoder_token_state_proj",
            "decoder_token_prev_proj",
            "decoder_token_bias",
            "pq_centroids",
            "pq_codes",
            "active_dim_mask",
            "struct_dim_mask",
            "background_dim_mask",
        ):
            value = snapshot.get(key)
            self.data[key] = None if value is None else value.clone()
        self.data["decoder_vocab_scale"] = float(self.decoder_vocab_scale)
        self.data["decoder_token_scale"] = float(self.decoder_token_scale)

    def standalone_generate(
        self,
        prompt_ids: torch.Tensor,
        m_star: torch.Tensor,
        *,
        max_tok: int,
        temperature: float,
        top_k: int,
        repeat_penalty: float,
        refresh_interval: int = 0,
        refresh_args=None,
        refresh_init_layer: int | None = None,
        refresh_phi: torch.Tensor | None = None,
    ) -> tuple[str, list[int], dict[str, float | int | None]]:
        if not self.has_standalone_lexicon():
            raise RuntimeError("Standalone decoder requires embeddings or PQ lexical memory")

        refresh_interval = max(int(refresh_interval), 0)
        if getattr(self, '_skip_ln_for_standalone', False):
            h = m_star.float().to(self.device)
        else:
            h = F.layer_norm(m_star, (self.d,), self.ln_w, self.ln_b)
        prev_id = int(prompt_ids[0, -1].item())
        running_ids = prompt_ids.clone()
        out_ids: list[int] = []
        phi_state = None if refresh_phi is None else refresh_phi.detach().clone().to(self.device)
        init_layer = refresh_init_layer
        refresh_count = 0
        refresh_steps = 0
        refresh_time_s = 0.0
        refresh_cos: list[float] = []
        chosen_risk: list[float] = []
        chosen_phase_grounding_risk: list[float] = []
        chosen_suppression: list[float] = []
        step_risk_score: list[float] = []
        suppression_hits = 0
        history_ids = running_ids[0].tolist()
        prev_hidden = None
        prev_prev_hidden = None
        context_anchor = h.detach().clone()

        for _ in range(max_tok):
            logits, step_meta = self.standalone_logits(
                h,
                prev_id,
                temperature=temperature,
                top_k=top_k,
                repeat_ids=out_ids[-max(int(self.repeat_window), 1) :],
                repeat_penalty=repeat_penalty,
                history_ids=history_ids,
                prev_hidden=prev_hidden,
                prev_prev_hidden=prev_prev_hidden,
                context_anchor=context_anchor,
                generated_len=len(out_ids),
                return_meta=True,
            )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break
            candidate_ids = step_meta["candidate_ids"]
            if candidate_ids.numel():
                match = torch.nonzero(candidate_ids == next_id, as_tuple=False)
                if match.numel():
                    idx = int(match[0, 0].item())
                    risk_values = step_meta["combined_risk"]
                    phase_values = step_meta["phase_grounding_risk"]
                    suppression_values = step_meta["suppression"]
                    chosen_risk.append(float(risk_values[idx].item()))
                    chosen_phase_grounding_risk.append(float(phase_values[idx].item()))
                    chosen_suppression.append(float(suppression_values[idx].item()))
            step_risk_score.append(float(step_meta["curvature_risk_score"]))
            suppression_hits += int(step_meta["suppressed_count"])

            step_hidden = h.detach().clone()
            out_ids.append(next_id)
            prev_id = next_id
            next_token = torch.tensor([[next_id]], device=self.device)
            running_ids = torch.cat([running_ids, next_token], dim=1)
            history_ids.append(next_id)
            prev_prev_hidden = prev_hidden
            prev_hidden = step_hidden
            if (
                refresh_interval > 0
                and refresh_args is not None
                and init_layer is not None
                and len(out_ids) < max_tok
                and len(out_ids) % refresh_interval == 0
            ):
                refresh_ctx = self.context_from_ids(
                    running_ids,
                    init_layer=init_layer,
                    phi=phi_state,
                    need_teacher=False,
                )
                refresh_result = self.relax_context(refresh_ctx, refresh_args)
                h = self.ce_hidden(refresh_result["m_star"])
                phi_state = refresh_result["phi_updated"]
                init_layer = refresh_ctx.best_layer
                context_anchor = h.detach().clone()
                refresh_count += 1
                refresh_steps += int(refresh_result["steps"])
                refresh_time_s += float(refresh_result["elapsed_s"])
                if refresh_result["cos_ms_h"] is not None:
                    refresh_cos.append(float(refresh_result["cos_ms_h"]))

        meta = {
            "refresh_interval": refresh_interval,
            "refresh_count": refresh_count,
            "refresh_steps": refresh_steps,
            "refresh_time_s": refresh_time_s,
            "refresh_cos_mean": None if not refresh_cos else sum(refresh_cos) / len(refresh_cos),
            "refresh_phi_norm": None if phi_state is None else float(phi_state.norm().item()),
            "curvature_risk_score": None if not step_risk_score else sum(step_risk_score) / len(step_risk_score),
            "chosen_risk_mean": None if not chosen_risk else sum(chosen_risk) / len(chosen_risk),
            "chosen_phase_grounding_risk_mean": (
                None
                if not chosen_phase_grounding_risk
                else sum(chosen_phase_grounding_risk) / len(chosen_phase_grounding_risk)
            ),
            "chosen_suppression_mean": None if not chosen_suppression else sum(chosen_suppression) / len(chosen_suppression),
            "suppression_hits": int(suppression_hits),
        }
        return self.tok.decode(out_ids, skip_special_tokens=True), out_ids, meta

    def legacy_generate(
        self,
        token_ids: list[int],
        *,
        n_tokens: int,
        residual_scale: float,
        temperature: float,
    ) -> list[int]:
        if self.emb is None or not self.data.get("W_layers"):
            raise RuntimeError("Legacy generator requires full embedding table and W_layers")

        generated = list(token_ids)
        used = set()
        m = self.emb[token_ids].mean(dim=0)
        phi = self.emb.var(dim=0).clamp(min=1e-8).sqrt()
        m_hist: list[torch.Tensor] = [m.clone()]

        for _ in range(n_tokens):
            m_out = m.clone()
            for w_layer in self.data["W_layers"]:
                w_dev = w_layer.to(self.device)
                delta = w_dev @ m_out
                m_out = m_out + residual_scale * delta
                m_out = m_out / (m_out.norm() + 1e-8) * m.norm()
            m_out = m_out * (self.h_norm_ref / (m_out.norm() + 1e-8))

            phi_hat = F.normalize(phi, dim=0)
            m_out = m_out + self.portal * phi_hat * self.h_norm_ref

            # Bypass force per E20 (docs/7_AGI/12_Equation.md 1.5/3.1):
            #   F_bypass(k) = (C_k / alpha_b) * phi,   C_k = ||m_k - 2 m_{k-1} + m_{k-2}||.
            # Until 3 trajectory samples are available C_k = 0, matching ce_ops.relax.
            if len(m_hist) >= 3:
                c_k = float((m_hist[-1] - 2.0 * m_hist[-2] + m_hist[-3]).norm().item())
                m_out = m_out + (c_k * self.bypass) * phi

            h = F.layer_norm(m_out, (self.d,), self.ln_w, self.ln_b)
            logits = h @ self.emb.T
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            candidates = torch.topk(probs, 50)

            next_id = None
            for cid in candidates.indices.tolist():
                if cid not in used:
                    next_id = cid
                    break
            if next_id is None:
                next_id = int(candidates.indices[0].item())

            generated.append(next_id)
            used.add(next_id)
            new_emb = self.emb[next_id]
            m = 0.3 * m + 0.7 * new_emb
            m_hist.append(m.clone())
            if len(m_hist) > 3:
                m_hist.pop(0)

        return generated

    def prompt_context(self, prompt: str) -> PromptContext:
        prompt_ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        return self.context_from_ids(prompt_ids, prompt=prompt)

    def _analyze_prompt_ids(
        self,
        prompt_ids: torch.Tensor,
        *,
        candidate_layers: list[int],
        need_teacher: bool,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor | None]:
        """Run a teacher forward pass and capture hidden states by layer.

        Returns ``(phi, captured, h_true)`` where ``phi`` is the normalized
        difference between the mean of all-but-last token hidden states and the
        last token hidden state (zero for single-token prompts), ``captured``
        is a dict of ``{layer_idx: hidden_state}`` for the requested layers,
        and ``h_true`` is the last-layer hidden state at the final position.

        Requires a teacher ``self.model`` to be present (e.g. for offline
        analysis or unit tests that inject a ``FakeModel``). Runtime-only
        artifacts must use :meth:`runtime_prompt_state` instead.
        """
        if self.model is None:
            raise RuntimeError(
                "_analyze_prompt_ids requires a teacher model; runtime artifacts must "
                "use runtime_prompt_state() / context_from_ids(need_teacher=False)"
            )

        ids = prompt_ids.to(device=self.device, dtype=torch.long)
        transformer = getattr(self.model, "transformer", None)
        if transformer is None:
            raise RuntimeError("teacher model is missing the .transformer attribute")

        with torch.no_grad():
            wte = transformer.wte(ids)
            pos_idx = torch.arange(ids.shape[1], device=self.device, dtype=torch.long)
            wpe = transformer.wpe(pos_idx).unsqueeze(0)
            h = wte + wpe

            seq = h[0]
            if seq.shape[0] <= 1:
                phi = torch.zeros(seq.shape[-1], device=self.device, dtype=seq.dtype)
            else:
                phi = normalize_vector(seq[:-1].mean(dim=0) - seq[-1])

            capture_set = sorted({int(layer) for layer in candidate_layers})
            captured: dict[int, torch.Tensor] = {}
            blocks = transformer.h
            target_layer = max(capture_set[-1] if capture_set else -1, len(blocks) - 1)
            for layer_idx in range(target_layer + 1):
                h = blocks[layer_idx](h)
                if isinstance(h, tuple):
                    h = h[0]
                if layer_idx in capture_set:
                    captured[layer_idx] = h[:, -1, :].squeeze(0).detach().float()

            h_true = transformer.ln_f(h)[:, -1, :].detach().float() if need_teacher else None
        return phi, captured, h_true

    def context_from_ids(
        self,
        prompt_ids: torch.Tensor,
        prompt: str | None = None,
        *,
        init_layer: int | None = None,
        phi: torch.Tensor | None = None,
        need_teacher: bool = True,
    ) -> PromptContext:
        if need_teacher and self.model is not None and not getattr(self, 'allow_pretrained_fallback', False):
            raise RuntimeError("teacher path is disabled in runtime-only mode")
        m0, phi_base = self.runtime_prompt_state(prompt_ids, phi=phi)
        best_layer = int(init_layer) if init_layer is not None else int(
            self.data.get("default_init_layer", max(self.n_layer - 1, 0))
        )
        phi_state = phi_base if phi is None else phi.detach().float().to(self.device)
        return PromptContext(
            prompt=prompt if prompt is not None else self.tok.decode(prompt_ids[0], skip_special_tokens=True),
            prompt_ids=prompt_ids,
            h_true=None,
            m0=m0,
            phi=phi_state,
            best_layer=best_layer,
            layer_scores={best_layer: float("nan")},
        )

    def relax_context(self, ctx: PromptContext, args):
        dt_eff = min(float(args.dt), 0.9 * self.tau)
        cb_weight = self.portal if args.cb_weight is None else float(args.cb_weight)
        codebook = self.build_runtime_codebook(ctx.m0, top_k=args.cb_topk)
        metric_basis = ce_build_metric_basis(
            codebook,
            ctx.m0,
            int(args.metric_rank),
            w_eigvecs=self._get_w_eigvecs(args.metric_rank),
            backend=args.backend,
        )
        t0 = time.time()
        m_star, hist, n_steps = ce_relax_packed(
            self.W_pack[0],
            self.W_pack[1],
            self.W_pack[2],
            ctx.m0,
            ctx.phi,
            ctx.m0,
            codebook,
            metric_basis,
            portal=self.portal,
            bypass=self.bypass,
            t_wake=self.t_wake,
            beta=args.beta,
            cb_w=cb_weight,
            tau=self.tau,
            dt=dt_eff,
            max_steps=args.steps,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=args.noise_scale,
            anneal_ratio=0.6,
            tol=1e-4,
            backend=args.backend,
            seed=args.seed,
            dense_w=self._dense_relax_w,
        )
        elapsed = time.time() - t0
        cos_ms = None
        if ctx.h_true is not None:
            cos_ms = F.cosine_similarity(m_star.unsqueeze(0), ctx.h_true).item()
        phi_var = hist.get("phi_var")
        if phi_var:
            phi_updated = update_phi(ctx.phi, m_star, phi_var=m_star.new_tensor(phi_var))
        else:
            phi_updated = update_phi(ctx.phi, m_star)
        return {
            "m_star": m_star,
            "hist": hist,
            "steps": n_steps,
            "elapsed_s": elapsed,
            "cos_m0_h": ctx.layer_scores[ctx.best_layer],
            "cos_ms_h": cos_ms,
            "phi_updated": phi_updated,
            "dt_eff": dt_eff,
        }

    def select_mode(self, phi_updated: torch.Tensor, args) -> str:
        mode = getattr(args, "decode_mode", "auto")
        if mode not in ("auto", "standalone"):
            raise RuntimeError("runtime-only mode supports standalone decoding")
        return "standalone"

    @staticmethod
    def _copy_args(args, **updates):
        payload = vars(args).copy()
        payload.update(updates)
        return SimpleNamespace(**payload)

    def decode_outputs(self, ctx: PromptContext, relax_result: dict, args):
        outputs: dict[str, str] = {}
        meta: dict[str, object] = {}
        chosen_mode = self.select_mode(relax_result["phi_updated"], args)

        def run_standalone():
            refresh_args = None
            if args.standalone_refresh_interval > 0:
                refresh_args = self._copy_args(
                    args,
                    steps=args.standalone_refresh_steps,
                    cb_topk=args.standalone_refresh_cb_topk,
                    metric_rank=args.standalone_refresh_metric_rank,
                    noise_scale=args.standalone_refresh_noise_scale,
                )
            text, token_ids, standalone_meta = self.standalone_generate(
                ctx.prompt_ids,
                relax_result["m_star"],
                max_tok=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
                refresh_interval=args.standalone_refresh_interval,
                refresh_args=refresh_args,
                refresh_init_layer=ctx.best_layer,
                refresh_phi=relax_result["phi_updated"],
            )
            outputs["standalone"] = ctx.prompt + text
            meta["standalone_token_ids"] = token_ids
            meta["standalone_refresh_interval"] = standalone_meta["refresh_interval"]
            meta["standalone_refresh_count"] = standalone_meta["refresh_count"]
            meta["standalone_refresh_steps"] = standalone_meta["refresh_steps"]
            meta["standalone_refresh_time_s"] = standalone_meta["refresh_time_s"]
            meta["standalone_refresh_cos_mean"] = standalone_meta["refresh_cos_mean"]
            meta["standalone_refresh_phi_norm"] = standalone_meta["refresh_phi_norm"]
            meta["standalone_curvature_risk"] = standalone_meta["curvature_risk_score"]
            meta["standalone_chosen_risk_mean"] = standalone_meta["chosen_risk_mean"]
            meta["standalone_chosen_phase_grounding_risk_mean"] = (
                standalone_meta["chosen_phase_grounding_risk_mean"]
            )
            meta["standalone_chosen_suppression_mean"] = standalone_meta["chosen_suppression_mean"]
            meta["standalone_suppression_hits"] = standalone_meta["suppression_hits"]

        run_standalone()

        return chosen_mode, outputs, meta

    def reference_generate(self, prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("reference generation is disabled in runtime-only mode")


def build_prompt_list(args) -> list[str]:
    base = list(args.prompts) if args.prompts else [args.prompt, *DEFAULT_PROMPTS]
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in base:
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def build_guard_list(args) -> list[str]:
    base = list(args.microsleep_guard_prompts) if args.microsleep_guard_prompts else list(DEFAULT_PROMPTS)
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in base:
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def load_microsleep_tools():
    if __package__:
        from .sleep import PromptReplayBuffer, evaluate_guard_set, run_guarded_microsleep_step
    else:
        from reality_stone.clarus.sleep import PromptReplayBuffer, evaluate_guard_set, run_guarded_microsleep_step
    return PromptReplayBuffer, evaluate_guard_set, run_guarded_microsleep_step


def main():
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="CE Equation-spec runtime")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--prompt", default="인공지능의 미래는")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--multiround-steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--compare-gpt2", action="store_true")
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cb-weight", type=float, default=None)
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-strength", type=float, default=0.3)
    ap.add_argument(
        "--decode-mode",
        default="standalone",
        choices=["auto", "standalone"],
    )
    ap.add_argument("--phi-threshold", type=float, default=1.0)
    ap.add_argument("--sleep-threshold", type=float, default=2.0)
    ap.add_argument("--sleep-decay", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repeat-penalty", type=float, default=3.0)
    ap.add_argument("--phase-grounding-lambda", type=float, default=None)
    ap.add_argument("--standalone-refresh-interval", type=int, default=1)
    ap.add_argument("--standalone-refresh-steps", type=int, default=48)
    ap.add_argument("--standalone-refresh-cb-topk", type=int, default=128)
    ap.add_argument("--standalone-refresh-metric-rank", type=int, default=0)
    ap.add_argument("--standalone-refresh-noise-scale", type=float, default=0.0)
    ap.add_argument("--microsleep-every", type=int, default=0)
    ap.add_argument("--microsleep-replay-capacity", type=int, default=16)
    ap.add_argument("--microsleep-guard-prompts", nargs="*", default=None)
    ap.add_argument("--microsleep-tokens", type=int, default=4)
    ap.add_argument("--microsleep-label-topk", "--microsleep-teacher-topk", dest="microsleep_teacher_topk", type=int, default=8)
    ap.add_argument("--microsleep-ridge", type=float, default=1e-3)
    ap.add_argument("--microsleep-rem-weight", type=float, default=2.5)
    ap.add_argument("--microsleep-rem-mix", type=float, default=0.35)
    ap.add_argument("--microsleep-token-head-max-vocab", type=int, default=2048)
    ap.add_argument("--microsleep-token-head-scale", type=float, default=1.0)
    ap.add_argument("--microsleep-guard-min-top10-delta", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-min-top50-delta", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-max-top10-drop", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-max-top50-drop", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-max-phase-grounding-risk-increase", type=float, default=None)
    ap.add_argument("--microsleep-output", default=None)
    ap.add_argument(
        "--residual",
        type=float,
        default=0.15,
        help="Legacy sequential residual argument retained for CLI compatibility.",
    )
    args = ap.parse_args()
    if args.compare_gpt2:
        raise RuntimeError("teacher/reference comparison is disabled in runtime-only mode")

    eng = CEEngine(args.engine, device=args.device, backend=args.backend)
    if eng.model is not None or eng.model_source != "runtime":
        raise RuntimeError("runtime-only execution requires a clone-free runtime artifact")
    if args.phase_grounding_lambda is not None:
        eng.phase_grounding_lambda = float(args.phase_grounding_lambda)
        eng.data["phase_grounding_lambda"] = float(eng.phase_grounding_lambda)

    mem = eng.memory_usage()
    prompts = build_prompt_list(args)
    microsleep_events: list[dict[str, object]] = []
    microsleep_guard_initial = None
    microsleep_guard_final = None
    microsleep_guard_prompts: list[str] = []
    microsleep_buffer = None
    run_microsleep_step = None
    evaluate_guard_set = None
    microsleep_accepted = 0
    microsleep_rejected = 0

    safe_print("\n=== CE Hopfield Engine (Equation Spec Runtime) ===")
    safe_print(f"  model={eng.model_name}")
    safe_print(f"  d={eng.d}  layers={eng.n_layer}  vocab={eng.vocab}")
    safe_print(f"  tau={eng.tau:.4f}  hidden_norm_ref={eng.h_norm_ref:.2f}")
    safe_print(f"  backend={args.backend}  device={eng.device}  model_source={eng.model_source}")

    safe_print("\n--- Memory ---")
    for key, value in mem.items():
        safe_print(f"  {key}: {value:.2f}")

    safe_print(
        f"\n--- Generation (steps={args.steps}, dt={min(args.dt, 0.9 * eng.tau):.4f}, "
        f"temp={args.temperature}, mode={args.decode_mode}, "
        f"phase_lambda={eng.phase_grounding_lambda:.4f}) ---"
    )

    if args.microsleep_every > 0:
        PromptReplayBuffer, evaluate_guard_set, run_microsleep_step = load_microsleep_tools()
        microsleep_guard_prompts = build_guard_list(args)
        microsleep_buffer = PromptReplayBuffer(capacity=max(1, args.microsleep_replay_capacity))
        microsleep_guard_initial = evaluate_guard_set(
            eng,
            microsleep_guard_prompts,
            args,
            max_new_tokens=args.microsleep_tokens,
            refresh_interval=args.standalone_refresh_interval,
            refresh_steps=args.standalone_refresh_steps,
            refresh_cb_topk=args.standalone_refresh_cb_topk,
            refresh_metric_rank=args.standalone_refresh_metric_rank,
            refresh_noise_scale=args.standalone_refresh_noise_scale,
        )
        safe_print("\n--- Microsleep ---")
        safe_print(
            f"  every={args.microsleep_every}  replay_capacity={args.microsleep_replay_capacity}  "
            f"tokens={args.microsleep_tokens}  guard_prompts={len(microsleep_guard_prompts)}"
        )
        safe_print(
            f"  guard_top10={microsleep_guard_initial['top10_acc']:.3f}  "
            f"guard_top50={microsleep_guard_initial['top50_acc']:.3f}"
        )

    results = []
    for idx, prompt in enumerate(prompts, start=1):
        ctx = eng.prompt_context(prompt)
        relax_result = eng.relax_context(ctx, args)
        chosen_mode, outputs, decode_meta = eng.decode_outputs(ctx, relax_result, args)

        safe_print(f"\n  [{prompt}]")
        safe_print(
            f"    init_layer={ctx.best_layer}  cos(m0,h)={_format_optional(relax_result['cos_m0_h'])}  "
            f"cos(m*,h)={_format_optional(relax_result['cos_ms_h'])}"
        )
        safe_print(
            f"    relax_steps={relax_result['steps']}  "
            f"time={relax_result['elapsed_s']:.2f}s  "
            f"phi={ctx.phi.norm().item():.2f}->{relax_result['phi_updated'].norm().item():.2f}"
        )
        if relax_result["hist"]["E"]:
            safe_print(
                f"    energy={relax_result['hist']['E'][0]:.4f}"
                f"->{relax_result['hist']['E'][-1]:.4f}"
            )

        output_text = outputs.get(chosen_mode, "")
        safe_print(f"    [{chosen_mode}] -> {output_text}")
        if "standalone_refresh_count" in decode_meta:
            safe_print(
                f"    [standalone-refresh] interval={decode_meta['standalone_refresh_interval']}  "
                f"count={decode_meta['standalone_refresh_count']}  "
                f"time={decode_meta['standalone_refresh_time_s']:.2f}s"
            )
        if "standalone_curvature_risk" in decode_meta:
            safe_print(
                f"    [standalone-guard] risk={_format_optional(decode_meta['standalone_curvature_risk'])}  "
                f"phase={_format_optional(decode_meta.get('standalone_chosen_phase_grounding_risk_mean'))}  "
                f"suppression_hits={int(decode_meta.get('standalone_suppression_hits', 0))}"
            )

        results.append(
            {
                "prompt": prompt,
                "best_init_layer": ctx.best_layer,
                "init_layer_sweep": {
                    str(k): (_optional_float(v) if _optional_float(v) is not None else None)
                    for k, v in ctx.layer_scores.items()
                },
                "mode": chosen_mode,
                "cos_m0_h": _optional_float(relax_result["cos_m0_h"]),
                "cos_ms_h": _optional_float(relax_result["cos_ms_h"]),
                "phi_norm": {
                    "initial": round(ctx.phi.norm().item(), 4),
                    "updated": round(relax_result["phi_updated"].norm().item(), 4),
                },
                "relax_steps": relax_result["steps"],
                "relax_time_s": round(relax_result["elapsed_s"], 4),
                "energy_start": round(relax_result["hist"]["E"][0], 4) if relax_result["hist"]["E"] else None,
                "energy_end": round(relax_result["hist"]["E"][-1], 4) if relax_result["hist"]["E"] else None,
                "outputs": outputs,
                "standalone_curvature_risk": _optional_float(decode_meta.get("standalone_curvature_risk")),
                "standalone_chosen_risk_mean": _optional_float(decode_meta.get("standalone_chosen_risk_mean")),
                "standalone_chosen_phase_grounding_risk_mean": _optional_float(
                    decode_meta.get("standalone_chosen_phase_grounding_risk_mean")
                ),
                "standalone_chosen_suppression_mean": _optional_float(
                    decode_meta.get("standalone_chosen_suppression_mean")
                ),
                "standalone_suppression_hits": int(decode_meta.get("standalone_suppression_hits", 0)),
                "multiround_phi_norms": [
                    round(v, 4) for v in decode_meta.get("multiround_phi_norms", [])[:32]
                ],
                "multiround_energies": [
                    round(v, 4) for v in decode_meta.get("multiround_energies", [])[:32]
                ],
                "gpt2_reference": None,
            }
        )

        if run_microsleep_step is not None and microsleep_buffer is not None:
            event = run_microsleep_step(
                eng,
                microsleep_buffer,
                prompt,
                microsleep_guard_prompts,
                args,
                step_index=idx,
                sleep_every=args.microsleep_every,
                max_new_tokens=args.microsleep_tokens,
                teacher_topk=args.microsleep_teacher_topk,
                ridge=args.microsleep_ridge,
                rem_weight=args.microsleep_rem_weight,
                rem_mix=args.microsleep_rem_mix,
                token_head_max_vocab=args.microsleep_token_head_max_vocab,
                token_head_scale=args.microsleep_token_head_scale,
                refresh_interval=args.standalone_refresh_interval,
                refresh_steps=args.standalone_refresh_steps,
                refresh_cb_topk=args.standalone_refresh_cb_topk,
                refresh_metric_rank=args.standalone_refresh_metric_rank,
                refresh_noise_scale=args.standalone_refresh_noise_scale,
                refresh_pq=False,
                pq_subdim=64,
                pq_bits=8,
                pq_iters=8,
                pq_batch_size=4096,
                pq_sample_size=16384,
                guard_min_top10_delta=args.microsleep_guard_min_top10_delta,
                guard_min_top50_delta=args.microsleep_guard_min_top50_delta,
                guard_max_top10_drop=args.microsleep_guard_max_top10_drop,
                guard_max_top50_drop=args.microsleep_guard_max_top50_drop,
                guard_max_phase_grounding_risk_increase=(
                    args.microsleep_guard_max_phase_grounding_risk_increase
                ),
            )
            if event is not None:
                if event["accepted"]:
                    microsleep_accepted += 1
                else:
                    microsleep_rejected += 1
                microsleep_events.append(event)
                safe_print(
                    f"    [microsleep] accepted={event['accepted']}  buffer={event['buffer_size']}  "
                    f"guard_top10={event['guard_before']['top10_acc']:.3f}"
                    f"->{event['guard_effective']['top10_acc']:.3f}  "
                    f"guard_top50={event['guard_before']['top50_acc']:.3f}"
                    f"->{event['guard_effective']['top50_acc']:.3f}  "
                    f"phase={event['guard_before'].get('phase_grounding_risk', 0.0):.3f}"
                    f"->{event['guard_effective'].get('phase_grounding_risk', 0.0):.3f}"
                )

    mem_after = mem
    microsleep_report = None
    if evaluate_guard_set is not None:
        microsleep_guard_final = evaluate_guard_set(
            eng,
            microsleep_guard_prompts,
            args,
            max_new_tokens=args.microsleep_tokens,
            refresh_interval=args.standalone_refresh_interval,
            refresh_steps=args.standalone_refresh_steps,
            refresh_cb_topk=args.standalone_refresh_cb_topk,
            refresh_metric_rank=args.standalone_refresh_metric_rank,
            refresh_noise_scale=args.standalone_refresh_noise_scale,
        )
        mem_after = eng.memory_usage()
        microsleep_report = {
            "sleep_every": args.microsleep_every,
            "replay_capacity": args.microsleep_replay_capacity,
            "tokens": args.microsleep_tokens,
            "guard_prompts": microsleep_guard_prompts,
            "initial_guard": microsleep_guard_initial,
            "final_guard": microsleep_guard_final,
            "accepted": microsleep_accepted,
            "rejected": microsleep_rejected,
            "events": microsleep_events,
        }
        safe_print("\n--- Microsleep Summary ---")
        safe_print(
            f"  accepted={microsleep_accepted}  rejected={microsleep_rejected}  "
            f"guard_top10={microsleep_guard_initial['top10_acc']:.3f}"
            f"->{microsleep_guard_final['top10_acc']:.3f}  "
            f"guard_top50={microsleep_guard_initial['top50_acc']:.3f}"
            f"->{microsleep_guard_final['top50_acc']:.3f}  "
            f"phase={microsleep_guard_initial.get('phase_grounding_risk', 0.0):.3f}"
            f"->{microsleep_guard_final.get('phase_grounding_risk', 0.0):.3f}"
        )
        if args.microsleep_output and microsleep_accepted > 0:
            eng.save_runtime_artifact(args.microsleep_output)
            microsleep_report["saved_engine"] = args.microsleep_output
            safe_print(f"  microsleep_engine={args.microsleep_output}")

    result_path = os.path.join(os.path.dirname(args.engine), "engine_results.json")
    payload = {
        "engine": os.path.basename(args.engine),
        "model_name": eng.model_name,
        "device": str(eng.device),
        "backend": args.backend,
        "decode_mode": args.decode_mode,
        "tokens": args.tokens,
        "steps": args.steps,
        "temperature": args.temperature,
        "memory": mem,
        "prompts": results,
    }
    if microsleep_report is not None:
        payload["memory_after"] = mem_after
        payload["microsleep"] = microsleep_report
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    safe_print(f"\n  Results -> {result_path}")


if __name__ == "__main__":
    main()
