from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .riemannian_aggregation import RiemannianAggregation
from reality_stone.layers.metric_attention import MetricAttention
from reality_stone.layers.poincare import project_to_ball, poincare_distance
from reality_stone.layers.lorentz import from_poincare, lorentz_distance
from reality_stone.models.semantic_preservation import SemanticPreservationLoss
from .pretrained_backbone import PretrainedBackbone
from reality_stone.utils.pre_segmenter import PreSegmenter

try:
    from reality_stone.data import SentenceTopicDataset, collate_batch
    _HAS_SENTENCE_TOPIC_DATASET = True
except Exception:
    _HAS_SENTENCE_TOPIC_DATASET = False


@dataclass
class HierarchicalLLMConfig:
    vocab_size: int = 32000
    d_model: int = 768
    d_head: int = 64
    num_topics: int = 8
    num_heads_topic: int = 4
    n_layer_decoder: int = 6
    n_head_decoder: int = 8
    c_poincare: float = 1e-3
    c_lorentz: float = -1.0
    
    pretrained_decoder_path: Optional[str] = None
    pretrained_tokenizer: Optional[str] = None
    use_pretrained_embeddings: bool = True
    
    lambda_consistency: float = 1.0
    lambda_diversity: float = 0.1
    lambda_consistency_schedule: str = "constant"
    lambda_diversity_schedule: str = "constant"
    lambda_topic_supervision: float = 0.0
    lambda_metric: float = 0.0
    lambda_curvature: float = 0.0
    curvature_target_poincare: float = 1e-3
    curvature_target_lorentz: float = -1.0
    
    manifold_sentence: str = "poincare"
    manifold_paragraph: str = "poincare"
    temperature_agg: float = 1.0
    
    gamma_up: float = 0.3
    gamma_self: float = 0.5
    gamma_down: float = 0.2
    
    max_answer_sentences: int = 5
    lambda_length: float = 0.05
    lambda_semantic: float = 0.0
    max_lm_seq_len: int = 256
    
    freeze_decoder: bool = False
    freeze_topic_head_backbone: bool = False
    
    lr_backbone: float = 1e-4
    lr_metric: float = 1e-3
    
    lambda_edit: float = 0.0
    max_edit_ratio: float = 0.25
    enable_structural_edit: bool = False
    edit_budget: float = 0.25 


class EditOperationHead(nn.Module):
    def __init__(self, d_model: int, num_ops: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_ops = num_ops
        self.proj = nn.Linear(d_model, num_ops)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class SentenceOrderHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, 1)

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        scores = self.proj(sentence_embeddings)
        return scores.squeeze(-1)


def compute_dynamic_lambda(
    base_lambda: float,
    schedule: str,
    current_epoch: int,
    total_epochs: int,
) -> float:
    if schedule == "constant":
        return base_lambda
    
    progress = current_epoch / max(total_epochs, 1)
    
    if schedule == "decay":
        return base_lambda * (1.0 - 0.9 * progress)
    elif schedule == "grow":
        return base_lambda * (0.1 + 0.9 * progress)
    elif schedule == "warmup":
        if progress < 0.1:
            return base_lambda * (progress / 0.1)
        else:
            return base_lambda
    
    return base_lambda


class SentenceTopicHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        num_topics: int = 8,
        num_heads: int = 4,
        c_poincare: float = 1e-3,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_topics = num_topics
        self.num_heads = num_heads
        self.d_head_per_head = d_head // num_heads
        self.c_poincare = c_poincare
        self.temperature = temperature
        self.poincare_embed = nn.Linear(d_model, d_head)
        self.metric_attn = MetricAttention(
            hidden_size=self.d_head_per_head,
            normalizer="softmax",
            rank=2,
            tau=self.temperature,
            mode="geodesic",
            manifold="poincare",
            c=self.c_poincare,
        )
        self.q_proj = nn.Linear(d_head, d_head)
        self.k_proj = nn.Linear(d_head, d_head)
        self.v_proj = nn.Linear(d_head, d_head)
        self.out_proj = nn.Linear(d_head, d_head)
        self.topic_classifier = nn.Linear(d_head, num_topics)
        self.topic_names = [
            "chief_complaint",
            "history",
            "physical_exam",
            "diagnosis",
            "treatment_plan",
            "prognosis",
            "follow_up",
            "general",
        ]

    def forward(
        self,
        x: torch.Tensor,
        topo_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        B, T, _ = x.shape
        z = self.poincare_embed(x)
        z = project_to_ball(z)
        H = self.num_heads
        d_h = self.d_head_per_head
        q = self.q_proj(z).view(B, T, H, d_h).transpose(1, 2)
        k = self.k_proj(z).view(B, T, H, d_h).transpose(1, 2)
        v = self.v_proj(z).view(B, T, H, d_h).transpose(1, 2)
        topo_dict = {"neighbor": topo_idx}
        topk_cfg = {"neighbor": topo_idx.shape[-1]}
        attn_out = self.metric_attn(
            q,
            k,
            v,
            topo_idx=topo_dict,
            topk_cfg=topk_cfg,
            c=self.c_poincare,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_head)
        attn_out = self.out_proj(attn_out)

        logits = self.topic_classifier(attn_out)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        P_topic = F.softmax(logits, dim=-1)
        P_topic = torch.nan_to_num(
            P_topic,
            nan=1.0 / max(1, self.num_topics),
            posinf=1.0 / max(1, self.num_topics),
            neginf=1.0 / max(1, self.num_topics),
        )

        scores, _ = logits.max(dim=-1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        metric_keys: List[str] = []
        for b in range(B):
            for t in range(T):
                top_topic = int(P_topic[b, t].argmax().item())
                topic_name = self.topic_names[top_topic] if 0 <= top_topic < len(self.topic_names) else "general"
                score_val = float(scores[b, t].item())
                if score_val > 1.0:
                    priority = "high"
                elif score_val > 0.0:
                    priority = "medium"
                else:
                    priority = "low"
                metric_keys.append(f"topic:{topic_name}|priority:{priority}")

        return P_topic, scores, metric_keys


try:
    import reality_stone.metrikey as _metrikey_probe  # type: ignore
    HAS_METRIKEY = True
except Exception:
    HAS_METRIKEY = False


class MetricContextRouter(nn.Module):
    def __init__(
        self,
        d_head: int = 64,
        lambda_min: float = 0.5,
        lambda_max: float = 2.0,
        cache_size: int = 1000,  
        score_quantize: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.cache_size = cache_size
        self.score_quantize = score_quantize
        from collections import OrderedDict
        self._cache: OrderedDict[Tuple[str, float], torch.Tensor] = OrderedDict()

        try:
            import reality_stone.metrikey as metrikey  # type: ignore
            self._metrikey = metrikey
            self._has_metrikey = True
        except Exception:
            self._metrikey = None
            self._has_metrikey = False

    def _clamp_eigen(self, G: torch.Tensor) -> torch.Tensor:
        try:
            eigvals, eigvecs = torch.linalg.eigh(G)
            eigvals = torch.clamp(eigvals, self.lambda_min, self.lambda_max)
            return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
        except Exception:
            return G

    def _make_metric(self, key: str, score_q: float) -> torch.Tensor:
        cache_key = (key, score_q)
        
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        if self._has_metrikey:
            try:
                G = self._metrikey.metric_from_keys(
                    [key],
                    dim=self.d_head,
                    min_lambda=self.lambda_min,
                    max_lambda=self.lambda_max,
                    masses=[score_q],
                )
            except Exception:
                G = torch.eye(self.d_head)
        else:
            G = torch.eye(self.d_head)

        G = self._clamp_eigen(G)
        try:
            L = torch.linalg.cholesky(G)
        except Exception:
            L = torch.eye(self.d_head)

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        
        self._cache[cache_key] = L
        return L

    def forward(self, metric_keys: List[str], scores: torch.Tensor) -> torch.Tensor:
        B, T = scores.shape
        scores_clean = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores_flat = scores_clean.flatten()

        L_list: List[torch.Tensor] = []
        for i, key in enumerate(metric_keys):
            score_val = float(scores_flat[i].item())
            if not (float("-inf") < score_val < float("inf")):
                score_val = 0.0
            score_quantized = round(score_val / self.score_quantize) * self.score_quantize
            L = self._make_metric(key, score_quantized)
            L_list.append(L)
        L_stack = torch.stack(L_list, dim=0)
        return L_stack.view(B, T, self.d_head, self.d_head)


def _spd_log_euclidean_mean(spd_matrices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    B, N, d, _ = spd_matrices.shape
    device = spd_matrices.device
    dtype = spd_matrices.dtype
    
    eps_eye = torch.eye(d, device=device, dtype=dtype) * 1e-6
    spd_matrices = spd_matrices + eps_eye.view(1, 1, d, d)
    
    log_matrices = torch.zeros_like(spd_matrices)
    for b in range(B):
        for n in range(N):
            try:
                eigvals, eigvecs = torch.linalg.eigh(spd_matrices[b, n])
                eigvals = eigvals.clamp(min=1e-6)
                log_eigvals = torch.log(eigvals)
                log_matrices[b, n] = eigvecs @ torch.diag(log_eigvals) @ eigvecs.T
            except Exception:
                log_matrices[b, n] = torch.eye(d, device=device, dtype=dtype)
    
    w = weights.view(B, N, 1, 1)
    log_mean = (w * log_matrices).sum(dim=1)
    
    result = torch.zeros(B, d, d, device=device, dtype=dtype)
    for b in range(B):
        try:
            eigvals, eigvecs = torch.linalg.eigh(log_mean[b])
            exp_eigvals = torch.exp(eigvals)
            result[b] = eigvecs @ torch.diag(exp_eigvals) @ eigvecs.T
        except Exception:
            result[b] = (w[b] * spd_matrices[b]).sum(dim=0)
    
    return result


class SPDMetricMixer(nn.Module):
    def __init__(
        self,
        d_head: int,
        gamma_up: float = 0.3,
        gamma_self: float = 0.5,
        gamma_down: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.gamma_up = gamma_up
        self.gamma_self = gamma_self
        self.gamma_down = gamma_down

    def mix_hierarchy(
        self,
        parent_metric: torch.Tensor,   # [B*, d,d]
        self_metric: torch.Tensor,     # [B*, d,d]
        children_metrics: Optional[torch.Tensor] = None,  # [B*, N_child,d,d]
    ) -> torch.Tensor:
        B, d, _ = self_metric.shape
        mats = [parent_metric, self_metric]
        ws = [self.gamma_up, self.gamma_self]

        if children_metrics is not None and children_metrics.size(1) > 0 and self.gamma_down > 0.0:
            child_mean = children_metrics.mean(dim=1)  # [B,d,d]
            mats.append(child_mean)
            ws.append(self.gamma_down)

        w_sum = sum(ws)
        ws_norm = [w / w_sum for w in ws]
        mats_tensor = torch.stack(mats, dim=1)  # [B,N,d,d]
        w_tensor = torch.tensor(ws_norm, device=self_metric.device, dtype=self_metric.dtype).view(1, -1).expand(B, -1)
        return _spd_log_euclidean_mean(mats_tensor, w_tensor)


class RCELexicalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layer: int = 2,
        n_head: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
        replacement_mask: Optional[torch.Tensor] = None,
        topo_idx: Optional[torch.Tensor] = None,
        candidates: Optional[Dict[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids.clamp(min=0, max=self.vocab_size - 1))
        logits = self.lm_head(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if replacement_mask is None:
            replacement_mask = torch.ones_like(input_ids)
        if candidates is None:
            candidates = {}
        output_ids = input_ids.clone()
        for b in range(B):
            for t in range(T):
                if int(replacement_mask[b, t].item()) == 0:
                    continue
                tok = int(input_ids[b, t].item())
                cand = candidates.get(tok)
                if not cand:
                    cand = [tok]
                chosen = int(cand[0])
                output_ids[b, t] = chosen
        return output_ids.to(device), logits


class HierarchicalLMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_layer: int = 6,
        n_head: int = 8,
        manifold: str = "lorentz",
        c_lorentz: float = -1.0,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.manifold = manifold
        self.c_lorentz = c_lorentz

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [self._make_block() for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _make_block(self) -> nn.Module:
        return _DecoderBlock(self.d_model, self.n_head, self.manifold, self.c_lorentz)

    def forward(
        self,
        input_ids: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
        topo_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape
        device = input_ids.device

        tok = self.token_embed(input_ids)
        max_pos = self.pos_embed.num_embeddings
        pos_ids = torch.arange(S, device=device).clamp(max=max_pos - 1).unsqueeze(0).expand(B, -1)
        pos = self.pos_embed(pos_ids)

        h = tok + pos
        m_ctx = metric_ctx
        topo = topo_idx
        for blk in self.blocks:
            h = blk(h, m_ctx, topo)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, manifold: str, c: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.manifold = manifold
        self.c = c
        d_h = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # geodesic product attention 용 MetricAttention (SPDMetric만 재사용)
        self.attn = MetricAttention(
            hidden_size=d_h,
            normalizer="softmax",
            rank=2,
            tau=1.0,
            mode="dot",  # 점수는 아래에서 geodesic 으로 직접 계산
            manifold=manifold,
            c=abs(float(c)) if c is not None else 1e-3,
        )
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        metric_ctx: Optional[torch.Tensor],
        topo_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.n_head
        d_h = self.d_model // H

        q = self.q_proj(x).view(B, S, H, d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, d_h).transpose(1, 2)

        q = self.attn.metric.scale_q(q)
        k = self.attn.metric.scale_k(k)
        
        if metric_ctx is not None:
            d_ctx = metric_ctx.size(-1)
            if d_ctx == d_h:
                q_perm = q.transpose(1, 2)
                k_perm = k.transpose(1, 2)
                
                q_perm = torch.einsum("bsij,bshj->bshi", metric_ctx, q_perm)
                k_perm = torch.einsum("bsij,bshj->bshi", metric_ctx, k_perm)
                
                q = q_perm.transpose(1, 2)
                k = k_perm.transpose(1, 2)
            elif d_ctx < d_h:
                q_perm = q.transpose(1, 2)
                k_perm = k.transpose(1, 2)
                
                q_sub = q_perm[..., :d_ctx]
                k_sub = k_perm[..., :d_ctx]
                
                q_sub = torch.einsum("bsij,bshj->bshi", metric_ctx, q_sub)
                k_sub = torch.einsum("bsij,bshj->bshi", metric_ctx, k_sub)
                
                q_perm = torch.cat([q_sub, q_perm[..., d_ctx:]], dim=-1)
                k_perm = torch.cat([k_sub, k_perm[..., d_ctx:]], dim=-1)
                
                q = q_perm.transpose(1, 2)
                k = k_perm.transpose(1, 2)

        BH = B * H
        q_flat = q.reshape(BH, S, d_h)
        k_flat = k.reshape(BH, S, d_h)

        device = x.device

        if topo_idx is not None:
            idx = topo_idx
            K = idx.shape[-1]
        else:
            K = S
            idx = torch.arange(S, device=device).view(1, 1, S).expand(B, S, S)

        arange_s = torch.arange(S, device=device).view(1, S, 1)
        idx_causal = torch.where(idx > arange_s, arange_s.expand_as(idx), idx)

        # gather keys by idx_causal
        idx_flat = idx_causal.view(B, S * K)            # [B,S*K]
        idx_flat_bh = idx_flat.unsqueeze(1).expand(B, H, S * K).reshape(BH, S * K)  # [BH,S*K]

        k_sel = k_flat.gather(
            dim=1,
            index=idx_flat_bh.unsqueeze(-1).expand(BH, S * K, d_h),
        )  # [BH,S*K,d_h]

        # replicate queries per K
        q_rep = q_flat.unsqueeze(2).expand(BH, S, K, d_h).reshape(BH, S * K, d_h)  # [BH,S*K,d_h]

        # ----- Geodesic distances on Poincaré & Lorentz -----
        # reshape to [N,d_h]
        N_pairs = BH * S * K
        q_pairs = q_rep.reshape(N_pairs, d_h)
        k_pairs = k_sel.reshape(N_pairs, d_h)

        # Poincaré: project to ball and compute distance
        c_p = 1e-3  # 사용 중인 기본 곡률 (config 와 동일하게)
        q_p = project_to_ball(q_pairs)
        k_p = project_to_ball(k_pairs)
        d_p = poincare_distance(q_p, k_p, c_p)  # [N_pairs]

        # Lorentz: Poincaré 좌표를 hyperboloid 로 올린 뒤 거리
        c_l = abs(float(self.c)) if self.c is not None else c_p
        q_l = from_poincare(q_p, c=c_p)
        k_l = from_poincare(k_p, c=c_p)
        d_l = lorentz_distance(q_l, k_l, c_l)  # [N_pairs]

        # Product manifold distance^2 = λ_p d_p^2 + λ_l d_l^2
        lambda_p = 0.5
        lambda_l = 0.5
        d2_total = lambda_p * (d_p ** 2) + lambda_l * (d_l ** 2)
        d2 = d2_total.reshape(B, H, S, K)  # [B,H,S,K]

        # scores = -d^2 / τ
        tau = max(self.attn.tau, 1e-6)
        scores = -d2 / tau  # [B,H,S,K]

        # softmax over K (Top‑k 이웃들)
        a = torch.softmax(scores, dim=-1)  # [B,H,S,K]

        # values gather & aggregation
        # v: [B,H,S,d_h], idx_causal: [B,S,K]
        v_flat = v  # 그대로 사용
        BH, S_v, Dh_v = BH, S, d_h
        v_flat2 = v_flat.reshape(BH, S_v, Dh_v)
        idx_flat2 = idx_causal.unsqueeze(1).expand(B, H, S, K).reshape(BH, S * K)
        v_g = v_flat2.gather(
            dim=1,
            index=idx_flat2.unsqueeze(-1).expand(BH, S * K, Dh_v),
        )  # [BH,S*K,Dh_v]
        v_sel = v_g.reshape(B, H, S, K, Dh_v)  # [B,H,S,K,Dh_v]

        y = (a.unsqueeze(-1) * v_sel).sum(dim=3)  # [B,H,S,Dh_v]

        # 합쳐서 출력 proj
        y = y.transpose(1, 2).contiguous().view(B, S, self.d_model)
        y = self.out_proj(y)

        x = x + y
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class HierarchicalSentenceTopicLLM(nn.Module):
    def __init__(self, config: HierarchicalLLMConfig) -> None:
        super().__init__()
        self.config = config

        # L0: Riemannian Aggregation (bottom-up encoding)
        self.sentence_aggregator = RiemannianAggregation(
            d_model=config.d_model,
            manifold=config.manifold_sentence,
            c=config.c_poincare,
            temperature=config.temperature_agg,
        )
        
        self.paragraph_aggregator = RiemannianAggregation(
            d_model=config.d_model,
            manifold=config.manifold_paragraph,
            c=config.c_poincare,
            temperature=config.temperature_agg,
        )

        # 문단 레벨 컨트롤러: 문단 임베딩 → 발화할 문장 수 분포
        self.paragraph_length_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.max_answer_sentences),
        )

        # L1: SentenceTopicHead (Poincaré + MetricAttention)
        self.topic_head = SentenceTopicHead(
            d_model=config.d_model,
            d_head=config.d_head,
            num_topics=config.num_topics,
            num_heads=config.num_heads_topic,
            c_poincare=config.c_poincare,
        )

        # L2: MetricContextRouter (MetriKey 기반 SPD metric slots)
        self.metric_router = MetricContextRouter(
            d_head=config.d_head,
            lambda_min=0.1,
            lambda_max=5.0,
        )
        
        # L2.5: SPD Metric Mixer (barycenter-based mixing)
        self.metric_mixer = SPDMetricMixer(
            d_head=config.d_head,
            gamma_up=config.gamma_up,
            gamma_self=config.gamma_self,
            gamma_down=config.gamma_down,
        )

        if config.use_pretrained_embeddings:
            self.backbone = PretrainedBackbone(
                model_name="klue/bert-base",
                freeze=config.freeze_decoder,
                d_model=config.d_model
            )
            self.token_embed = self.backbone
            config.vocab_size = self.backbone.get_vocab_size()
        else:
            self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # L3: HierarchicalLMDecoder (geodesic MetricAttention, 순수 LM)
        self.decoder = HierarchicalLMDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layer=config.n_layer_decoder,
            n_head=config.n_head_decoder,
            manifold="lorentz",
            c_lorentz=config.c_lorentz,
        )
        
        # Decoder와 Embedding 공유
        self.decoder.token_embed = self.token_embed
        self.semantic_loss = SemanticPreservationLoss(
            manifold=config.manifold_sentence,
            c=config.c_poincare,
        )
        self.edit_head = EditOperationHead(config.d_model, num_ops=5)
        self.sentence_order_head = SentenceOrderHead(config.d_model)
        
        # Freeze backbone if specified (문서 7.1절: pretrain 후 거의 고정)
        # 현재는 pretrain이 없으므로 freeze하지 않음
        if config.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print("[Init] Decoder frozen (requires pretrained weights)")
        
        if config.freeze_topic_head_backbone:
            # Freeze all except metric-related parameters
            for name, param in self.topic_head.named_parameters():
                if "metric" not in name.lower() and "spd" not in name.lower():
                    param.requires_grad = False
            print("[Init] TopicHead backbone frozen (requires pretrained weights)")

        if config.pretrained_decoder_path:
            state = torch.load(config.pretrained_decoder_path)
            self.decoder.load_state_dict(state['decoder'])
            if config.freeze_decoder:
                for p in self.decoder.parameters():
                    p.requires_grad = False

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> "HierarchicalSentenceTopicLLM":
        """
        scripts/train.py 에서 사용하던 checkpoint dict 로부터 모델을 재구성하는 helper.

        checkpoint 형식:
            {
                "config": {...},          # 기존 train config dict
                "topic_head": state_dict,
                "decoder": state_dict,
                ...
            }
        """
        cfg_dict = checkpoint["config"]
        cfg = HierarchicalLLMConfig(
            vocab_size=cfg_dict["vocab_size"],
            d_model=cfg_dict["d_model"],
            d_head=cfg_dict["d_head"],
            num_topics=cfg_dict["num_topics"],
            num_heads_topic=cfg_dict["num_heads"],
            n_layer_decoder=cfg_dict["n_layer"],
            n_head_decoder=cfg_dict["n_head"],
        )
        model = cls(cfg)
        model.topic_head.load_state_dict(checkpoint["topic_head"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        return model

    def encode_tokens_to_sentences(
        self,
        tokens: torch.Tensor,  # [B, T, L]
        metric_ctx_sentence: Optional[torch.Tensor] = None,  # [B, T, d, d]
    ) -> torch.Tensor:
        """
        토큰 → 문장 상향식 인코딩 (Riemannian message passing).
        
        h_sentence = RiemannAgg({h_token : token ∈ sentence}; M_sentence, G_sentence)
        
        Args:
            tokens: [B, T, L] 토큰 ID 텐서
            metric_ctx_sentence: [B, T, d, d] 문장별 SPD 메트릭 (optional)
            
        Returns:
            sentence_embeddings: [B, T, d_model]
        """
        B, T, L = tokens.shape
        
        # 토큰 임베딩 (Decoder와 공유)
        # CUDA assert 방지: 음수 및 범위 밖 값 제거
        tokens_clamped = tokens.clamp(min=0, max=self.config.vocab_size - 1)  # [B, T, L]
        
        # PretrainedBackbone은 [B*T, L]로 reshape 필요
        if isinstance(self.token_embed, PretrainedBackbone):
            tokens_flat_input = tokens_clamped.view(B * T, L)  # [B*T, L]
            token_embeddings_flat = self.token_embed(tokens_flat_input)  # [B*T, L, d_model]
            token_embeddings = token_embeddings_flat.view(B, T, L, self.config.d_model)  # [B, T, L, d_model]
        else:
            token_embeddings = self.token_embed(tokens_clamped)  # [B, T, L, d_model]
        
        # 문장별로 토큰들을 Riemannian aggregation
        # 배치 연산으로 최적화: [B, T, L, d_model] -> [B*T, L, d_model]
        BT = B * T
        tokens_flat = token_embeddings.reshape(BT, L, self.config.d_model)  # [B*T, L, d_model]
        
        if metric_ctx_sentence is not None:
            # [B, T, d, d] -> [B*T, d, d]
            metric_flat = metric_ctx_sentence.reshape(BT, metric_ctx_sentence.size(-2), metric_ctx_sentence.size(-1))
        else:
            metric_flat = None
        
        # 한번에 aggregation
        sentence_embeddings_flat = self.sentence_aggregator(
            tokens_flat,  # [B*T, L, d_model]
            metric_ctx=metric_flat,
        )  # [B*T, d_model]
        
        sentence_embeddings = sentence_embeddings_flat.reshape(B, T, self.config.d_model)  # [B, T, d_model]
        return sentence_embeddings
    
    def encode_sentences_to_paragraph(
        self,
        sentence_embeddings: torch.Tensor,  # [B, T, d_model]
        metric_ctx_paragraph: Optional[torch.Tensor] = None,  # [B, d, d]
    ) -> torch.Tensor:
        """
        문장 → 문단 상향식 인코딩 (Riemannian message passing).
        
        h_paragraph = RiemannAgg({h_sentence : sentence ∈ paragraph}; M_paragraph, G_paragraph)
        
        Args:
            sentence_embeddings: [B, T, d_model]
            metric_ctx_paragraph: [B, d, d] 문단 SPD 메트릭 (optional)
            
        Returns:
            paragraph_embedding: [B, d_model]
        """
        # RiemannAgg
        paragraph_embedding = self.paragraph_aggregator(
            sentence_embeddings,  # [B, T, d_model]
            metric_ctx=metric_ctx_paragraph,
        )  # [B, d_model]
        return paragraph_embedding

    def encode_sentences(
        self,
        tokens: torch.Tensor,  # [B, T, L]
        metric_ctx_sentence: Optional[torch.Tensor] = None,  # [B, T, d_h, d_h]
    ) -> torch.Tensor:
        """
        호환성 helper:
        - 기존 QA/인덱싱 유틸에서 사용하던 encode_sentences(tokens)를
          현재 구현의 encode_tokens_to_sentences로 연결한다.

        Args:
            tokens: [B, T, L] 토큰 ID 텐서
            metric_ctx_sentence: [B, T, d_h, d_h] 문장별 SPD 메트릭 (선택)

        Returns:
            sentence_embeddings: [B, T, d_model]
        """
        return self.encode_tokens_to_sentences(
            tokens,
            metric_ctx_sentence=metric_ctx_sentence,
        )


    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        한 배치에 대해 전체 L1–L3 파이프라인을 통과시키고,
        (원하면) 토픽 + LM loss 를 함께 반환한다.

        Args:
            batch:
                - "tokens": [B, T, L]
                - "topo_idx": [B, T, K]
            compute_loss: 손실 계산 여부

        Returns:
            logits: [B, S, V] (토큰 시퀀스에 대한 다음 토큰 분포, S = T*L 또는 LM 시퀀스 길이)
            info: {
                "P_topic": [B, T, C],
                "scores": [B, T],
                "metric_keys": List[str],
                "metric_ctx": [B, T, d_h, d_h],
                "logits": [B, T, V],
                "hidden": [B, T, d_model],
                (옵션) "loss", "loss_lm", "loss_consistency", "loss_diversity"
            }
        """
        tokens = batch["tokens"]          # [B, T, L]
        topo_idx = batch["topo_idx"]      # [B, T, K]

        device = next(self.parameters()).device
        tokens = tokens.to(device)
        topo_idx = topo_idx.to(device)

        B, T, L = tokens.shape

        # ========== 상향식 인코딩 (Bottom-up) ==========
        # Step 1: 토큰 → 문장 (Riemannian message passing)
        # 초기에는 metric 없이 인코딩
        sentence_embeddings_raw = self.encode_tokens_to_sentences(tokens)  # [B, T, d_model]
        
        # Step 2: 문장 → 주제/메트릭 키 (SentenceTopicHead)
        P_topic, scores, metric_keys = self.topic_head(sentence_embeddings_raw, topo_idx)
        C = P_topic.size(-1)

        # 문단 내 consistency: KL(P_topic || paragraph_mean)
        paragraph_mean = P_topic.mean(dim=1, keepdim=True).detach()  # [B,1,C]
        paragraph_mean = paragraph_mean.expand(-1, T, -1)
        log_p = (P_topic + 1e-10).log()
        loss_consistency = nn.KLDivLoss(reduction="batchmean")(log_p, paragraph_mean)
        
        # 배치 전체 diversity: KL(batch_mean || uniform)
        batch_mean = P_topic.mean(dim=(0, 1))  # [C]
        uniform = torch.full_like(batch_mean, 1.0 / C)
        loss_diversity = nn.KLDivLoss(reduction="batchmean")(
            (batch_mean + 1e-10).log(), uniform
        )

        # Step 3: MetriKey → SPD 메트릭 (MetricContextRouter)
        metric_ctx_sentence = self.metric_router(metric_keys, scores)  # [B, T, d_h, d_h]
        
        # Step 4: 문장 → 문단 (Riemannian message passing with metric)
        # 문단 메트릭: 문장 메트릭들의 평균
        metric_ctx_paragraph = metric_ctx_sentence.mean(dim=1)  # [B, d_h, d_h]
        
        # 메트릭 적용하여 재인코딩
        sentence_embeddings = self.encode_tokens_to_sentences(
            tokens,
            metric_ctx_sentence=metric_ctx_sentence,
        )  # [B, T, d_model]
        
        paragraph_embedding = self.encode_sentences_to_paragraph(
            sentence_embeddings,
            metric_ctx_paragraph=metric_ctx_paragraph,
        )  # [B, d_model]
        
        # 문단 임베딩 기반 문장 수 분포 (paragraph-level controller)
        length_logits = self.paragraph_length_head(paragraph_embedding)  # [B, max_answer_sentences]
        sentence_order_scores = self.sentence_order_head(sentence_embeddings)
        
        # Step 5: 상·하위 메트릭 혼합 (SPD barycenter)
        # parent_metric: [B, d_h, d_h] -> [B, 1, d_h, d_h] -> [B, T, d_h, d_h]
        parent_metric_expanded = metric_ctx_paragraph.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, d_h, d_h]

        # children_metrics: 각 문장의 이웃(시간/순서 기반)을 "자식"으로 간주하여
        # SPD 바리센터 혼합에 포함시킨다.
        # topo_idx: [B, T, K] 에 대해,
        #   children_metrics[b, t] = metric_ctx_sentence[b, topo_idx[b, t, :]]
        children_metrics: Optional[torch.Tensor]
        if topo_idx.numel() > 0 and metric_ctx_sentence.numel() > 0:
            B_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(topo_idx)  # [B, T, K]
            # 패딩된 topo_idx 가 T 범위를 벗어나지 않도록 클램프
            sent_idx = topo_idx.clamp(min=0, max=T - 1)
            children_metrics = metric_ctx_sentence[B_idx, sent_idx]  # [B, T, K, d_h, d_h]
            BT = B * T
            parent_flat = parent_metric_expanded.reshape(BT, self.config.d_head, self.config.d_head)
            self_flat = metric_ctx_sentence.reshape(BT, self.config.d_head, self.config.d_head)
            children_flat = children_metrics.reshape(
                BT,
                children_metrics.size(2),
                self.config.d_head,
                self.config.d_head,
            )  # [B*T, K, d_h, d_h]
            effective_flat = self.metric_mixer.mix_hierarchy(
                parent_metric=parent_flat,
                self_metric=self_flat,
                children_metrics=children_flat,
            )  # [B*T, d_h, d_h]
        else:
            # 안전 장치: children 이 없으면 parent/self 만 사용
            BT = B * T
            parent_flat = parent_metric_expanded.reshape(BT, self.config.d_head, self.config.d_head)
            self_flat = metric_ctx_sentence.reshape(BT, self.config.d_head, self.config.d_head)
            effective_flat = self.metric_mixer.mix_hierarchy(
                parent_metric=parent_flat,
                self_metric=self_flat,
                children_metrics=None,
            )  # [B*T, d_h, d_h]

        metric_ctx = effective_flat.reshape(B, T, self.config.d_head, self.config.d_head)  # [B, T, d_h, d_h]

        # ===== L3: HierarchicalLMDecoder (순수 LM, 토큰 시퀀스 전체를 학습) =====
        # 토큰/메트릭/토폴로지를 토큰 단위 시퀀스로 평탄화
        S_full = T * L
        tokens_flat = tokens.clamp(min=0, max=self.config.vocab_size - 1).view(B, S_full)  # [B, S_full]

        # 문장 메트릭을 토큰 수준으로 브로드캐스트
        metric_ctx_flat_full = (
            metric_ctx  # [B, T, d_h, d_h]
            .unsqueeze(2)  # [B, T, 1, d_h, d_h]
            .expand(B, T, L, self.config.d_head, self.config.d_head)
            .contiguous()
            .view(B, S_full, self.config.d_head, self.config.d_head)
        )  # [B, S_full, d_h, d_h]

        # topology index를 토큰 수준으로 변환
        # topo_idx: [B, T, K] - 문장 인덱스 (0..T-1)
        # 토큰 인덱스로 변환: sent_idx * L + token_offset
        # 각 문장의 첫 토큰 위치로 매핑 (간단한 근사)
        K = topo_idx.size(-1)
        topo_idx_token = topo_idx * L  # [B, T, K] - 각 문장의 시작 토큰 인덱스
        
        # 이를 토큰 수준으로 브로드캐스트
        topo_idx_flat_full = (
            topo_idx_token
            .unsqueeze(2)  # [B, T, 1, K]
            .expand(B, T, L, K)
            .contiguous()
            .view(B, S_full, K)
        )  # [B, S_full, K]
        
        # 각 토큰 위치에서 자신의 문장 내 offset을 더해 정확한 이웃 토큰 인덱스 생성
        token_offset = torch.arange(L, device=device).view(1, 1, L, 1).expand(B, T, L, K)
        token_offset_flat = token_offset.contiguous().view(B, S_full, K)
        topo_idx_flat_full = (topo_idx_flat_full + token_offset_flat).clamp(min=0, max=S_full - 1)

        # LM 시퀀스 길이 상한 적용 (메모리 보호)
        if S_full > self.config.max_lm_seq_len:
            S = self.config.max_lm_seq_len
            tokens_flat = tokens_flat[:, :S]
            metric_ctx_flat = metric_ctx_flat_full[:, :S]
            topo_idx_flat = topo_idx_flat_full[:, :S]
        else:
            S = S_full
            metric_ctx_flat = metric_ctx_flat_full
            topo_idx_flat = topo_idx_flat_full

        logits, hidden = self.decoder(
            input_ids=tokens_flat,
            metric_ctx=metric_ctx_flat,
            topo_idx=topo_idx_flat,
        )
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        edit_logits = self.edit_head(hidden)

        info: Dict[str, torch.Tensor] = {
            "P_topic": P_topic,
            "scores": scores,
            "metric_ctx": metric_ctx,
            "logits": logits,
            "hidden": hidden,
            "edit_logits": edit_logits,
            "sentence_order_scores": sentence_order_scores,
        }
        info_str: Dict[str, object] = {
            **info,
            "metric_keys": metric_keys,
        }
        info_str["length_logits"] = length_logits

        if compute_loss:
            # 문장 non-empty 마스크 (여러 loss에서 재사용)
            sentence_nonempty = (tokens > 0).any(dim=-1)  # [B, T]

            # 문장 수 예측 loss (문단 레벨)
            true_lengths = sentence_nonempty.sum(dim=1)   # [B]
            # 최소 1문장, 최대 max_answer_sentences 로 클램프 후 0-base 인덱스로 변환
            length_targets = true_lengths.clamp(
                min=1, max=self.config.max_answer_sentences
            ) - 1  # [B]
            length_loss = F.cross_entropy(length_logits, length_targets)

            # 선택적 토픽 supervision (batch 에 topic_labels 가 있을 때만 사용)
            topic_loss = None
            topic_labels = batch.get("topic_labels")
            if topic_labels is not None:
                topic_labels_t = topic_labels.to(device)  # [B, T]
                # 패딩 문장은 ignore_index(-1) 로 마스킹
                topic_targets = topic_labels_t.clone()
                topic_targets[~sentence_nonempty] = -1
                log_p_topic = (P_topic + 1e-10).log().view(B * T, C)
                topic_targets_flat = topic_targets.view(B * T)
                topic_loss = F.nll_loss(
                    log_p_topic,
                    topic_targets_flat,
                    ignore_index=-1,
                )

            semantic_mask = sentence_nonempty.to(sentence_embeddings_raw.dtype)
            semantic_loss = self.semantic_loss(
                sentence_embeddings_raw,
                sentence_embeddings,
                mask=semantic_mask,
            )

            # ===== 논문 설계: Lexical Editing Loss =====
            # replacement_mask가 1인 위치만 예측 학습 (문장 구조 보존)
            # next-token prediction이 아닌 masked token prediction
            
            replacement_mask = batch.get("replacement_mask")  # [B, T, L]
            if replacement_mask is not None:
                # replacement_mask를 토큰 레벨로 평탄화
                mask_flat = replacement_mask.view(B, T * L)  # [B, S]
                S_max = min(mask_flat.size(1), logits.size(1))
                mask_flat = mask_flat[:, :S_max]  # [B, S_max]
                
                # logits와 targets를 같은 위치에서 비교 (next-token이 아님!)
                logits_pred = logits[:, :S_max, :]  # [B, S_max, V]
                targets = tokens_flat[:, :S_max].clamp(0, self.config.vocab_size - 1)  # [B, S_max]
                
                # replacement_mask가 1인 위치만 loss 계산
                V = logits_pred.size(-1)
                logits_flat = logits_pred.reshape(-1, V)  # [B*S_max, V]
                targets_flat = targets.reshape(-1)  # [B*S_max]
                mask_binary = mask_flat.reshape(-1).bool()  # [B*S_max]
                
                if mask_binary.any():
                    # 마스크된 위치만 선택
                    logits_masked = logits_flat[mask_binary]  # [N_masked, V]
                    targets_masked = targets_flat[mask_binary]  # [N_masked]
                    
                    lm_loss = F.cross_entropy(
                        logits_masked,
                        targets_masked,
                        ignore_index=0,  # PAD 무시
                    )
                else:
                    lm_loss = torch.tensor(0.0, device=device)
            else:
                # replacement_mask가 없으면 전체 시퀀스에 대해 학습 (fallback)
                S = tokens_flat.size(1)
                S_max = min(S, logits.size(1))
                logits_pred = logits[:, :S_max, :]
                targets = tokens_flat[:, :S_max].clamp(0, self.config.vocab_size - 1)
                
                V = logits_pred.size(-1)
                lm_loss = F.cross_entropy(
                    logits_pred.reshape(-1, V),
                    targets.reshape(-1),
                    ignore_index=0,
                )

            # NaN 방지: 각 loss 성분을 안전하게 정규화
            lm_loss = torch.nan_to_num(lm_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss_consistency = torch.nan_to_num(
                loss_consistency, nan=0.0, posinf=0.0, neginf=0.0
            )
            loss_diversity = torch.nan_to_num(
                loss_diversity, nan=0.0, posinf=0.0, neginf=0.0
            )
            length_loss = torch.nan_to_num(
                length_loss, nan=0.0, posinf=0.0, neginf=0.0
            )
            if topic_loss is not None:
                topic_loss = torch.nan_to_num(
                    topic_loss, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Metric regularization: ||G - I||_F^2, G = L L^T
            d_h = self.config.d_head
            eye = torch.eye(d_h, device=device, dtype=metric_ctx.dtype)
            G_sentence = metric_ctx_sentence.reshape(B * T, d_h, d_h)
            G_sentence = G_sentence @ G_sentence.transpose(-2, -1)
            diff_G = G_sentence - eye
            loss_metric = (diff_G.pow(2).sum(dim=(-2, -1))).mean()

            # Curvature regularization: (c - c_target)^2 for Poincaré / Lorentz
            c_p = float(self.config.c_poincare)
            c_p_target = float(self.config.curvature_target_poincare)
            loss_curv_p = (c_p - c_p_target) ** 2
            c_l = float(self.config.c_lorentz)
            c_l_target = float(self.config.curvature_target_lorentz)
            loss_curv_l = (c_l - c_l_target) ** 2
            loss_curvature = torch.as_tensor(
                loss_curv_p + loss_curv_l, device=device, dtype=logits.dtype
            )

            # 최종 loss 구성
            loss = (
                lm_loss
                + self.config.lambda_consistency * loss_consistency
                + self.config.lambda_diversity * loss_diversity
                + self.config.lambda_length * length_loss
            )
            if self.config.lambda_metric > 0.0:
                loss = loss + self.config.lambda_metric * loss_metric
                info_str["loss_metric"] = loss_metric
            if self.config.lambda_curvature > 0.0:
                loss = loss + self.config.lambda_curvature * loss_curvature
                info_str["loss_curvature"] = loss_curvature
            if topic_loss is not None and self.config.lambda_topic_supervision > 0.0:
                loss = loss + self.config.lambda_topic_supervision * topic_loss
                info_str["loss_topic"] = topic_loss
            if self.config.lambda_semantic > 0.0:
                semantic_loss = torch.nan_to_num(
                    semantic_loss, nan=0.0, posinf=0.0, neginf=0.0
                )
                loss = loss + self.config.lambda_semantic * semantic_loss
                info_str["loss_semantic"] = semantic_loss

            if self.config.lambda_edit > 0.0:
                num_ops = edit_logits.size(-1)
                probs_edit = F.softmax(edit_logits, dim=-1)
                cost_vec = torch.tensor(
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    device=probs_edit.device,
                    dtype=probs_edit.dtype,
                )
                expected_cost = (probs_edit * cost_vec.view(1, 1, num_ops)).sum(dim=-1)
                loss_edit = expected_cost.mean()
                loss = loss + self.config.lambda_edit * loss_edit
                info_str["loss_edit"] = loss_edit

            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            info_str["loss"] = loss
            info_str["loss_lm"] = lm_loss
            info_str["loss_consistency"] = loss_consistency
            info_str["loss_diversity"] = loss_diversity
            info_str["loss_length"] = length_loss

        return logits, info_str  # type: ignore[return-value]


def train_hierarchical_llm_from_text(
    data_path: str,
    config: Optional[HierarchicalLLMConfig] = None,
    epochs: int = 50,
    batch_size: int = 4,
    max_paragraphs: int = 1000,
    device: Optional[str] = None,
) -> Tuple[HierarchicalSentenceTopicLLM, Dict[str, object]]:
    if config is None:
        config = HierarchicalLLMConfig()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    if not _HAS_SENTENCE_TOPIC_DATASET:
        raise RuntimeError(
            "SentenceTopicDataset/ collate_batch 가 로드되지 않았습니다. "
            "reality_stone.data 모듈이 제대로 설치되었는지 확인하세요."
        )

    # 모델 초기화
    model = HierarchicalSentenceTopicLLM(config).to(device_t)

    # 데이터셋/로더 구성
    dataset = SentenceTopicDataset(data_path, max_paragraphs=max_paragraphs)
    from torch.utils.data import DataLoader  # local import

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # Optimizer: 메트릭 슬롯에 더 큰 LR, 백본에 작은 LR
    # (pretrain 없이는 백본도 함께 학습해야 함)
    
    # Metric-related parameters (high LR)
    metric_params = []
    for name, param in model.topic_head.named_parameters():
        if param.requires_grad and ("metric" in name.lower() or "spd" in name.lower()):
            metric_params.append(param)
    metric_params.extend(model.metric_mixer.parameters())
    metric_params.extend(model.sentence_aggregator.parameters())
    metric_params.extend(model.paragraph_aggregator.parameters())
    
    # Backbone parameters (low LR)
    backbone_params = []
    for name, param in model.topic_head.named_parameters():
        if param.requires_grad and not ("metric" in name.lower() or "spd" in name.lower()):
            backbone_params.append(param)
    backbone_params.extend(model.decoder.parameters())
    
    # Filter only trainable
    metric_params = [p for p in metric_params if p.requires_grad]
    backbone_params = [p for p in backbone_params if p.requires_grad]
    
    if len(metric_params) == 0 and len(backbone_params) == 0:
        raise RuntimeError("No trainable parameters found.")
    
    print(f"[Training] Metric parameters: {sum(p.numel() for p in metric_params)} (LR={config.lr_metric})")
    print(f"[Training] Backbone parameters: {sum(p.numel() for p in backbone_params)} (LR={config.lr_backbone})")
    print(f"[Training] Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer with different LRs
    optimizer = torch.optim.AdamW([
        {'params': metric_params, 'lr': config.lr_metric},
        {'params': backbone_params, 'lr': config.lr_backbone},
    ])

    model.train()
    total_loss = 0.0

    from tqdm import tqdm  # local import

    for epoch in range(epochs):
        # 동적 lambda 계산
        lambda_consistency_current = compute_dynamic_lambda(
            config.lambda_consistency,
            config.lambda_consistency_schedule,
            epoch,
            epochs,
        )
        lambda_diversity_current = compute_dynamic_lambda(
            config.lambda_diversity,
            config.lambda_diversity_schedule,
            epoch,
            epochs,
        )
        
        # 모델의 lambda 업데이트 (forward에서 사용)
        model.config.lambda_consistency = lambda_consistency_current
        model.config.lambda_diversity = lambda_diversity_current
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Hierarchical LLM epoch {epoch+1}/{epochs}")
        for batch in pbar:
            try:
                optimizer.zero_grad()
                logits, info = model(batch, compute_loss=True)
                loss = info["loss"]  # type: ignore[index]
                assert isinstance(loss, torch.Tensor)
                loss.backward()
                
                # Gradient clipping (레이어별)
                torch.nn.utils.clip_grad_norm_(metric_params, 0.5)  # 메트릭은 더 보수적
                torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
                
                optimizer.step()
                epoch_loss += float(loss.item())
                pbar.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    λ_cons=f"{lambda_consistency_current:.3f}",
                    λ_div=f"{lambda_diversity_current:.3f}",
                )
            except Exception as e:  # pragma: no cover - 안전 장치
                print(f"[train_hierarchical_llm_from_text] Error in batch: {e}")
                continue
        epoch_loss /= max(len(dataloader), 1)
        print(
            f"[Hierarchical LLM] epoch {epoch+1}/{epochs}, "
            f"loss={epoch_loss:.4f}, "
            f"λ_consistency={lambda_consistency_current:.3f}, "
            f"λ_diversity={lambda_diversity_current:.3f}"
        )
        total_loss = epoch_loss

    info_out: Dict[str, object] = {
        "final_loss": total_loss,
        "num_samples": len(dataset),
        "config": config,
    }
    return model, info_out


def infer_hierarchical_llm_on_text(
    model: HierarchicalSentenceTopicLLM,
    text: str,
    max_length: int = 128,
    k_neighbors: int = 3,
    max_new_tokens: int = 20,
) -> Dict[str, object]:
    """
    PreSegmenter 를 사용해 단일 문단 텍스트에 대해
    계층적 Sentence-Topic LLM 의 추론을 수행하는 헬퍼 (생성 모드).

    Returns:
        {
            "original_text": str,
            "sentences": List[str],
            "generated_text": str,
            "topics": [...],
        }
    """
    from reality_stone.utils.pre_segmenter import PreSegmenter  # local import

    device = next(model.parameters()).device

    segmenter = PreSegmenter(max_length=max_length, k_neighbors=k_neighbors)
    seg_output = segmenter(text)

    if seg_output["metadata"]["num_sentences"] == 0:
        return {
            "original_text": text,
            "sentences": [],
            "generated_text": text,
            "topics": [],
        }

    # L0 output -> batch dict 형식으로 변환
    tokens = seg_output["tokens"].unsqueeze(0).to(device)   # [1, T, L] -> model device
    topo_idx = seg_output["topo_idx"].unsqueeze(0).to(device)

    batch: Dict[str, torch.Tensor] = {"tokens": tokens, "topo_idx": topo_idx}

    # 계층적 인코딩 + 메트릭 컨텍스트 + LM 디코딩을 한 번에 수행
    model.eval()
    with torch.no_grad():
        logits, info = model(batch, compute_loss=False)

    original_sentences: List[str] = seg_output["sentences"]
    tokenizer = segmenter.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0

    # ===== 문장 템플릿 유지 + 단어 치환 모드 =====
    # logits: [1, S, V] (이미 디코더를 거친 결과)
    # tokens: [1, T, L], replacement_mask: [T, L]
    replacement_mask = seg_output["replacement_mask"].unsqueeze(0).to(device)  # [1, T, L]

    B, T, L = tokens.shape
    S = T * L  # 전체 토큰 길이

    # 토큰/마스크를 평탄화
    input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)          # [1, S]
    mask_flat = replacement_mask.view(1, S)                                           # [1, S]

    # 문장별 메트릭 컨텍스트를 토큰 수준으로 브로드캐스트 (autoregressive 생성용)
    metric_ctx_sent = info.get("metric_ctx", None)  # [1, T, d_h, d_h]
    if isinstance(metric_ctx_sent, torch.Tensor):
        d_h = metric_ctx_sent.size(-1)
        metric_ctx_token = (
            metric_ctx_sent  # [1, T, d_h, d_h]
            .unsqueeze(2)    # [1, T, 1, d_h, d_h]
            .expand(1, T, L, d_h, d_h)
            .contiguous()
            .view(1, S, d_h, d_h)
        )  # [1, S, d_h, d_h]
    else:
        metric_ctx_token = None

    # topology index 역시 토큰 수준으로 브로드캐스트
    K_topo = topo_idx.size(-1)
    topo_idx_token_base = topo_idx * L  # 문장 시작 토큰 인덱스
    topo_idx_token = (
        topo_idx_token_base
        .unsqueeze(2)  # [1, T, 1, K]
        .expand(1, T, L, K_topo)
        .contiguous()
        .view(1, S, K_topo)
    )  # [1, S, K]

    # logits에서 가장 확률 높은 토큰 선택 (이미 forward에서 계산됨)
    # logits: [1, S_actual, V] (max_lm_seq_len로 잘렸을 수 있음)
    S_actual = logits.size(1)
    if S_actual < S:
        # 잘린 경우 앞부분만 사용
        input_ids_flat = input_ids_flat[:, :S_actual]
        mask_flat = mask_flat[:, :S_actual]
    
    pred_ids_flat = torch.argmax(logits, dim=-1)  # [1, S_actual]

    # Lexical constraint: 교체 가능(=1) 토큰만 새 토큰으로 바꾸고,
    # 나머지는 원본을 그대로 유지 → 문장 구조/문법 보존
    edited_flat = torch.where(mask_flat.bool(), pred_ids_flat, input_ids_flat)  # [1, S_actual]

    tokens_seq = edited_flat

    if getattr(model.config, "enable_structural_edit", False):
        edit_logits = info.get("edit_logits")
        if isinstance(edit_logits, torch.Tensor):
            ops = torch.argmax(edit_logits[:, :S_actual, :], dim=-1)
            max_edits = max(1, int(S_actual * float(model.config.max_edit_ratio)))
            new_tokens = []
            insert_count = 0
            delete_count = 0
            for i in range(S_actual):
                op = int(ops[0, i].item())
                base_tok = int(tokens_seq[0, i].item())
                if op == 4 and delete_count < max_edits:
                    delete_count += 1
                    continue
                if op == 2 and insert_count < max_edits:
                    new_tokens.append(int(pred_ids_flat[0, i].item()))
                    insert_count += 1
                new_tokens.append(base_tok)
                if op == 3 and insert_count < max_edits:
                    new_tokens.append(int(pred_ids_flat[0, i].item()))
                    insert_count += 1
            if not new_tokens:
                new_tokens = tokens_seq[0].tolist()
            tokens_seq = torch.tensor(new_tokens, device=device).unsqueeze(0)
            S_actual = tokens_seq.size(1)

    final_ids_flat = tokens_seq[0].tolist()
    if tokenizer is not None:
        try:
            if getattr(model.config, "enable_structural_edit", False):
                token_ids_no_pad = [tid for tid in final_ids_flat if tid != pad_id]
                if token_ids_no_pad:
                    generated_text = tokenizer.decode(token_ids_no_pad, skip_special_tokens=True)
                else:
                    generated_text = text
            else:
                generated_sentences: List[str] = []
                for sent_idx in range(T):
                    start_idx = sent_idx * L
                    end_idx = min(start_idx + L, len(final_ids_flat))
                    sent_token_ids = final_ids_flat[start_idx:end_idx]
                    sent_token_ids_no_pad = [tid for tid in sent_token_ids if tid != pad_id]
                    if sent_token_ids_no_pad:
                        sent_text = tokenizer.decode(sent_token_ids_no_pad, skip_special_tokens=True)
                        generated_sentences.append(sent_text)
                generated_text = " ".join(generated_sentences)
        except Exception:
            generated_text = text
    else:
        generated_text = text

    # 문단 레벨 컨트롤러가 예측한 문장 수에 맞게, 문장을 잘라서 상위 레벨에서 발화 길이를 제어
    length_logits_tensor = info.get("length_logits")
    if isinstance(length_logits_tensor, torch.Tensor):
        length_probs = torch.softmax(length_logits_tensor, dim=-1)  # [B, max_answer_sentences]
        pred_sentences = int(length_probs[0].argmax().item()) + 1   # 1..max_answer_sentences

        seg_generated = segmenter(generated_text)
        gen_sents = seg_generated.get("sentences", [])
        if gen_sents:
            order_scores = info.get("sentence_order_scores")
            if isinstance(order_scores, torch.Tensor) and not getattr(model.config, "enable_structural_edit", False):
                scores_np = order_scores[0, : len(gen_sents)].detach().cpu()
                indices = list(range(len(gen_sents)))
                indices.sort(key=lambda i: float(scores_np[i].item()), reverse=True)
                indices = indices[:pred_sentences]
                selected = [gen_sents[i] for i in indices]
                generated_text = " ".join(selected)
            else:
                generated_text = " ".join(gen_sents[:pred_sentences])

    P_topic = info.get("P_topic")
    metric_keys = info.get("metric_keys", [])

    topic_entries: List[Dict[str, object]] = []
    if isinstance(P_topic, torch.Tensor):
        topic_names = model.topic_head.topic_names
        for i, sent in enumerate(original_sentences):
            if i >= P_topic.size(1):
                break
            probs = P_topic[0, i]
            top_idx = int(probs.argmax().item())
            entry = {
                "sentence": sent,
                "topic": topic_names[top_idx],
                "confidence": float(probs[top_idx].item()),
                "metric_key": metric_keys[i] if i < len(metric_keys) else None,
            }
            topic_entries.append(entry)

    return {
        "original_text": text,
        "sentences": original_sentences,
        "generated_text": generated_text,
        "topics": topic_entries,
    }


def build_sentence_index_from_corpus(
    model: HierarchicalSentenceTopicLLM,
    data_path: str,
    max_paragraphs: int = 1000,
) -> List[Dict[str, object]]:
    if not _HAS_SENTENCE_TOPIC_DATASET:
        raise RuntimeError(
            "SentenceTopicDataset 이 로드되지 않았습니다. scripts/train.py 위치를 확인하세요."
        )
    device = next(model.parameters()).device
    dataset = SentenceTopicDataset(data_path, max_paragraphs=max_paragraphs)
    index: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            tokens = sample["tokens"].unsqueeze(0).to(device)          # [1, T, L]
            topo_idx = sample["topo_idx"].unsqueeze(0).to(device)      # [1, T, K]
            sentences: List[str] = sample["sentences"]
            sent_emb = model.encode_sentences(tokens)                   # [1, T, d_model]
            z = model.topic_head.poincare_embed(sent_emb)               # [1, T, d_head]
            z = project_to_ball(z)                                      # ball projection
            P_topic, _, metric_keys = model.topic_head(sent_emb, topo_idx)
            T = len(sentences)
            for t in range(T):
                entry: Dict[str, object] = {
                    "paragraph": sample["paragraph"],
                    "sentence": sentences[t],
                    "z": z[0, t].detach().cpu(),          # [d_head]
                    "topic_probs": P_topic[0, t].detach().cpu(),
                    "metric_key": metric_keys[t] if t < len(metric_keys) else None,
                }
                index.append(entry)

    return index


def answer_question_from_corpus(
    model: HierarchicalSentenceTopicLLM,
    question: str,
    data_path: str,
    max_paragraphs: int = 1000,
    top_k: int = 3,
) -> Dict[str, object]:
    # NOTE: 테스트 코드에서 이 import 라인을 직접 검사한다.
    from reality_stone.utils.pre_segmenter import PreSegmenter  # noqa: F401

    index = build_sentence_index_from_corpus(
        model, data_path=data_path, max_paragraphs=max_paragraphs
    )
    if not index:
        return {"question": question, "answers": [], "support": []}
    device = next(model.parameters()).device
    segmenter = PreSegmenter(max_length=128, k_neighbors=3)
    seg_q = segmenter(question)
    if seg_q["metadata"]["num_sentences"] == 0:
        return {"question": question, "answers": [], "support": []}

    q_tokens = seg_q["tokens"].unsqueeze(0).to(device)  # [1, Tq, Lq]
    q_tokens_first = q_tokens[:, :1, :]                 # [1, 1, Lq]
    with torch.no_grad():
        q_emb = model.encode_sentences(q_tokens_first)          # [1,1,d_model]
        q_z = model.topic_head.poincare_embed(q_emb)            # [1,1,d_head]
        q_z = project_to_ball(q_z)[0, 0]                        # [d_head] - device 유지

    # 3) 코퍼스의 모든 문장 임베딩과 거리 계산 (Poincaré + Lorentz product manifold 거리)
    import torch as _torch
    z_corpus = _torch.stack([e["z"] for e in index], dim=0).to(device)  # [N,d_head] - device 통일
    # Poincaré 거리: 문서 3.3, 5.2의 d_{M^(ℓ)} 항
    c_p = float(model.config.c_poincare)
    N = z_corpus.shape[0]
    q_rep = q_z.unsqueeze(0).expand(N, -1)  # [N,d_head] - 이미 device에 있음
    d_p = poincare_distance(q_rep, z_corpus, c_p)  # [N] - 둘 다 같은 device
    # Lorentz 거리: Poincaré 임베딩을 Hyperboloid 로 올려서 second manifold 로 사용
    c_l = abs(float(model.config.c_lorentz)) if hasattr(model.config, "c_lorentz") else c_p
    q_l = from_poincare(q_rep, c=c_p)              # [N, d_l]
    z_l = from_poincare(z_corpus, c=c_p)           # [N, d_l] - device 통일
    d_l = lorentz_distance(q_l, z_l, c_l)          # [N]

    # Product manifold 거리: d_total^2 = λ_p d_p^2 + λ_l d_l^2
    lambda_p = 0.5
    lambda_l = 0.5
    dists = lambda_p * (d_p ** 2) + lambda_l * (d_l ** 2)  # [N]

    k = min(top_k, z_corpus.shape[0])
    topk_vals, topk_idx = _torch.topk(dists, k=k, largest=False)

    answers: List[Dict[str, object]] = []
    for rank, idx_i in enumerate(topk_idx.tolist(), start=1):
        e = index[idx_i]
        answers.append(
            {
                "rank": rank,
                "sentence": e["sentence"],
                "paragraph": e["paragraph"],
                "distance": float(topk_vals[rank - 1].item()),
                "metric_key": e["metric_key"],
            }
        )

    return {
        "question": question,
        "answers": answers,
        "support": [a["paragraph"] for a in answers],
    }


def answer_question_with_llm(
    model: HierarchicalSentenceTopicLLM,
    question: str,
    data_path: str,
    max_paragraphs: int = 1000,
    top_k: int = 3,
    max_new_tokens: int = 32,
) -> Dict[str, object]:
    qa_ret = answer_question_from_corpus(
        model=model,
        question=question,
        data_path=data_path,
        max_paragraphs=max_paragraphs,
        top_k=top_k,
    )
    support = qa_ret.get("support", [])
    if not support:
        prompt_text = f"질문: {question}\n\n답변:"
    else:
        context = "\n\n".join(support)
        prompt_text = f"{context}\n\n질문: {question}\n\n답변:"

    # 2) 계층적 LLM을 이용한 디코딩 (문단 단위 편집 + autoregressive 확장)
    infer_out = infer_hierarchical_llm_on_text(
        model=model,
        text=prompt_text,
        max_length=128,
        k_neighbors=3,
        max_new_tokens=max_new_tokens,
    )
    generated_text = infer_out.get("generated_text", prompt_text)

    # "답변:" 마커 기준으로 최종 answer 부분만 최대한 잘라낸다.
    answer_text = generated_text
    marker = "답변:"
    if marker in generated_text:
        idx = generated_text.rfind(marker)
        tail = generated_text[idx + len(marker) :].strip()
        if tail:
            answer_text = tail

    return {
        "question": question,
        "answer": answer_text,
        "support": support,
        "retrieval": qa_ret,
    }


__all__ = [
    "HierarchicalLLMConfig",
    "HierarchicalSentenceTopicLLM",
    "SentenceTopicHead",
    "MetricContextRouter",
    "HierarchicalLMDecoder",
    "RCELexicalDecoder",
    "train_hierarchical_llm_from_text",
    "infer_hierarchical_llm_on_text",
    "build_sentence_index_from_corpus",
    "answer_question_from_corpus",
    "answer_question_with_llm",
]


