from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import reality_stone as rs

from .riemannian_aggregation import RiemannianAggregation
from reality_stone.layers.metric_attention import MetricAttention
from reality_stone.layers.poincare import project_to_ball, poincare_distance
from reality_stone.layers.lorentz import from_poincare, lorentz_distance
from reality_stone.layers.suppression import HyperbolicSuppressionField
from reality_stone.models.semantic_preservation import SemanticPreservationLoss
from .pretrained_backbone import PretrainedBackbone
from reality_stone.utils.pre_segmenter import PreSegmenter, TreeNode, DocumentTree

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
    
    lambda_consistency: float = 0.5
    lambda_diversity: float = 0.1
    lambda_consistency_schedule: str = "constant"
    lambda_diversity_schedule: str = "constant"
    lambda_topic_supervision: float = 0.5
    lambda_metric: float = 0.1
    lambda_curvature: float = 0.0
    curvature_target_poincare: float = 1e-3
    curvature_target_lorentz: float = -1.0
    
    manifold_sentence: str = "poincare"
    manifold_paragraph: str = "poincare"
    temperature_agg: float = 1.0
    
    gamma_up: float = 0.3
    gamma_self: float = 0.5
    gamma_down: float = 0.2
    
    max_answer_sentences: int = 20
    lambda_length: float = 0.2
    lambda_semantic: float = 0.3
    max_lm_seq_len: int = 1024
    
    freeze_decoder: bool = False
    freeze_topic_head_backbone: bool = False
    
    lr_backbone: float = 1e-4
    lr_metric: float = 1e-3
    
    lambda_edit: float = 0.0
    max_edit_ratio: float = 0.25
    enable_structural_edit: bool = False
    edit_budget: float = 0.25
    use_fast_spd_mixing: bool = True
    
    logit_clip_value: float = 20.0
    loss_clip_max: float = 100.0
    spd_eps: float = 1e-5
    spd_eigval_min: float = 1e-5
    spd_eigval_max: float = 1e5
    spd_log_eigval_clip: float = 10.0
    metric_lambda_min: float = 0.1
    metric_lambda_max: float = 5.0
    grad_clip_norm: float = 1.0 
    
    # SFE Variable Suppression Parameters
    suppression_base: float = 0.37
    suppression_linear: float = 0.0
    suppression_hyp: float = 0.1
    suppression_scale: float = 1.0
    enable_variable_suppression: bool = True
    diffusion_steps: int = 0
    use_diffusion_hidden: bool = False
    diffusion_alpha: float = 0.9
    diffusion_dt: float = 0.1


class EditOperationHead(nn.Module):
    def __init__(self, d_model: int, num_ops: int = 5, edit_budget: float = 0.25) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_ops = num_ops
        self.edit_budget = edit_budget
        self.proj = nn.Linear(d_model, num_ops)
        self.value_proj = nn.Linear(d_model, d_model)
        for p in self.value_proj.parameters():
            p.requires_grad = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)
    
    def apply_edits(
        self,
        tokens: torch.Tensor,
        edit_logits: torch.Tensor,
        pred_tokens: torch.Tensor,
        enable_structural: bool = False,
        replacement_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S = tokens.shape
        device = tokens.device
        
        if not enable_structural:
            if replacement_mask is not None:
                return torch.where(replacement_mask.bool(), pred_tokens, tokens)
            return tokens
        
        ops = torch.argmax(edit_logits, dim=-1)
        max_edits = max(1, int(S * self.edit_budget))
        
        result_tokens = []
        for b in range(B):
            new_seq = []
            insert_count = 0
            delete_count = 0
            replace_count = 0
            
            for i in range(S):
                op = int(ops[b, i].item())
                tok = int(tokens[b, i].item())
                pred_tok = int(pred_tokens[b, i].item())
                
                if tok == 0:
                    continue
                
                is_replaceable = True
                if replacement_mask is not None:
                    is_replaceable = bool(replacement_mask[b, i].item())
                
                if op == 0:
                    new_seq.append(tok)
                elif op == 1 and is_replaceable and replace_count < max_edits:
                    new_seq.append(pred_tok)
                    replace_count += 1
                elif op == 2 and insert_count < max_edits:
                    new_seq.append(pred_tok)
                    new_seq.append(tok)
                    insert_count += 1
                elif op == 3 and insert_count < max_edits:
                    new_seq.append(tok)
                    new_seq.append(pred_tok)
                    insert_count += 1
                elif op == 4 and delete_count < max_edits:
                    delete_count += 1
                    continue
                else:
                    new_seq.append(tok)
            
            result_tokens.append(new_seq)
        
        if not result_tokens or all(len(seq) == 0 for seq in result_tokens):
            return tokens
        
        max_len = max(len(seq) for seq in result_tokens)
        padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, seq in enumerate(result_tokens):
            if seq:
                padded[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        
        return padded


class SentenceOrderHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, 1)

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        scores = self.proj(sentence_embeddings)
        return scores.squeeze(-1)


class TreeNodeOperator(nn.Module):
    def __init__(
        self,
        d_model: int,
        manifold: str = "poincare",
        c: float = 1e-3,
        enable_dynamic_manifold: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.manifold = manifold
        self.c = c
        self.enable_dynamic_manifold = enable_dynamic_manifold
        self.aggregator = RiemannianAggregation(d_model, manifold, c, temperature=1.0)
        
        if enable_dynamic_manifold:
            self.manifold_selector = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 3),
            )
            self.aggregator_poincare = RiemannianAggregation(d_model, "poincare", c, temperature=1.0)
            self.aggregator_lorentz = RiemannianAggregation(d_model, "lorentz", c, temperature=1.0)
            self.aggregator_klein = RiemannianAggregation(d_model, "klein", c, temperature=1.0)
    
    def up_operator(
        self,
        children_embeddings: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enable_dynamic_manifold:
            return self.aggregator(children_embeddings, metric_ctx)
        
        B = children_embeddings.shape[0]
        mean_emb = children_embeddings.mean(dim=1)
        manifold_logits = self.manifold_selector(mean_emb)
        manifold_probs = torch.softmax(manifold_logits, dim=-1)
        
        result_poincare = self.aggregator_poincare(children_embeddings, metric_ctx)
        result_lorentz = self.aggregator_lorentz(children_embeddings, metric_ctx)
        result_klein = self.aggregator_klein(children_embeddings, metric_ctx)
        
        results = torch.stack([result_poincare, result_lorentz, result_klein], dim=1)
        weighted_result = (results * manifold_probs.unsqueeze(-1)).sum(dim=1)
        
        return weighted_result
    
    def down_operator(
        self,
        parent_embedding: torch.Tensor,
        num_children: int,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = parent_embedding.shape[0]
        parent_exp = parent_embedding.unsqueeze(1).expand(B, num_children, self.d_model)
        
        if self.enable_dynamic_manifold and hasattr(self, 'manifold_selector'):
            manifold_logits = self.manifold_selector(parent_embedding)
            manifold_probs = torch.softmax(manifold_logits, dim=-1)
            
            noise = torch.randn_like(parent_exp) * 0.01
            parent_exp = parent_exp + noise
        
        return parent_exp


class LevelInvariantTreeProcessor(nn.Module):
    def __init__(self, d_model: int, enable_dynamic_manifold: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.enable_dynamic_manifold = enable_dynamic_manifold
        self.node_operators: Dict[str, TreeNodeOperator] = nn.ModuleDict({
            "document": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
            "paragraph": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
            "sentence": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
            "token": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
        })
    
    def process_tree(
        self,
        tree: DocumentTree,
        node_embeddings: Dict[int, torch.Tensor],
        direction: str = "up",
    ) -> Dict[int, torch.Tensor]:
        result_embeddings: Dict[int, torch.Tensor] = {}
        
        if direction == "up":
            sorted_nodes = sorted(tree.nodes, key=lambda n: -self._depth(tree, n.id))
            for node in sorted_nodes:
                children_ids = tree.children(node.id)
                if not children_ids:
                    if node.id in node_embeddings:
                        result_embeddings[node.id] = node_embeddings[node.id]
                    else:
                        continue
                else:
                    available_children = [cid for cid in children_ids if cid in result_embeddings]
                    if not available_children:
                        if node.id in node_embeddings:
                            result_embeddings[node.id] = node_embeddings[node.id]
                        continue
                    children_embs = torch.stack([result_embeddings[cid] for cid in available_children])
                    if children_embs.dim() == 2:
                        children_embs = children_embs.unsqueeze(0)
                    
                    operator = self.node_operators[node.type] if node.type in self.node_operators else None
                    if operator:
                        result_embeddings[node.id] = operator.up_operator(children_embs).squeeze(0)
                    else:
                        result_embeddings[node.id] = children_embs.mean(dim=1).squeeze(0)
        
        elif direction == "down":
            sorted_nodes = sorted(tree.nodes, key=lambda n: self._depth(tree, n.id))
            for node in sorted_nodes:
                children_ids = tree.children(node.id)
                if node.id in result_embeddings:
                    parent_emb = result_embeddings[node.id]
                elif node.id in node_embeddings:
                    parent_emb = node_embeddings[node.id]
                else:
                    continue
                
                if children_ids:
                    operator = self.node_operators[node.type] if node.type in self.node_operators else None
                    if operator:
                        parent_emb_batched = parent_emb.unsqueeze(0) if parent_emb.dim() == 1 else parent_emb
                        children_embs = operator.down_operator(parent_emb_batched, len(children_ids))
                        for idx, cid in enumerate(children_ids):
                            result_embeddings[cid] = children_embs[0, idx]
        
        return result_embeddings
    
    def _depth(self, tree: DocumentTree, node_id: int) -> int:
        if not hasattr(self, '_depth_cache'):
            self._depth_cache = {}
        if node_id in self._depth_cache:
            return self._depth_cache[node_id]
        
        node = next((n for n in tree.nodes if n.id == node_id), None)
        if node is None or node.parent is None:
            self._depth_cache[node_id] = 0
            return 0
        depth = 1 + self._depth(tree, node.parent)
        self._depth_cache[node_id] = depth
        return depth


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
        warmup_ratio = 0.1
        if progress < warmup_ratio:
            min_factor = 0.1
            return base_lambda * (min_factor + (1.0 - min_factor) * (progress / warmup_ratio))
        else:
            return base_lambda
    
    return base_lambda


class RiemannianDiffusionStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: torch.Tensor, flow: torch.Tensor, diffusion_engine, alpha: float, dt: float) -> torch.Tensor:
        h = h.contiguous()
        flow = flow.contiguous()
        h_next = torch.empty_like(h)
        batch_size, dim = h.shape
        if h.is_cuda and getattr(rs, "_has_cuda", False) and diffusion_engine is not None:
            diffusion_engine.step_cuda(
                h.data_ptr(),
                flow.data_ptr(),
                h_next.data_ptr(),
                batch_size,
                dim,
            )
        else:
            h_np = h.detach().cpu().numpy().astype("float32")
            flow_np = flow.detach().cpu().numpy().astype("float32")
            h_next_np = diffusion_engine.step_cpu(h_np, flow_np)
            h_next = torch.from_numpy(h_next_np).to(h.device)
        ctx.alpha = float(alpha)
        ctx.dt = float(dt)
        return h_next

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        alpha = ctx.alpha
        dt = ctx.dt
        a = 1.0 - (1.0 - alpha) * dt
        b = (1.0 - alpha) * dt
        grad_h = grad_output * a
        grad_flow = grad_output * b
        return grad_h, grad_flow, None, None, None


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
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        C = logits.size(-1)
        P_topic = F.softmax(logits, dim=-1)
        P_topic = torch.where(torch.isfinite(P_topic), P_topic, torch.full_like(P_topic, 1.0 / max(C, 1)))
        P_topic = P_topic + 1e-8
        P_topic = P_topic / P_topic.sum(dim=-1, keepdim=True)

        scores, _ = logits.max(dim=-1)
        scores = torch.clamp(scores, min=-10.0, max=10.0)

        metric_keys: List[str] = []
        with torch.no_grad():
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
        spd_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.cache_size = cache_size
        self.score_quantize = score_quantize
        self.spd_eps = spd_eps
        from collections import OrderedDict
        self._cache: OrderedDict[Tuple[str, float, str], torch.Tensor] = OrderedDict()

        try:
            import reality_stone.metrikey as metrikey  # type: ignore
            self._metrikey = metrikey
            self._has_metrikey = True
        except Exception:
            self._metrikey = None
            self._has_metrikey = False

        self._metrikey = None
        self._has_metrikey = False
        
        self.metric_adjustment = nn.Parameter(torch.zeros(d_head, d_head))

    def _clamp_eigen(self, G: torch.Tensor) -> torch.Tensor:
        G_sym = (G + G.transpose(-2, -1)) / 2.0
        G_sym = G_sym + torch.eye(G.shape[-1], device=G.device, dtype=G.dtype) * self.spd_eps
        
        eigvals, eigvecs = torch.linalg.eigh(G_sym)
        eigvals = torch.clamp(eigvals, self.lambda_min, self.lambda_max)
        result = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
        
        result = (result + result.transpose(-2, -1)) / 2.0
        return result

    def _make_metric(self, key: str, score_q: float, device: torch.device) -> torch.Tensor:
        cache_key = (key, score_q, str(device))
        
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
                G = G.to(device)
            except Exception:
                scale = 1.0 + score_q * 0.1
                G = torch.eye(self.d_head, device=device) * scale
        else:
            scale = 1.0 + score_q * 0.1
            G = torch.eye(self.d_head, device=device) * scale

        G = self._clamp_eigen(G)
        G_reg = G + torch.eye(self.d_head, device=device) * self.spd_eps
        L = torch.linalg.cholesky(G_reg)

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        
        self._cache[cache_key] = L
        return L

    def forward(self, metric_keys: List[str], scores: torch.Tensor) -> torch.Tensor:
        B, T = scores.shape
        device = scores.device
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        if self.score_quantize is not None and self.score_quantize > 0:
            q = torch.as_tensor(self.score_quantize, dtype=scores.dtype, device=device)
            scores = torch.round(scores / q) * q
        
        eye_base = torch.eye(self.d_head, device=device, dtype=scores.dtype)
        
        scores_norm = torch.tanh(scores / 10.0)
        scale = 1.0 + 0.2 * scores_norm
        
        adjustment_sym = (self.metric_adjustment + self.metric_adjustment.t()) / 2.0
        adjustment_scale = 0.1 * torch.tanh(adjustment_sym)
        
        L_list = []
        for b in range(B):
            for t in range(T):
                s = scale[b, t]
                L_bt = eye_base * s + adjustment_scale
                L_bt = L_bt + eye_base * self.spd_eps
                L_list.append(L_bt)
        
        L_stacked = torch.stack(L_list, dim=0)
        L_adjusted = L_stacked.view(B, T, self.d_head, self.d_head)
        
        return L_adjusted


def _spd_log_euclidean_mean(
    spd_matrices: torch.Tensor, 
    weights: torch.Tensor,
    eps: float = 1e-5,
    eigval_min: float = 1e-5,
    eigval_max: float = 1e5,
    log_clip: float = 10.0,
) -> torch.Tensor:
    B, N, d, _ = spd_matrices.shape
    device = spd_matrices.device
    dtype = spd_matrices.dtype
    
    eps_eye = torch.eye(d, device=device, dtype=dtype) * eps
    spd_matrices = spd_matrices + eps_eye.view(1, 1, d, d)
    
    spd_flat = spd_matrices.reshape(B * N, d, d)
    eigvals, eigvecs = torch.linalg.eigh(spd_flat)
    eigvals = eigvals.clamp(min=eigval_min, max=eigval_max)
    log_eigvals = torch.log(eigvals)
    
    log_matrices_flat = torch.bmm(
        torch.bmm(eigvecs, torch.diag_embed(log_eigvals)),
        eigvecs.transpose(-2, -1)
    )
    log_matrices = log_matrices_flat.reshape(B, N, d, d)
    
    w = weights.view(B, N, 1, 1)
    log_mean = (w * log_matrices).sum(dim=1)
    
    eigvals_mean, eigvecs_mean = torch.linalg.eigh(log_mean)
    eigvals_mean = eigvals_mean.clamp(min=-log_clip, max=log_clip)
    exp_eigvals = torch.exp(eigvals_mean)
    exp_eigvals = exp_eigvals.clamp(min=eigval_min, max=eigval_max)
    
    result = torch.bmm(
        torch.bmm(eigvecs_mean, torch.diag_embed(exp_eigvals)),
        eigvecs_mean.transpose(-2, -1)
    )
    
    result = (result + result.transpose(-2, -1)) / 2.0
    result = result + eps_eye
    
    return result


class SPDMetricMixer(nn.Module):
    def __init__(
        self,
        d_head: int,
        gamma_up: float = 0.3,
        gamma_self: float = 0.5,
        gamma_down: float = 0.2,
        use_fast_mixing: bool = True,
        spd_eps: float = 1e-5,
        spd_eigval_min: float = 1e-5,
        spd_eigval_max: float = 1e5,
        spd_log_eigval_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.use_fast_mixing = use_fast_mixing
        self.spd_eps = spd_eps
        self.spd_eigval_min = spd_eigval_min
        self.spd_eigval_max = spd_eigval_max
        self.spd_log_eigval_clip = spd_log_eigval_clip
        self.gamma_up = nn.Parameter(torch.tensor(gamma_up))
        self.gamma_self = nn.Parameter(torch.tensor(gamma_self))
        self.gamma_down = nn.Parameter(torch.tensor(gamma_down))

    def mix_hierarchy(
        self,
        parent_metric: torch.Tensor,
        self_metric: torch.Tensor,
        children_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, d, _ = self_metric.shape
        
        gamma_up = torch.abs(self.gamma_up) + 1e-6
        gamma_self = torch.abs(self.gamma_self) + 1e-6
        gamma_down = torch.abs(self.gamma_down) + 1e-6
        
        mats = [parent_metric, self_metric]
        ws_raw = [gamma_up, gamma_self]

        if children_metrics is not None and children_metrics.size(1) > 0:
            child_mean = children_metrics.mean(dim=1)
            mats.append(child_mean)
            ws_raw.append(gamma_down)

        ws_tensor = torch.stack(ws_raw)
        ws_norm = F.softmax(ws_tensor, dim=0)
        
        if self.use_fast_mixing:
            mats_tensor = torch.stack(mats, dim=1)
            result = (ws_norm.view(1, -1, 1, 1) * mats_tensor).sum(dim=1)
            return result
        else:
            mats_tensor = torch.stack(mats, dim=1)
            w_expanded = ws_norm.view(1, -1).expand(B, -1)
            return _spd_log_euclidean_mean(
                mats_tensor, 
                w_expanded,
                eps=self.spd_eps,
                eigval_min=self.spd_eigval_min,
                eigval_max=self.spd_eigval_max,
                log_clip=self.spd_log_eigval_clip,
            )


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
        self.lm_head.weight = self.token_embed.weight

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
        
        self.lambda_p = nn.Parameter(torch.tensor(0.5))
        self.lambda_l = nn.Parameter(torch.tensor(0.5))
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
        # 논문 Section 3.3 & 9.5: 학습 가능한 manifold 가중치
        lambda_p = torch.sigmoid(self.lambda_p)
        lambda_l = torch.sigmoid(self.lambda_l)
        lambda_sum = lambda_p + lambda_l + 1e-8
        lambda_p_norm = lambda_p / lambda_sum
        lambda_l_norm = lambda_l / lambda_sum
        d2_total = lambda_p_norm * (d_p ** 2) + lambda_l_norm * (d_l ** 2)
        d2 = d2_total.reshape(B, H, S, K)  # [B,H,S,K]

        # scores = -d^2 / τ
        tau = max(self.attn.tau, 1e-6)
        scores = -d2 / tau  # [B,H,S,K]
        # Low-rank auxiliary term to ensure gradient flows to SPDMetric.U
        qu = self.attn.metric.lowrank_proj(q)
        ku = self.attn.metric.lowrank_proj(k)
        if qu is not None and ku is not None:
            s_lr_full = torch.einsum("bhtr,bhsr->bhts", qu, ku)  # [B,H,S,S]
            idx_bhs = idx_causal.unsqueeze(1).expand(B, H, S, K)  # [B,H,S,K]
            s_lr = s_lr_full.gather(dim=3, index=idx_bhs)  # [B,H,S,K]
            scores = scores + 1e-3 * s_lr

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

        # SFE Variable Suppression Field
        self.suppression_field = HyperbolicSuppressionField(
            base=getattr(config, "suppression_base", 0.37),
            linear=getattr(config, "suppression_linear", 0.0),
            hyp=getattr(config, "suppression_hyp", 0.1),
            scale=getattr(config, "suppression_scale", 1.0)
        ) if getattr(config, "enable_variable_suppression", False) else None

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
            lambda_min=config.metric_lambda_min,
            lambda_max=config.metric_lambda_max,
            spd_eps=config.spd_eps,
        )
        
        # L2.5: SPD Metric Mixer (barycenter-based mixing)
        use_fast_mixing = getattr(config, "use_fast_spd_mixing", True)
        self.metric_mixer = SPDMetricMixer(
            d_head=config.d_head,
            gamma_up=config.gamma_up,
            gamma_self=config.gamma_self,
            gamma_down=config.gamma_down,
            use_fast_mixing=use_fast_mixing,
            spd_eps=config.spd_eps,
            spd_eigval_min=config.spd_eigval_min,
            spd_eigval_max=config.spd_eigval_max,
            spd_log_eigval_clip=config.spd_log_eigval_clip,
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
        engine = None
        if getattr(config, "use_diffusion_hidden", False) and getattr(config, "diffusion_steps", 0) > 0:
            engine_cls = getattr(rs, "PyRiemannianDiffusion", None)
            if engine_cls is not None:
                try:
                    engine = engine_cls(config.d_model, config.diffusion_alpha, config.diffusion_dt)
                except Exception:
                    engine = None
        self.diffusion_engine = engine
        
        # Decoder와 Embedding 공유 + Weight Tying
        self.decoder.token_embed = self.token_embed
        if hasattr(self.token_embed, "weight"):
            self.decoder.lm_head.weight = self.token_embed.weight
        self.semantic_loss = SemanticPreservationLoss(
            manifold=config.manifold_sentence,
            c=config.c_poincare,
        )
        self.edit_head = EditOperationHead(config.d_model, num_ops=5, edit_budget=config.edit_budget)
        self.sentence_order_head = SentenceOrderHead(config.d_model)
        
        enable_dynamic_manifold = getattr(config, "enable_dynamic_manifold", False)
        self.tree_processor = LevelInvariantTreeProcessor(config.d_model, enable_dynamic_manifold)
        
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
        # SFE: Dynamic Temperature 적용
        # 억압장이 강할수록(epsilon↑) -> 유효 질량 감소(m_eff↓) -> 온도 증가(T_eff↑) -> 분포가 평평해짐 (Smoothing)
        # 반대로 epsilon이 작으면 -> T_eff 감소 -> 분포가 뾰족해짐 (Sharpening/Focusing)
        
        metric_ctx_reshaped = metric_flat # [BT, d, d] or None
        
        # 토큰들의 Norm을 억압장의 입력으로 사용 (원점에서의 거리 = 정보량/깊이)
        # tokens_flat: [BT, L, d_model]
        token_norms = tokens_flat.norm(dim=-1).mean(dim=-1, keepdim=True) # [BT, 1]
        
        current_temp = self.config.temperature_agg
        temperature_override = None
        if self.suppression_field is not None:
            dynamic_temp = self.suppression_field.compute_effective_temperature(
                t0=current_temp,
                x=token_norms
            )  # [BT, 1]
            temperature_override = dynamic_temp.mean()  # Tensor (keeps grad path)

        sentence_embeddings_flat = self.sentence_aggregator(
            tokens_flat,  # [B*T, L, d_model]
            metric_ctx=metric_flat,
            temperature_override=temperature_override,
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
        """
        
        # SFE: Dynamic Temperature 적용 (Paragraph Level)
        sent_norms = sentence_embeddings.norm(dim=-1).mean(dim=-1, keepdim=True) # [B, 1]
        
        current_temp = self.config.temperature_agg
        temperature_override = None
        if self.suppression_field is not None:
            dynamic_temp = self.suppression_field.compute_effective_temperature(
                t0=current_temp,
                x=sent_norms
            )
            temperature_override = dynamic_temp.mean()

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
    
    def _encode_with_tree_processor(
        self,
        tokens: torch.Tensor,
        trees: List[DocumentTree],
        direction: str = "up",
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, L = tokens.shape
        device = tokens.device
        
        sentence_embeddings_list = []
        
        for b in range(B):
            tree = trees[b] if b < len(trees) else None
            if tree is None:
                sent_emb = self.encode_tokens_to_sentences(tokens[b:b+1])
                sentence_embeddings_list.append(sent_emb[0])
                continue
            
            node_embeddings: Dict[int, torch.Tensor] = {}
            
            sentence_nodes = [n for n in tree.nodes if n.type == "sentence"]
            if len(sentence_nodes) > T:
                sentence_nodes = sentence_nodes[:T]
            
            for sent_idx, sent_node in enumerate(sentence_nodes):
                if sent_idx >= T:
                    break
                tok_ids = tokens[b, sent_idx].clamp(0, self.config.vocab_size - 1)
                if tok_ids.dim() == 1:
                    tok_ids = tok_ids.unsqueeze(0)
                token_embs = self.token_embed(tok_ids)
                
                if metric_ctx is not None:
                    sent_metric = metric_ctx[b, sent_idx].unsqueeze(0)
                else:
                    sent_metric = None
                sent_emb = self.sentence_aggregator(token_embs, metric_ctx=sent_metric)
                
                if sent_emb.dim() == 1:
                    sent_emb = sent_emb.unsqueeze(0)
                
                node_embeddings[sent_node.id] = sent_emb.squeeze(0)
            
            if direction == "up":
                processed_embs = self.tree_processor.process_tree(
                    tree,
                    node_embeddings,
                    direction="up",
                )
                
                sent_embs_batch = []
                for sent_node in sentence_nodes[:T]:
                    if sent_node.id in processed_embs:
                        sent_embs_batch.append(processed_embs[sent_node.id])
                    elif sent_node.id in node_embeddings:
                        sent_embs_batch.append(node_embeddings[sent_node.id])
                    else:
                        sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                while len(sent_embs_batch) < T:
                    sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                sentence_embeddings_list.append(torch.stack(sent_embs_batch[:T]))
            else:
                sent_embs_batch = []
                for sent_node in sentence_nodes[:T]:
                    if sent_node.id in node_embeddings:
                        sent_embs_batch.append(node_embeddings[sent_node.id])
                    else:
                        sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                while len(sent_embs_batch) < T:
                    sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                sentence_embeddings_list.append(torch.stack(sent_embs_batch[:T]))
        
        return torch.stack(sentence_embeddings_list, dim=0)


    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        use_tree_processing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        한 배치에 대해 전체 L1–L3 파이프라인을 통과시키고,
        (원하면) 토픽 + LM loss 를 함께 반환한다.

        Args:
            batch:
                - "tokens": [B, T, L]
                - "topo_idx": [B, T, K]
                - "tree": Optional[List[DocumentTree]] - 트리 구조 (배치별)
            compute_loss: 손실 계산 여부
            use_tree_processing: 트리 프로세서 사용 여부

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
        trees = batch.get("tree", None)   # Optional[List[DocumentTree]]

        device = next(self.parameters()).device
        tokens = tokens.to(device)
        topo_idx = topo_idx.to(device)

        B, T, L = tokens.shape

        # ========== 상향식 인코딩 (Bottom-up) ==========
        # Step 1: 토큰 → 문장 (Riemannian message passing)
        if use_tree_processing and trees is not None and len(trees) > 0:
            sentence_embeddings_raw = self._encode_with_tree_processor(tokens, trees, direction="up")  # [B, T, d_model]
        else:
            sentence_embeddings_raw = self.encode_tokens_to_sentences(tokens)  # [B, T, d_model]
        
        # Step 2: 문장 → 주제/메트릭 키 (SentenceTopicHead)
        P_topic, scores, metric_keys = self.topic_head(sentence_embeddings_raw, topo_idx)
        C = P_topic.size(-1)

        # 문단 내 consistency: KL(P_topic || paragraph_mean)
        paragraph_mean = P_topic.mean(dim=1, keepdim=True).detach()  # [B,1,C]
        paragraph_mean = paragraph_mean + 1e-8
        paragraph_mean = paragraph_mean / paragraph_mean.sum(dim=-1, keepdim=True)
        paragraph_mean = paragraph_mean.expand(-1, T, -1)
        
        log_p = torch.log(P_topic + 1e-8)
        loss_consistency = nn.KLDivLoss(reduction="batchmean")(log_p, paragraph_mean)
        loss_consistency = torch.clamp(loss_consistency, min=0.0, max=10.0)
        
        # 배치 전체 diversity: KL(batch_mean || uniform)
        batch_mean = P_topic.mean(dim=(0, 1))  # [C]
        batch_mean = batch_mean + 1e-8
        batch_mean = batch_mean / batch_mean.sum()
        uniform = torch.full_like(batch_mean, 1.0 / C)
        
        log_batch = torch.log(batch_mean + 1e-8)
        loss_diversity = nn.KLDivLoss(reduction="batchmean")(log_batch, uniform)
        loss_diversity = torch.clamp(loss_diversity, min=0.0, max=10.0)

        # Step 3: MetriKey → SPD 메트릭 (MetricContextRouter)
        metric_ctx_sentence = self.metric_router(metric_keys, scores)  # [B, T, d_h, d_h]
        
        # Step 4: 문장 → 문단 (Riemannian message passing with metric)
        # 문단 메트릭: 문장 메트릭들의 평균
        metric_ctx_paragraph = metric_ctx_sentence.mean(dim=1)  # [B, d_h, d_h]
        
        # 메트릭 적용하여 재인코딩
        if use_tree_processing and trees is not None and len(trees) > 0:
            sentence_embeddings = self._encode_with_tree_processor(
                tokens, trees, direction="up", metric_ctx=metric_ctx_sentence
            )  # [B, T, d_model]
        else:
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
        length_logits = torch.where(torch.isfinite(length_logits), length_logits, torch.zeros_like(length_logits))
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
        if getattr(self.config, "use_diffusion_hidden", False) and getattr(self.config, "diffusion_steps", 0) > 0 and getattr(self, "diffusion_engine", None) is not None:
            B_hidden, S_hidden, D_hidden = hidden.shape
            h_flat = hidden.reshape(B_hidden * S_hidden, D_hidden)
            for _ in range(self.config.diffusion_steps):
                flow = torch.tanh(h_flat)
                h_flat = RiemannianDiffusionStep.apply(
                    h_flat,
                    flow,
                    self.diffusion_engine,
                    self.config.diffusion_alpha,
                    self.config.diffusion_dt,
                )
            hidden = h_flat.view(B_hidden, S_hidden, D_hidden)
            logits = self.decoder.lm_head(hidden)
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        logits = torch.clamp(logits, min=-self.config.logit_clip_value, max=self.config.logit_clip_value)
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
        info_str["paragraph_embedding"] = paragraph_embedding

        has_lm_target = True

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

            # ===== 논문 설계: Next-token prediction (Autoregressive) =====
            # Decoder는 autoregressive하게 다음 토큰을 예측
            # input: tokens[:, :-1], target: tokens[:, 1:]
            
            S = tokens_flat.size(1)
            S_max = min(S, logits.size(1))
            
            if S_max > 1:
                logits_pred = logits[:, :S_max-1, :]
                targets = tokens_flat[:, 1:S_max].clamp(0, self.config.vocab_size - 1)
            else:
                logits_pred = logits[:, :0, :]
                targets = tokens_flat[:, :0].clamp(0, self.config.vocab_size - 1)
            
            V = logits.size(-1)
            targets_flat = targets.reshape(-1)
            valid_mask = targets_flat.ne(0)
            if valid_mask.any():
                logits_flat = logits_pred.reshape(-1, V)
                logits_flat_valid = logits_flat[valid_mask]
                targets_flat_valid = targets_flat[valid_mask]
                lm_loss = F.cross_entropy(
                    logits_flat_valid,
                    targets_flat_valid,
                )
                has_lm_target = True
            else:
                if logits_pred.numel() == 0 or targets.numel() == 0:
                    raise RuntimeError("No LM targets available; check tokenization and sequence lengths.")
                logits_flat = logits_pred.reshape(-1, V)
                targets_all = targets.reshape(-1)
                lm_loss = F.cross_entropy(
                    logits_flat,
                    targets_all,
                )
                has_lm_target = True

            loss_clip = self.config.loss_clip_max
            
            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                raise RuntimeError("lm_loss is NaN or Inf; check dataset/tokenization and model configuration.")
            lm_loss = torch.clamp(lm_loss, min=0.0, max=loss_clip)
            
            if torch.isnan(loss_consistency) or torch.isinf(loss_consistency):
                raise RuntimeError("loss_consistency is NaN or Inf; check topic distributions.")
            loss_consistency = torch.clamp(loss_consistency, min=0.0, max=loss_clip * 0.1)
            
            if torch.isnan(loss_diversity) or torch.isinf(loss_diversity):
                raise RuntimeError("loss_diversity is NaN or Inf; check topic distributions.")
            loss_diversity = torch.clamp(loss_diversity, min=0.0, max=loss_clip * 0.1)
            
            if torch.isnan(length_loss) or torch.isinf(length_loss):
                raise RuntimeError("length_loss is NaN or Inf; check sentence_nonempty / length_logits.")
            length_loss = torch.clamp(length_loss, min=0.0, max=loss_clip * 0.1)
            
            if topic_loss is not None:
                if torch.isnan(topic_loss) or torch.isinf(topic_loss):
                    raise RuntimeError("topic_loss is NaN or Inf; check topic_labels and P_topic.")
                topic_loss = torch.clamp(topic_loss, min=0.0, max=loss_clip * 0.1)

            # Metric regularization: ||G - I||_F^2, G = L L^T
            d_h = self.config.d_head
            eye = torch.eye(d_h, device=device, dtype=metric_ctx.dtype)
            G_sentence = metric_ctx_sentence.reshape(B * T, d_h, d_h)
            G_sentence = G_sentence @ G_sentence.transpose(-2, -1)
            diff_G = G_sentence - eye
            loss_metric = (diff_G.pow(2).sum(dim=(-2, -1))).mean()

            loss_curvature = torch.tensor(0.0, device=device, dtype=logits.dtype, requires_grad=False)

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
                semantic_loss = torch.clamp(semantic_loss, min=0.0, max=self.config.loss_clip_max * 0.1)
                loss = loss + self.config.lambda_semantic * semantic_loss
                info_str["loss_semantic"] = semantic_loss

            # Tiny regularizer to keep sentence_order_head in graph
            order_reg = (sentence_order_scores ** 2).mean() * 1e-6
            loss = loss + order_reg
            info_str["loss_sentence_order_reg"] = order_reg

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
            else:
                loss_edit_reg = (edit_logits ** 2).mean() * 1e-6
                loss = loss + loss_edit_reg
                info_str["loss_edit_reg"] = loss_edit_reg

            gamma_reg = (
                (self.metric_mixer.gamma_up ** 2)
                + (self.metric_mixer.gamma_self ** 2)
                + (self.metric_mixer.gamma_down ** 2)
            ) * 1e-6
            loss = loss + gamma_reg
            info_str["loss_gamma_reg"] = gamma_reg
            
            loss = torch.clamp(loss, min=0.0, max=self.config.loss_clip_max * 2.0)

            info_str["loss"] = loss
            info_str["loss_lm"] = lm_loss
            info_str["has_lm_target"] = has_lm_target
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
    teacher_model=None,
    teacher_tokenizer=None,
    kd_proj: Optional[nn.Module] = None,
    kd_weight: float = 0.0,
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

    use_kd = teacher_model is not None and teacher_tokenizer is not None and kd_proj is not None and kd_weight > 0.0

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
    metric_params.extend(model.metric_router.parameters())
    metric_params.extend(model.metric_mixer.parameters())
    metric_params.extend(model.sentence_aggregator.parameters())
    metric_params.extend(model.paragraph_aggregator.parameters())
    if model.suppression_field is not None:
        metric_params.extend(model.suppression_field.parameters())
    
    # Backbone parameters (low LR)
    backbone_params = []
    for name, param in model.topic_head.named_parameters():
        if param.requires_grad and not ("metric" in name.lower() or "spd" in name.lower()):
            backbone_params.append(param)
    backbone_params.extend(model.decoder.parameters())
    if use_kd:
        backbone_params.extend(list(kd_proj.parameters()))
    
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

    base_lambda_consistency = config.lambda_consistency
    base_lambda_diversity = config.lambda_diversity
    
    for epoch in range(epochs):
        # 동적 lambda 계산
        lambda_consistency_current = compute_dynamic_lambda(
            base_lambda_consistency,
            config.lambda_consistency_schedule,
            epoch,
            epochs,
        )
        lambda_diversity_current = compute_dynamic_lambda(
            base_lambda_diversity,
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
                if use_kd:
                    paragraphs = batch.get("paragraphs", None)
                    paragraph_emb = info.get("paragraph_embedding", None)
                    if paragraphs is not None and isinstance(paragraphs, list) and isinstance(paragraph_emb, torch.Tensor):
                        enc = teacher_tokenizer(paragraphs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                        for k in enc:
                            enc[k] = enc[k].to(device_t)
                        with torch.no_grad():
                            teacher_out = teacher_model(**enc)
                            if hasattr(teacher_out, "last_hidden_state"):
                                teacher_hidden = teacher_out.last_hidden_state[:, 0, :]
                            else:
                                teacher_hidden = teacher_out[0][:, 0, :]
                        teacher_proj = kd_proj(teacher_hidden)
                        teacher_proj = teacher_proj.to(paragraph_emb.dtype)
                        loss_kd = F.mse_loss(paragraph_emb, teacher_proj)
                        loss = loss + kd_weight * loss_kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(metric_params, config.grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(backbone_params, config.grad_clip_norm)
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


def _apply_top_down_decoding(
    model: HierarchicalSentenceTopicLLM,
    tree: DocumentTree,
    info: Dict[str, object],
    tokens: torch.Tensor,
    replacement_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    B, T, L = tokens.shape
    
    paragraph_nodes = [n for n in tree.nodes if n.type == "document"]
    sentence_nodes = [n for n in tree.nodes if n.type == "sentence"]
    
    if not paragraph_nodes:
        S = T * L
        input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)
        return input_ids_flat
    
    para_node = paragraph_nodes[0]
    
    hidden = info.get("hidden")
    if hidden is None or not isinstance(hidden, torch.Tensor):
        S = T * L
        input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)
        return input_ids_flat
    
    paragraph_embedding = hidden.mean(dim=1)
    
    node_embeddings: Dict[int, torch.Tensor] = {}
    node_embeddings[para_node.id] = paragraph_embedding[0]
    
    for sent_idx, sent_node in enumerate(sentence_nodes[:T]):
        if sent_idx >= T:
            break
        pos = min(sent_idx * L, hidden.size(1) - 1)
        node_embeddings[sent_node.id] = hidden[0, pos]
    
    processed_embs = model.tree_processor.process_tree(
        tree,
        node_embeddings,
        direction="down",
    )
    
    result_tokens = []
    for sent_idx, sent_node in enumerate(sentence_nodes[:T]):
        if sent_node.id in processed_embs:
            sent_emb = processed_embs[sent_node.id]
        elif sent_node.id in node_embeddings:
            sent_emb = node_embeddings[sent_node.id]
        else:
            sent_emb = torch.zeros(model.config.d_model, device=device)
        
        sent_emb_expanded = sent_emb.unsqueeze(0).unsqueeze(0)
        logits_sent = model.decoder.lm_head(sent_emb_expanded)
        pred_tokens_sent = torch.argmax(logits_sent, dim=-1)
        
        pred_tokens_sent = pred_tokens_sent.expand(1, L)
        
        original_tokens_sent = tokens[0, sent_idx].clamp(0, model.config.vocab_size - 1)
        replacement_mask_sent = replacement_mask[sent_idx].to(device)
        
        edited_tokens_sent = torch.where(
            replacement_mask_sent.bool(),
            pred_tokens_sent[0],
            original_tokens_sent,
        )
        result_tokens.append(edited_tokens_sent)
    
    result_flat = torch.cat(result_tokens, dim=0).unsqueeze(0)
    return result_flat


def infer_hierarchical_llm_on_text(
    model: HierarchicalSentenceTopicLLM,
    text: str,
    max_length: int = 128,
    k_neighbors: int = 3,
    max_new_tokens: int = 20,
    use_top_down: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.9,
    use_sampling: bool = True,
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
    from reality_stone.utils.pre_segmenter import PreSegmenter

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

    tokens = seg_output["tokens"].unsqueeze(0).to(device)
    topo_idx = seg_output["topo_idx"].unsqueeze(0).to(device)
    tree = seg_output.get("tree")

    batch: Dict[str, torch.Tensor] = {
        "tokens": tokens,
        "topo_idx": topo_idx,
    }
    if tree is not None:
        batch["tree"] = [tree]

    model.eval()
    with torch.no_grad():
        logits, info = model(batch, compute_loss=False, use_tree_processing=use_top_down)

    original_sentences: List[str] = seg_output["sentences"]
    tokenizer = segmenter.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
    
    # T, L 변수 미리 정의 (inference에서 사용)
    B, T, L = tokens.shape

    if use_top_down and tree is not None:
        tokens_seq = _apply_top_down_decoding(
            model=model,
            tree=tree,
            info=info,
            tokens=tokens,
            replacement_mask=seg_output["replacement_mask"],
            device=device,
        )
        S_actual = tokens_seq.size(1)
    else:
        replacement_mask = seg_output["replacement_mask"].unsqueeze(0).to(device)
        B, T, L = tokens.shape
        S = T * L
        input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)
        mask_flat = replacement_mask.view(1, S)
        
        S_actual = logits.size(1)
        if S_actual < S:
            input_ids_flat = input_ids_flat[:, :S_actual]
            mask_flat = mask_flat[:, :S_actual]
        
        if use_sampling and temperature > 0:
            V = logits.size(-1)
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask_p = cumsum_probs > top_p
                mask_p[..., 0] = False
                sorted_probs = sorted_probs.clone()
                sorted_probs[mask_p] = 0.0
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                sampled_sorted_idx = torch.multinomial(sorted_probs.view(-1, V), num_samples=1).view(probs.shape[:-1], 1)
                pred_ids_flat = sorted_indices.gather(-1, sampled_sorted_idx).squeeze(-1)
            else:
                pred_ids_flat = torch.multinomial(probs.view(-1, V), num_samples=1).view(probs.shape[:-1])
        else:
            pred_ids_flat = torch.argmax(logits, dim=-1)
        
        edited_flat = torch.where(mask_flat.bool(), pred_ids_flat, input_ids_flat)
        tokens_seq = edited_flat

        if getattr(model.config, "enable_structural_edit", False):
            edit_logits = info.get("edit_logits")
            if isinstance(edit_logits, torch.Tensor):
                tokens_seq = model.edit_head.apply_edits(
                    tokens=tokens_seq,
                    edit_logits=edit_logits[:, :S_actual, :],
                    pred_tokens=pred_ids_flat[:, :S_actual],
                    enable_structural=True,
                    replacement_mask=mask_flat[:, :S_actual] if 'mask_flat' in locals() else None,
                )
                S_actual = tokens_seq.size(1)

    final_ids_flat = tokens_seq[0].tolist()
    if tokenizer is not None:
        try:
            if getattr(model.config, "enable_structural_edit", False):
                token_ids_no_pad = [tid for tid in final_ids_flat if tid != pad_id and tid > 0]
                if token_ids_no_pad:
                    generated_text = tokenizer.decode(token_ids_no_pad, skip_special_tokens=True)
                else:
                    generated_text = ""
            else:
                generated_sentences: List[str] = []
                for sent_idx in range(T):
                    start_idx = sent_idx * L
                    end_idx = min(start_idx + L, len(final_ids_flat))
                    sent_token_ids = final_ids_flat[start_idx:end_idx]
                    sent_token_ids_no_pad = [tid for tid in sent_token_ids if tid != pad_id and tid > 0]
                    if sent_token_ids_no_pad:
                        sent_text = tokenizer.decode(sent_token_ids_no_pad, skip_special_tokens=True)
                        if sent_text.strip():
                            generated_sentences.append(sent_text)
                if generated_sentences:
                    generated_text = " ".join(generated_sentences)
                else:
                    generated_text = ""
        except Exception as e:
            import traceback
            print(f"[WARNING] Tokenizer decode failed: {e}")
            print(traceback.format_exc())
            generated_text = ""
    else:
        generated_text = ""
    
    if not generated_text or generated_text == text:
        generated_text = text

    # 문단 레벨 컨트롤러가 예측한 문장 수에 맞게, 문장을 잘라서 상위 레벨에서 발화 길이를 제어
    length_logits_tensor = info.get("length_logits")
    if isinstance(length_logits_tensor, torch.Tensor) and len(generated_text) > 0:
        length_probs = torch.softmax(length_logits_tensor, dim=-1)
        pred_sentences = int(length_probs[0].argmax().item()) + 1
        pred_sentences = min(pred_sentences, model.config.max_answer_sentences)
        
        seg_generated = segmenter(generated_text)
        gen_sents = seg_generated.get("sentences", [])
        if gen_sents and len(gen_sents) > pred_sentences:
            order_scores = info.get("sentence_order_scores")
            if isinstance(order_scores, torch.Tensor) and not getattr(model.config, "enable_structural_edit", False):
                scores_np = order_scores[0, : len(gen_sents)].detach().cpu()
                indices = list(range(len(gen_sents)))
                indices.sort(key=lambda i: float(scores_np[i].item()), reverse=True)
                indices = indices[:pred_sentences]
                indices.sort()
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
    # 학습된 lambda 사용 (첫 번째 decoder block에서)
    if hasattr(model.decoder.blocks[0], 'lambda_p'):
        lambda_p = torch.sigmoid(model.decoder.blocks[0].lambda_p).item()
        lambda_l = torch.sigmoid(model.decoder.blocks[0].lambda_l).item()
        lambda_sum = lambda_p + lambda_l + 1e-8
        lambda_p = lambda_p / lambda_sum
        lambda_l = lambda_l / lambda_sum
    else:
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
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
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
        prompt_text = (
            f"질문: {question}\n\n"
            f"답변: (한국어로, 가능한 한 자세하고 쉽게 설명해 주세요.)"
        )
    else:
        context = "\n\n".join(support)
        prompt_text = (
            f"{context}\n\n"
            f"질문: {question}\n\n"
            f"답변: (위 컨텍스트를 참고하여, 한국어로 자세하고 쉽게 설명해 주세요.)"
        )

    # 2) 계층적 LLM을 이용한 디코딩 (문단 단위 편집 + autoregressive 확장)
    infer_out = infer_hierarchical_llm_on_text(
        model=model,
        text=prompt_text,
        max_length=256,
        k_neighbors=3,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_sampling=True,
    )
    generated_text = infer_out.get("generated_text", "")

    answer_text = generated_text
    marker = "답변:"
    if marker in generated_text:
        parts = generated_text.split(marker)
        if len(parts) > 1:
            tail = parts[-1].strip()
            if tail and len(tail) > 3:
                answer_text = tail
    
    if not answer_text or answer_text == prompt_text:
        answer_text = "죄송합니다. 답변을 생성할 수 없습니다."


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


