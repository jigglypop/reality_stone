import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

import reality_stone as rs
from reality_stone.layers.poincare import project_to_ball
from reality_stone.models.rsulf import RSULF, RSULFStack


@dataclass
class LLMAdapterConfig:
    pretrained_model_name: str = "gpt2"
    use_causal_lm: bool = True
    
    hidden_dim: int = 768
    num_hyperbolic_layers: int = 4
    hyperbolic_insertion_positions: Optional[List[int]] = None
    
    use_bellman_coordinates: bool = True
    use_riemannian_metric: bool = True
    use_triple_hyperbolic: bool = True
    use_lagrangian: bool = True
    use_temporal_creativity: bool = False
    
    c_poincare: float = 1e-3
    c_lorentz: float = -1.0
    c_klein: float = 1e-3
    
    gamma_bellman: float = 0.99
    key_size: int = 32
    
    freeze_pretrained: bool = True
    convert_linear_to_hyperbolic: bool = False
    
    metric_regularization_weight: float = 0.01
    lagrangian_weight: float = 0.1
    creativity_reward_weight: float = 0.01


class RiemannianMetricAdapter(nn.Module):
    def __init__(self, dim: int, key_size: int = 32):
        super().__init__()
        self.dim = dim
        self.key_size = key_size
        
        self.metric_generator = nn.Sequential(
            nn.Linear(dim, dim * dim),
            nn.Tanh()
        )
        
        self.key_encoder = nn.Linear(key_size, dim)
    
    def forward(self, hidden_state: torch.Tensor, key: Optional[torch.Tensor] = None):
        metric_flat = self.metric_generator(hidden_state)
        metric = metric_flat.view(-1, self.dim, self.dim)
        
        metric = (metric + metric.transpose(-2, -1)) / 2
        
        eye = torch.eye(self.dim, device=metric.device)
        metric = metric + 0.1 * eye.unsqueeze(0)
        
        if key is not None:
            key_enc = self.key_encoder(key)
            scale = torch.exp(key_enc).unsqueeze(-1)
            metric = metric * scale
        
        return metric


class TripleHyperbolicAdapter(nn.Module):
    def __init__(self, dim: int, c_p: float = 1e-3, c_l: float = -1.0, c_k: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.c_p = c_p
        self.c_l = c_l
        self.c_k = c_k
        
        self.poincare_proj = nn.Linear(dim, dim)
        self.lorentz_proj = nn.Linear(dim, dim)
        self.klein_proj = nn.Linear(dim, dim)
        
        self.weight_net = nn.Linear(dim, 3)
    
    def forward(self, x: torch.Tensor, metric: Optional[torch.Tensor] = None):
        x_p = self.poincare_proj(x)
        x_p = project_to_ball(x_p, eps=1e-5)
        
        x_l = self.lorentz_proj(x)
        
        x_k = self.klein_proj(x)
        x_k = project_to_ball(x_k, eps=1e-5)
        
        weights = torch.softmax(self.weight_net(x), dim=-1)
        
        if metric is not None:
            metric_det = torch.det(metric)
            metric_trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(dim=-1)
            metric_norm = torch.abs(torch.sum(metric, dim=(-2, -1)))
            
            metric_weights = torch.stack([metric_det, metric_trace, metric_norm], dim=-1)
            metric_weights = torch.softmax(metric_weights, dim=-1)
            
            weights = weights * metric_weights
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        output = (
            weights[:, 0:1] * x_p +
            weights[:, 1:2] * x_l +
            weights[:, 2:3] * x_k
        )
        
        return output, weights


class BellmanCoordinateAdapter(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.value_net = nn.Linear(state_dim, 1)
        self.q_net = nn.Linear(state_dim + action_dim, 1)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        if action is None:
            return self.value_net(state)
        
        sa = torch.cat([state, action], dim=-1)
        return self.q_net(sa)


class LagrangianAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def kinetic_energy(self, velocity: torch.Tensor, metric: torch.Tensor):
        v_expanded = velocity.unsqueeze(-1)
        kinetic = 0.5 * torch.bmm(
            torch.bmm(v_expanded.transpose(-2, -1), metric),
            v_expanded
        ).squeeze(-1).squeeze(-1)
        return kinetic
    
    def potential_energy(self, value: torch.Tensor):
        return -value
    
    def lagrangian(self, velocity: torch.Tensor, metric: torch.Tensor, value: torch.Tensor):
        T = self.kinetic_energy(velocity, metric)
        V = self.potential_energy(value)
        return T - V





def _get_transformer_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported transformer architecture for RSULF conversion")


def _extract_gpt2_weights(layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = getattr(layer, "attn", None)
    mlp = getattr(layer, "mlp", None)
    if attn is None or mlp is None:
        # Try another common naming (e.g. older GPT-2 implementations might vary, but HF usually consistent)
        # Some implementations use self_attn / mlp
        attn = getattr(layer, "self_attn", None) if attn is None else attn
        # GPT-2 block has 'attn' and 'mlp' usually.
        if attn is None or mlp is None:
            raise ValueError("Layer does not have expected attn/mlp structure for GPT-2")

    # Attention weights (c_attn is Conv1D)
    # Conv1D weights are (in_features, out_features)
    # c_attn combines Q, K, V
    c_attn = getattr(attn, "c_attn")
    weight = c_attn.weight # (d, 3*d)
    
    d = weight.shape[0]
    # Split Q, K, V
    # HF GPT2 implementation: split(3, dim=1)
    # But weight is (d, 3d), so split dim 1
    wq, wk, wv = torch.split(weight, d, dim=1)
    
    # MLP weights
    # c_fc: (d, 4d) -> W1 (up projection)
    # c_proj: (4d, d) -> W2 (down projection)
    # Note: Conv1D weights are (in, out)
    # So W1: (d, 4d) -> x @ W1
    # W2: (4d, d) -> h @ W2
    # But RSULF expects:
    # W1 for F.linear(x, W1) -> W1 should be (out, in) = (4d, d)
    # W2 for F.linear(h, W2) -> W2 should be (out, in) = (d, 4d)
    # Conv1D weights are stored as (in, out), so we need to transpose them for F.linear
    
    w1_weight = getattr(mlp, "c_fc").weight   # (d, 4d)
    w2_weight = getattr(mlp, "c_proj").weight # (4d, d)
    
    # Transpose for F.linear usage in RSULF
    W1 = w1_weight.t() # (4d, d)
    W2 = w2_weight.t() # (d, 4d)
    
    # For Q, K
    # Attention in GPT-2: x @ c_attn -> split
    # So Q = x @ wq. In RSULF we use g = WQ.t() @ WK ??
    # Wait, extract_metric(WQ, WK) uses WQ.t() @ WK
    # If we assume standard attention: score = (xWq)(xWk)^T = x Wq Wk^T x^T
    # g represents the "metric" in the space.
    # If we follow Mistral logic: WQ, WK are linear layer weights (out, in)
    # GPT-2 Conv1D weights are (in, out).
    # So we should transpose them to match Linear(in, out) weight shape (out, in).
    
    WQ = wq.t()
    WK = wk.t()
    
    return WQ, WK, W1, W2



def _extract_mistral_like_weights(layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    if attn is None or mlp is None:
        raise ValueError("Layer does not have expected self_attn/mlp structure")

    WQ = getattr(attn, "q_proj").weight
    WK = getattr(attn, "k_proj").weight
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if gate is None or down is None:
        raise ValueError("MLP does not have expected gate_proj/down_proj structure")
    if up is not None:
        W1 = 0.5 * (gate.weight + up.weight)
    else:
        W1 = gate.weight
    W2 = down.weight
    return WQ, WK, W1, W2


def _extract_weights_auto(layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Try Mistral/Llama style
    try:
        return _extract_mistral_like_weights(layer)
    except ValueError:
        pass
        
    # Try GPT-2 style
    try:
        return _extract_gpt2_weights(layer)
    except ValueError:
        pass
        
    # Try Bert style (if needed in future) or others
    
    raise ValueError(f"Could not extract weights from layer type: {type(layer)}")


def convert_transformer_to_rsulf_layers(
    model: nn.Module,
    laplacian: torch.Tensor,
    lr: float = 0.02,
    alpha: float = 0.04,
    beta: float = 0.01,
    gamma: float = 0.98,
) -> nn.ModuleList:
    layers = _get_transformer_layers(model)
    rs_layers: List[RSULF] = []

    for idx, layer in enumerate(layers):
        try:
            WQ, WK, W1, W2 = _extract_weights_auto(layer)
        except Exception as e:
            # print(f"Skipping layer {idx} due to extraction error: {e}")
            continue

        rs_layer = RSULF(
            d_model=WQ.shape[1],
            WQ=WQ,
            WK=WK,
            W1=W1,
            W2=W2,
            L_matrix=laplacian,
            lr=lr,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        rs_layers.append(rs_layer)

    return nn.ModuleList(rs_layers)


def build_mistral7b_rsulf(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    lr: float = 0.02,
    alpha: float = 0.04,
    beta: float = 0.01,
    gamma: float = 0.98,
) -> Tuple[nn.Module, nn.ModuleList]:
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers required")
    base = AutoModelForCausalLM.from_pretrained(model_name)
    base = base.to(device)
    d = int(base.config.hidden_size)
    L = torch.eye(d, device=device)
    rs_layers = convert_transformer_to_rsulf_layers(
        base,
        L,
        lr=lr,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    rs_stack = RSULFStack(rs_layers)
    rs_stack = rs_stack.to(device)
    return base, rs_stack


class MistralRSULFAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str = "cuda",
        lr: float = 0.02,
        alpha: float = 0.04,
        beta: float = 0.01,
        gamma: float = 0.98,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        base, rs_stack = build_mistral7b_rsulf(
            model_name=model_name,
            device=device,
            lr=lr,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.base = base
        self.rsulf_stack = rs_stack
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rsulf_cache: Optional[List[Optional[torch.Tensor]]] = None,
        **kwargs: Any,
    ):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden_states = list(outputs.hidden_states)
        rs_layers = list(self.rsulf_stack.layers)
        if rsulf_cache is None or len(rsulf_cache) != len(rs_layers):
            rsulf_cache = [None] * len(rs_layers)
        rs_cache_next: List[Optional[torch.Tensor]] = []
        h_rs: Optional[torch.Tensor] = None
        for idx, (layer, V_prev) in enumerate(zip(rs_layers, rsulf_cache)):
            # hidden_states[0] = embeddings, hidden_states[1] = first block 출력
            if idx + 1 >= len(hidden_states):
                break
            h_in = hidden_states[idx + 1]
            h_out, V_next = layer(h_in, V_prev)
            rs_cache_next.append(V_next)
            h_rs = h_out
        rsulf_logits = None
        if h_rs is not None and hasattr(self.base, "lm_head"):
            rsulf_logits = self.base.lm_head(h_rs)
        return {
            "logits": outputs.logits,
            "rsulf_logits": rsulf_logits,
            "hidden_states": outputs.hidden_states,
            "rsulf_hidden": h_rs,
            "rsulf_cache": rs_cache_next,
        }

    def generate(
        self,
        prompt: str,
        max_length: int = 64,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.base.eval()
        device = self.device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = self.base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_pretrained(self, save_dir: Union[str, Path]):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        payload = {
            "state_dict": self.rsulf_stack.state_dict(),
            "config": {
                "model_name": self.model_name,
                "lr": self.lr,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        }
        torch.save(payload, save_path / "rsulf.pt")

    @classmethod
    def from_pretrained(
        cls,
        save_dir: Union[str, Path],
        device: Optional[str] = None,
    ) -> "MistralRSULFAdapter":
        save_path = Path(save_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(save_path / "rsulf.pt", map_location="cpu")
        cfg = payload.get("config", {})
        model_name = str(save_path)
        lr = float(cfg.get("lr", 0.02))
        alpha = float(cfg.get("alpha", 0.04))
        beta = float(cfg.get("beta", 0.01))
        gamma = float(cfg.get("gamma", 0.98))
        obj = cls(
            model_name=model_name,
            device=device,
            lr=lr,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        state_dict = payload.get("state_dict", payload)
        obj.rsulf_stack.load_state_dict(state_dict, strict=False)
        return obj


class RealityStoneLLMAdapter(nn.Module):
    def __init__(self, config: LLMAdapterConfig):
        super().__init__()
        self.config = config
        
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers required")
        
        if config.use_causal_lm:
            self.pretrained_llm = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model_name
            )
        else:
            self.pretrained_llm = AutoModel.from_pretrained(
                config.pretrained_model_name
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        
        if config.freeze_pretrained:
            for param in self.pretrained_llm.parameters():
                param.requires_grad = False
        
        actual_hidden_dim = self.pretrained_llm.config.hidden_size
        
        if actual_hidden_dim != config.hidden_dim:
            self.dim_adapter = nn.Linear(actual_hidden_dim, config.hidden_dim)
            self.dim_adapter_back = nn.Linear(config.hidden_dim, actual_hidden_dim)
        else:
            self.dim_adapter = nn.Identity()
            self.dim_adapter_back = nn.Identity()
        
        if config.use_bellman_coordinates:
            self.bellman = BellmanCoordinateAdapter(
                config.hidden_dim,
                config.hidden_dim,
                config.gamma_bellman
            )
        else:
            self.bellman = None
        
        if config.use_riemannian_metric:
            self.metric_adapters = nn.ModuleList([
                RiemannianMetricAdapter(config.hidden_dim, config.key_size)
                for _ in range(config.num_hyperbolic_layers)
            ])
        else:
            self.metric_adapters = None
        
        if config.use_triple_hyperbolic:
            self.hyperbolic_adapters = nn.ModuleList([
                TripleHyperbolicAdapter(
                    config.hidden_dim,
                    config.c_poincare,
                    config.c_lorentz,
                    config.c_klein
                )
                for _ in range(config.num_hyperbolic_layers)
            ])
        else:
            self.hyperbolic_adapters = None
        
        if config.use_lagrangian:
            self.lagrangian = LagrangianAdapter(config.hidden_dim)
        else:
            self.lagrangian = None
        
        if config.hyperbolic_insertion_positions is None:
            total_layers = len(self.pretrained_llm.transformer.h) if hasattr(self.pretrained_llm, 'transformer') else 12
            step = max(1, total_layers // config.num_hyperbolic_layers)
            self.insertion_positions = list(range(0, total_layers, step))[:config.num_hyperbolic_layers]
        else:
            self.insertion_positions = config.hyperbolic_insertion_positions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        return_all: bool = False
    ):
        outputs = self.pretrained_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        metrics = []
        velocities = []
        lagrangian_losses = []
        
        adapted_hidden = None
        
        for idx, hidden in enumerate(hidden_states):
            if idx not in self.insertion_positions:
                continue
            
            layer_idx = self.insertion_positions.index(idx)
            
            if layer_idx >= len(self.hyperbolic_adapters):
                break
            
            adapted = self.dim_adapter(hidden)
            
            if self.metric_adapters is not None:
                metric = self.metric_adapters[layer_idx](adapted, key)
                metrics.append(metric)
            else:
                metric = None
            
            if self.hyperbolic_adapters is not None:
                prev_adapted = adapted.clone()
                adapted, weights = self.hyperbolic_adapters[layer_idx](adapted, metric)
                
                velocity = adapted - prev_adapted
                velocities.append(velocity)
                
                if self.lagrangian is not None and metric is not None:
                    value = torch.randn(adapted.shape[0], 1, device=adapted.device)
                    L = self.lagrangian.lagrangian(velocity.mean(dim=1), metric.mean(dim=1), value)
                    lagrangian_losses.append(L.mean())
            
            adapted_hidden = adapted
        
        if adapted_hidden is not None:
            final_hidden = self.dim_adapter_back(adapted_hidden)
        else:
            final_hidden = hidden_states[-1]
        
        if return_all:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'final_hidden': final_hidden,
                'adapted_hidden': adapted_hidden,
                'metrics': metrics,
                'velocities': velocities,
                'lagrangian_losses': lagrangian_losses,
            }
        
        return logits if logits is not None else final_hidden
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        key: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.pretrained_llm.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None
    ):
        outputs = self.forward(input_ids, attention_mask, key, return_all=True)
        
        if outputs['logits'] is not None:
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            lm_loss = torch.tensor(0.0, device=input_ids.device)
        
        lagrangian_loss = sum(outputs['lagrangian_losses']) if outputs['lagrangian_losses'] else torch.tensor(0.0, device=input_ids.device)
        
        metric_reg_loss = torch.tensor(0.0, device=input_ids.device)
        if outputs['metrics']:
            for metric in outputs['metrics']:
                det = torch.det(metric)
                det_loss = torch.abs(det - 1.0).mean()
                
                eigvals = torch.linalg.eigvalsh(metric)
                spd_loss = torch.relu(-eigvals).mean()
                
                metric_reg_loss += det_loss + spd_loss
        
        total_loss = (
            lm_loss +
            self.config.lagrangian_weight * lagrangian_loss +
            self.config.metric_regularization_weight * metric_reg_loss
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'lm': lm_loss.item(),
            'lagrangian': lagrangian_loss.item() if isinstance(lagrangian_loss, torch.Tensor) else 0.0,
            'metric_reg': metric_reg_loss.item()
        }


def convert_pretrained_llm_to_reality_stone(
    model_name: str,
    config: Optional[LLMAdapterConfig] = None,
    device: str = "cuda"
):
    if config is None:
        config = LLMAdapterConfig(pretrained_model_name=model_name)
    
    model = RealityStoneLLMAdapter(config)
    model = model.to(device)
    
    return model


def finetune_adapted_llm(
    model: RealityStoneLLMAdapter,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda"
):
    model = model.to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'pretrained_llm' not in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if 'pretrained_llm' in n], 'lr': lr * 0.1}
    ])
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            key = torch.randn(input_ids.shape[0], model.config.key_size, device=device)
            
            loss, loss_dict = model.compute_loss(input_ids, labels, attention_mask, key)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                key = torch.randn(input_ids.shape[0], model.config.key_size, device=device)
                
                loss, loss_dict = model.compute_loss(input_ids, labels, attention_mask, key)
                val_losses.append(loss.item())
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {sum(train_losses)/len(train_losses):.4f}")
        print(f"  Val Loss: {sum(val_losses)/len(val_losses):.4f}")
    
    return model


