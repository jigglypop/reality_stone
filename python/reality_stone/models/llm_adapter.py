import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

import reality_stone as rs
from reality_stone.layers.poincare import project_to_ball


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


