import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import time

class BellmanCoordinateSystem(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.value_net = nn.Linear(state_dim, 1)
        self.q_net = nn.Linear(state_dim + action_dim, 1)
    
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_net(state)
    
    def q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q_net(sa)
    
    def bellman_error(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        current_q = self.q_value(state, action)
        next_v = self.value(next_state).detach()
        target_q = reward + self.gamma * next_v
        return F.mse_loss(current_q, target_q)


class RiemannianMetricTensor(nn.Module):
    def __init__(self, dim: int, use_encryption: bool = False, key_size: int = 32):
        super().__init__()
        self.dim = dim
        self.use_encryption = use_encryption
        self.key_size = key_size
        
        self.metric_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim * dim)
        )
        
        if use_encryption:
            self.key_net = nn.Linear(key_size, dim)
    
    def forward(
        self,
        state: torch.Tensor,
        key: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = state.shape[0]
        
        metric_flat = self.metric_net(state)
        metric = metric_flat.view(B, self.dim, self.dim)
        
        metric = (metric + metric.transpose(-2, -1)) / 2.0
        
        eye = torch.eye(self.dim, device=state.device, dtype=state.dtype)
        metric = metric + 0.1 * eye.unsqueeze(0)
        
        if self.use_encryption and key is not None:
            key_scale = torch.exp(self.key_net(key))
            metric = metric * key_scale.unsqueeze(-1)
        
        return metric
    
    def christoffel(self, metric: torch.Tensor) -> torch.Tensor:
        B, d, _ = metric.shape
        
        metric_inv = torch.linalg.inv(metric)
        
        eps = 1e-3
        metric_grad = torch.zeros(B, d, d, d, device=metric.device, dtype=metric.dtype)
        
        for k in range(d):
            idx_k = min(k + 1, d - 1)
            for i in range(d):
                for j in range(d):
                    metric_grad[:, k, i, j] = (metric[:, idx_k, i, j] - metric[:, k, i, j]) / eps
        
        gamma = torch.zeros(B, d, d, d, device=metric.device, dtype=metric.dtype)
        for k in range(d):
            for i in range(d):
                for j in range(d):
                    for l in range(d):
                        gamma[:, k, i, j] += 0.5 * metric_inv[:, k, l] * (
                            metric_grad[:, i, l, j] +
                            metric_grad[:, j, l, i] -
                            metric_grad[:, l, i, j]
                        )
        
        return gamma


class LagrangianEnergySystem(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def kinetic_energy(
        self,
        velocity: torch.Tensor,
        metric: torch.Tensor
    ) -> torch.Tensor:
        v = velocity.unsqueeze(-1)
        T = 0.5 * torch.bmm(torch.bmm(v.transpose(-2, -1), metric), v)
        return T.squeeze(-1).squeeze(-1)
    
    def potential_energy(self, value: torch.Tensor) -> torch.Tensor:
        return -value.squeeze(-1)
    
    def lagrangian(
        self,
        velocity: torch.Tensor,
        metric: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        T = self.kinetic_energy(velocity, metric)
        V = self.potential_energy(value)
        return T - V
    
    def action_integral(
        self,
        velocities: List[torch.Tensor],
        metrics: List[torch.Tensor],
        values: List[torch.Tensor],
        dt: float = 1.0
    ) -> torch.Tensor:
        action = 0.0
        for v, g, val in zip(velocities, metrics, values):
            L = self.lagrangian(v, g, val)
            action = action + L.mean() * dt
        return action


class TemporalCreativityModule(nn.Module):
    def __init__(self, dim: int, num_steps: int = 3):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        
        self.time_encoder = nn.Linear(1, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
    
    def forward(
        self,
        state: torch.Tensor,
        metric: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        device = state.device
        
        time_points = torch.linspace(0, 1, self.num_steps, device=device)
        time_enc = self.time_encoder(time_points.unsqueeze(-1))
        
        state_seq = state.unsqueeze(1).expand(-1, self.num_steps, -1)
        state_seq = state_seq + time_enc.unsqueeze(0)
        
        temporal_output, _ = self.gru(state_seq)
        
        derivative = (temporal_output[:, -1] - temporal_output[:, 0]) / self.num_steps
        
        d_exp = derivative.unsqueeze(-1)
        metric_inv = torch.linalg.inv(metric)
        creativity = torch.bmm(torch.bmm(d_exp.transpose(-2, -1), metric_inv), d_exp)
        creativity = torch.sqrt(torch.abs(creativity.squeeze(-1).squeeze(-1)))
        
        return derivative, creativity


class BellmanRiemannianNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        gamma: float = 0.99,
        use_encryption: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.bellman = BellmanCoordinateSystem(state_dim, action_dim, gamma)
        
        self.encoder = nn.Linear(state_dim, hidden_dim)
        
        self.metrics = nn.ModuleList([
            RiemannianMetricTensor(hidden_dim, use_encryption)
            for _ in range(num_layers)
        ])
        
        self.transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.lagrangian = LagrangianEnergySystem(hidden_dim)
        
        self.temporal = TemporalCreativityModule(hidden_dim)
        
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        x = self.encoder(state)
        
        bellman_value = self.bellman.value(state)
        
        metrics_list = []
        velocities_list = []
        values_list = []
        
        for i, (metric_module, transform) in enumerate(zip(self.metrics, self.transforms)):
            metric = metric_module(x, key)
            metrics_list.append(metric)
            
            x_prev = x.clone()
            x = transform(x)
            
            velocity = x - x_prev
            velocities_list.append(velocity)
            
            intermediate_value = self.value_head(x)
            values_list.append(intermediate_value)
        
        time_deriv, creativity = self.temporal(x, metrics_list[-1])
        
        action_value = self.lagrangian.action_integral(
            velocities_list,
            metrics_list,
            values_list
        )
        
        final_value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        if return_details:
            return {
                'value': final_value,
                'policy': policy_logits,
                'bellman_value': bellman_value,
                'action_integral': action_value,
                'creativity': creativity,
                'time_derivative': time_deriv,
                'metrics': metrics_list,
                'velocities': velocities_list
            }
        
        return {
            'value': final_value,
            'policy': policy_logits
        }
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        key: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        outputs = self.forward(state, action, key, return_details=True)
        
        bellman_loss = self.bellman.bellman_error(state, action, reward, next_state)
        
        value_target = reward.unsqueeze(-1) if reward.dim() == 1 else reward
        value_loss = F.mse_loss(outputs['value'], value_target)
        
        lagrangian_loss = outputs['action_integral']
        
        creativity_reg = outputs['creativity'].mean()
        
        total_loss = (
            1.0 * bellman_loss +
            0.5 * value_loss +
            0.1 * lagrangian_loss -
            0.01 * creativity_reg
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'bellman': bellman_loss.item(),
            'value': value_loss.item(),
            'lagrangian': lagrangian_loss.item(),
            'creativity': creativity_reg.item()
        }
        
        return total_loss, loss_dict


def demo_bellman_riemannian():
    print("=== Bellman-Riemannian Network Demo ===\n")
    
    state_dim = 64
    action_dim = 8
    hidden_dim = 128
    batch_size = 16
    
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Batch Size: {batch_size}\n")
    
    model = BellmanRiemannianNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        use_encryption=True
    )
    
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    state = torch.randn(batch_size, state_dim)
    action = F.one_hot(torch.randint(0, action_dim, (batch_size,)), action_dim).float()
    reward = torch.randn(batch_size)
    next_state = torch.randn(batch_size, state_dim)
    key = torch.randn(batch_size, 32)
    
    print("--- Forward Pass (with details) ---")
    start = time.time()
    outputs = model.forward(state, action, key, return_details=True)
    forward_time = time.time() - start
    
    print(f"Value shape: {outputs['value'].shape}")
    print(f"Policy shape: {outputs['policy'].shape}")
    print(f"Bellman value shape: {outputs['bellman_value'].shape}")
    print(f"Action integral: {outputs['action_integral'].item():.4f}")
    print(f"Creativity mean: {outputs['creativity'].mean().item():.4f}")
    print(f"Forward time: {forward_time*1000:.2f} ms\n")
    
    print("--- Loss Computation ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start = time.time()
    loss, loss_dict = model.compute_loss(state, action, reward, next_state, key)
    loss_time = time.time() - start
    
    print(f"Total Loss: {loss_dict['total']:.4f}")
    print(f"  Bellman: {loss_dict['bellman']:.4f}")
    print(f"  Value: {loss_dict['value']:.4f}")
    print(f"  Lagrangian: {loss_dict['lagrangian']:.4f}")
    print(f"  Creativity: {loss_dict['creativity']:.4f}")
    print(f"Loss computation time: {loss_time*1000:.2f} ms\n")
    
    print("--- Backward Pass ---")
    start = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_time = time.time() - start
    print(f"Backward time: {backward_time*1000:.2f} ms\n")
    
    print("--- Mini Training Loop (10 iterations) ---")
    model.train()
    for i in range(10):
        state = torch.randn(batch_size, state_dim)
        action = F.one_hot(torch.randint(0, action_dim, (batch_size,)), action_dim).float()
        reward = torch.randn(batch_size)
        next_state = torch.randn(batch_size, state_dim)
        key = torch.randn(batch_size, 32)
        
        loss, loss_dict = model.compute_loss(state, action, reward, next_state, key)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"Iter {i+1}: Loss = {loss_dict['total']:.4f}, "
                  f"Creativity = {loss_dict['creativity']:.4f}")
    
    print("\n--- Inference Test ---")
    model.eval()
    test_state = torch.randn(1, state_dim)
    test_key = torch.randn(1, 32)
    
    with torch.no_grad():
        start = time.time()
        result = model.forward(test_state, key=test_key, return_details=True)
        inference_time = time.time() - start
    
    value = result['value'].item()
    policy = torch.softmax(result['policy'], dim=-1).squeeze()
    creativity = result['creativity'].item()
    
    print(f"Predicted Value: {value:.4f}")
    print(f"Policy distribution: {policy.numpy()}")
    print(f"Creativity score: {creativity:.4f}")
    print(f"Inference time: {inference_time*1000:.2f} ms\n")
    
    print("--- Metric Tensor Analysis ---")
    metrics = result['metrics']
    print(f"Number of metric tensors: {len(metrics)}")
    for i, metric in enumerate(metrics):
        eigenvalues = torch.linalg.eigvalsh(metric[0])
        det = torch.det(metric[0])
        print(f"Layer {i+1}:")
        print(f"  Eigenvalues: min={eigenvalues.min().item():.4f}, "
              f"max={eigenvalues.max().item():.4f}")
        print(f"  Determinant: {det.item():.4f}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    torch.manual_seed(42)
    demo_bellman_riemannian()

