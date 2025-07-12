import torch
import torch.nn as nn
import torch.optim as optim
from reality_stone import dynamic_poincare_ball_layer, dynamic_mobius_add

torch.manual_seed(42)

batch_size = 8
dim = 16

u = torch.randn(batch_size, dim) * 0.1
v = torch.randn(batch_size, dim) * 0.1

kappa = torch.nn.Parameter(torch.tensor(0.0))
c_min = -2.0
c_max = -0.1

u.requires_grad = True
v.requires_grad = True

optimizer = optim.Adam([kappa], lr=0.01)

print("Initial kappa:", kappa.item())
initial_c = c_min + (c_max - c_min) / (1.0 + torch.exp(-kappa))
print("Initial curvature:", initial_c.item())

for i in range(10):
    optimizer.zero_grad()
    
    output = dynamic_mobius_add(u, v, kappa.item(), c_min, c_max)
    
    loss = output.norm()
    loss.backward()
    
    grad_kappa = kappa.grad
    print(f"Step {i+1}: loss={loss.item():.4f}, grad_kappa={grad_kappa.item():.4f}")
    
    optimizer.step()

print("\nFinal kappa:", kappa.item())
final_c = c_min + (c_max - c_min) / (1.0 + torch.exp(-kappa))
print("Final curvature:", final_c.item())

print("\nTesting Poincare Ball Layer with dynamic curvature:")
t = 0.5
output_layer = dynamic_poincare_ball_layer(u, v, kappa.item(), c_min, c_max, t)
print("Output shape:", output_layer.shape)
print("Output mean:", output_layer.mean().item()) 