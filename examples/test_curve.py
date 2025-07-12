import torch
import torch.nn as nn
import torch.optim as optim
from reality_stone.layers.poincare import AdaptiveCurvatureNetwork, poincare_to_lorentz_adaptive

# Mock data
BATCH_SIZE = 4
IN_FEATURES = 10
OUT_FEATURES = 5
NUM_MANIFOLDS = 3

# Create a simple model with adaptive curvature
model = AdaptiveCurvatureNetwork(IN_FEATURES, OUT_FEATURES, NUM_MANIFOLDS)
# Initialize input tensor inside the Poincare ball
x = torch.randn(BATCH_SIZE, NUM_MANIFOLDS, IN_FEATURES) * 0.1 
x.requires_grad = True
x = x.cuda()
model = model.cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

print("--- Initial State ---")
print(f"Initial curvature (c): {model.get_c().cpu().numpy()}")
print("-" * 20)

# Training loop simulation
for i in range(5):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(x)
    
    # Simple loss
    target = torch.randn_like(output) * 0.1
    loss = (output - target).pow(2).sum()
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    print(f"--- Iteration {i+1} ---")
    print(f"Loss: {loss.item()}")
    print(f"Updated curvature (c): {model.get_c().cpu().numpy()}")
    if model.c.grad is not None:
        print(f"Gradient of c: {model.c.grad.cpu().numpy()}")
    else:
        print("Gradient of c is None")
    print("-" * 20)

print("\n--- Test finished ---")
print("Check the output for curvature values and their gradients.")
print("If 'c' becomes negative or its gradient explodes, it indicates instability.") 