# test_debug.py
import torch, hyper_butterfly as hb

x = torch.randn(4, 128, device='cuda')
params = torch.randn(128, device='cuda') * 1e-3
c, L = 1e-3, 1

print("→ log_map start")
u = hb.hyper_butterfly(x, c)
torch.cuda.synchronize()
print("   log_map ok, u[0]=", u[0,0].item())

