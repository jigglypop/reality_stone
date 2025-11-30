import subprocess
import sys

print("Building with maturin...")
result = subprocess.run([sys.executable, "-m", "maturin", "develop", "--release"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)

print("\nTesting import...")
from reality_stone import RSULFLayerCUDA
import torch
import numpy as np

print(f"numpy version: {np.__version__}")
print(f"torch version: {torch.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

wq = torch.randn(4096, 4096)
wk = torch.randn(1024, 4096)
w1 = torch.randn(11008, 4096)
w2 = torch.randn(4096, 11008)

layer = RSULFLayerCUDA(wq, wk, w1, w2, d_model=4096, r=256, device=device)
print(f"RSULFLayerCUDA created")

x = torch.randn(4, 4096).to(device)
out, v = layer(x)
print(f"Forward OK: {out.shape}")

print("\nAll tests passed!")

