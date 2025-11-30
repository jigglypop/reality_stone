from reality_stone import RSULFLayerCUDA, RSULFLMHeadCUDA
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

wq = torch.randn(4096, 4096)
wk = torch.randn(1024, 4096)
w1 = torch.randn(11008, 4096)
w2 = torch.randn(4096, 11008)

layer = RSULFLayerCUDA(wq, wk, w1, w2, d_model=4096, r=256, device=device)
comp, orig, ratio = layer.param_count()
print(f'Compression: {ratio:.2f}x')

x = torch.randn(4, 128, 4096).to(device)
x_flat = x.view(-1, 4096)
out, v = layer(x_flat)
print(f'Input: {x_flat.shape}, Output: {out.shape}')

layers = [layer]
head = RSULFLMHeadCUDA(layers, 4096, 32000, device=device)
logits = head(x)
print(f'Logits: {logits.shape}')
print('OK')

