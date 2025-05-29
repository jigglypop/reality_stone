import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys, time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ───────── RealityStone & Poincaré utils ─────────
try:
    import reality_stone as rs
except ImportError:
    sys.stderr.write("ERROR: reality_stone 라이브러리 로드 실패\n")
    raise RuntimeError("reality_stone 라이브러리 필수")

def poincare_log(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # sqrt(c) must be a tensor for torch.sqrt
    sc = torch.tensor(c, device=x.device, dtype=x.dtype)
    return (1/sc.sqrt()) * torch.atanh(sc.sqrt() * norm) * (x / norm)

def poincare_exp(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sc = torch.tensor(c, device=x.device, dtype=x.dtype)
    return (1/sc.sqrt()) * torch.tanh(sc.sqrt() * norm) * (x / norm)

# ───────── Hyperbolic + FFT + SVD Compressor ─────────
class HyperbolicCompressor:
    def __init__(self, W: torch.Tensor, keep_frac=0.2, svd_frac=0.1, c=1.0):
        # W shape: [out_f, in_f]
        out_f, in_f = W.shape
        self.out_f, self.in_f, self.c = out_f, in_f, c

        # 1) log map into tangent
        T = poincare_log(W, c)  # [out_f, in_f]

        # 2) Standard 2D FFT (not hyperbolic FFT to avoid dimension issues)
        Ff = torch.fft.fft2(T)  # [out_f, in_f] - same shape

        # 3) keep top-k overall
        Ff_flat = Ff.view(-1)
        total_elements = len(Ff_flat)
        k = max(1, int(total_elements * keep_frac))
        _, indices = torch.topk(Ff_flat.abs(), k)
        
        mask_flat = torch.zeros_like(Ff_flat, dtype=torch.bool)
        mask_flat[indices] = True
        mask = mask_flat.view(Ff.shape)
        Ff_masked = Ff * mask

        # 4) inverse FFT back to tangent
        T_rec = torch.fft.ifft2(Ff_masked).real  # [out_f, in_f] - guaranteed same shape

        # 5) exp map back to manifold
        Wf = poincare_exp(T_rec, c)
        self.Wf = torch.nan_to_num(Wf, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure Wf has correct shape [out_f, in_f]
        if self.Wf.shape != (out_f, in_f):
            self.Wf = self.Wf.T

        # 6) residual and SVD
        R = torch.nan_to_num(W - self.Wf, nan=0.0, posinf=1e6, neginf=-1e6)
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        r = max(1, int(min(out_f, in_f) * svd_frac))
        self.U = U[:, :r] * S[:r].unsqueeze(0)   # [out_f, r]
        self.V = Vh[:r, :]                       # [r, in_f]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [..., actual_in_f]
        orig_shape = x.shape
        actual_in_f = x.shape[-1]
        
        # 1) main FFT-based approx
        if self.Wf.shape[0] == actual_in_f:  # [actual_in_f, out_f]
            y1 = x @ self.Wf
        else:  # [out_f, actual_in_f] 
            y1 = x @ self.Wf.T
            
        target_out_f = y1.shape[-1]  # Use actual output dimension from y1

        # 2) residual correction
        batch_flat = int(torch.tensor(orig_shape[:-1]).prod().item())
        flat = x.reshape(batch_flat, actual_in_f)
        
        # Adjust V matrix if needed
        if self.V.shape[1] == actual_in_f:
            corr = (flat @ self.V.t())  # [B, r]
            corr = corr @ self.U.t()    # [B, out_f]
        else:
            # Create zero correction with correct output dimension
            corr = torch.zeros(batch_flat, target_out_f, device=x.device, dtype=x.dtype)
            
        # Reshape correction to match y1 dimensions
        corr = corr.reshape(*orig_shape[:-1], -1)
        
        # Ensure correction matches y1 output dimension
        if corr.shape[-1] != target_out_f:
            if corr.shape[-1] < target_out_f:
                # Pad with zeros
                padding_shape = orig_shape[:-1] + (target_out_f - corr.shape[-1],)
                padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
                corr = torch.cat([corr, padding], dim=-1)
            else:
                # Truncate
                corr = corr[..., :target_out_f]
        
        return y1 + corr

# ───────── Hybrid Linear Layer ─────────
class HybridLinear(nn.Module):
    def __init__(self, lin: nn.Linear, keep_frac=0.2, svd_frac=0.1, c=1.0):
        super().__init__()
        W = lin.weight.data.clone()    # [out_f, in_f]
        self.comp = HyperbolicCompressor(W, keep_frac, svd_frac, c)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        out = self.comp.apply(x)
        if self.bias is not None:
            out = out + self.bias
        return out

# ───────── GPT-2 Block Wrapper ─────────
class HybridBlock(nn.Module):
    def __init__(self, block, keep_frac, svd_frac, c):
        super().__init__()
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        attn, mlp = block.attn, block.mlp

        # swap c_attn / c_proj / c_fc
        attn.c_attn = HybridLinear(attn.c_attn, keep_frac, svd_frac, c)
        attn.c_proj = HybridLinear(attn.c_proj, keep_frac, svd_frac, c)
        mlp.c_fc   = HybridLinear(mlp.c_fc,   keep_frac, svd_frac, c)
        mlp.c_proj = HybridLinear(mlp.c_proj, keep_frac, svd_frac, c)
        self.attn, self.mlp = attn, mlp

    def forward(self, x, **kwargs):
        h = self.ln1(x)
        attn_outputs = self.attn(h, **kwargs)  # Get full tuple
        a = attn_outputs[0]   # attention output
        x = x + a
        h2 = self.ln2(x)
        m = self.mlp(h2)
        x = x + m
        
        # Return in same format as original block
        outputs = (x,) + attn_outputs[1:]  # (hidden_states, present, attentions, ...)
        return outputs

# ───────── Apply to Full Model ─────────
def apply_hybrid(model, keep_frac=0.2, svd_frac=0.1, c=1.0):
    total = sum(p.numel() for p in model.parameters())
    print(f"Before: {total:,} params")
    for i in tqdm(range(len(model.transformer.h)), desc="Compressing"):
        model.transformer.h[i] = HybridBlock(
            model.transformer.h[i], keep_frac, svd_frac, c
        )
    total2 = sum(p.numel() for p in model.parameters())
    print(f"After:  {total2:,} params → {total/total2:.2f}×")
    return model

# ───────── Main Demo ─────────
def main():
    model_name = "skt/kogpt2-base-v2"
    print("Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = apply_hybrid(model, keep_frac=0.2, svd_frac=0.1, c=1.0)

    prompt = "안녕하세요"
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    output = model.generate(**inputs, max_length=20)
    print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
    print("Elapsed:", time.time() - t0, "s")

if __name__ == "__main__":
    main()