import copy
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP

import reality_stone as rs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class HFModule(nn.Module):
    """Projection→FFT→Poincaré→Down 만 수행하는 초경량 모듈"""
    def __init__(self, in_dim, hidden_dim, out_dim, compression_rate=4, use_full_fft=False):
        super().__init__()
        self.C = hidden_dim // compression_rate

        # 입력 투영
        if hidden_dim > in_dim:
            self.proj = nn.Linear(in_dim, self.C, bias=False)
            self.sep  = False
        else:
            self.g1 = nn.Linear(in_dim, self.C, bias=False)
            self.g2 = nn.Linear(in_dim, self.C, bias=False)
            self.sep = True

        fft_len = self.C if use_full_fft else (self.C // 2 + 1)
        self.freq_w = nn.Parameter(torch.zeros(fft_len))

        self.c = 1e-4
        self.t = nn.Parameter(torch.tensor(0.3))
        self.ln = nn.LayerNorm(self.C)

        self.down  = nn.Linear(self.C, out_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 초기화
        std = 0.02
        if self.sep:
            nn.init.normal_(self.g1.weight, std=std)
            nn.init.normal_(self.g2.weight, std=std)
        else:
            nn.init.normal_(self.proj.weight, std=std)
        nn.init.zeros_(self.freq_w)
        nn.init.normal_(self.down.weight, std=std)

        self.use_full_fft = use_full_fft

    def forward(self, x):
        B, L, _ = x.shape

        # projection
        if self.sep:
            comp = F.silu(self.g1(x)) * self.g2(x)
        else:
            comp = F.silu(self.proj(x))

        # norm + flatten
        comp = self.ln(comp).reshape(-1, self.C)

        # Fourier
        if self.use_full_fft:
            Xf = torch.fft.fft(comp, dim=1)
            w  = F.softmax(self.freq_w, dim=0).unsqueeze(0)
            flat = torch.fft.ifft(Xf*w, dim=1).real
        else:
            Xf = torch.fft.rfft(comp, dim=1)
            w  = F.softmax(self.freq_w, dim=0).unsqueeze(0)
            flat = torch.fft.irfft(Xf*w, n=self.C, dim=1)

        flat = torch.nan_to_num(flat)

        # hyperbolic
        shift = torch.nan_to_num(self.t * flat)
        hyp   = rs.poincare_ball_layer(flat, shift, self.c, self.t)
        hyp   = torch.nan_to_num(hyp).reshape(B, L, self.C)

        # down
        return self.down(hyp) * self.scale

def compress_llm_model(model_name, compression_rate=4, use_full_fft=False, device="cuda"):
    """
    원본 LLaMA 모델 로드 → 모든 LlamaMLP를 HFModule로 교체
    """
    orig_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    print(f"> Original params: {count_parameters(orig_model):,}")

    comp_model = copy.deepcopy(orig_model).to(device)

    # 첫 MLP 레이어 차원 추출
    for lyr in comp_model.model.layers:
        if hasattr(lyr, "mlp") and isinstance(lyr.mlp, LlamaMLP):
            in_d  = lyr.mlp.gate_proj.in_features
            hid_d = lyr.mlp.gate_proj.out_features
            out_d = lyr.mlp.down_proj.out_features
            break

    # HFModule 하나만 생성
    hf = HFModule(in_d, hid_d, out_d, compression_rate, use_full_fft).to(device)

    # 모든 mlp 완전 교체
    for lyr in comp_model.model.layers:
        if hasattr(lyr, "mlp") and isinstance(lyr.mlp, LlamaMLP):
            lyr.mlp = hf

    print(f"> Compressed params: {count_parameters(comp_model):,}")
    return orig_model, comp_model, tokenizer

def evaluate_models(orig, comp, tok, prompts, device="cuda"):
    """
    Perplexity, latency, speedup 계산
    """
    loss_f = nn.CrossEntropyLoss()
    print("\n=== Evaluation ===")
    for p in prompts:
        inp = tok(p, return_tensors="pt", padding=True).to(device)
        gc.collect(); torch.cuda.empty_cache()

        # orig
        t0   = time.time()
        out0 = orig(**inp)
        l0   = loss_f(out0.logits[:,:-1,:].reshape(-1, out0.logits.size(-1)),
                      inp.input_ids[:,1:].reshape(-1)).item()
        dt0  = time.time() - t0

        # comp
        t1   = time.time()
        out1 = comp(**inp)
        l1   = loss_f(out1.logits[:,:-1,:].reshape(-1, out1.logits.size(-1)),
                      inp.input_ids[:,1:].reshape(-1)).item()
        dt1  = time.time() - t1

        ppl0 = np.exp(l0)
        ppl1 = np.exp(l1)
        delta = (ppl1 - ppl0)/ppl0 * 100

        print(f"Prompt: {p}")
        print(f"  Orig: time={dt0:.3f}s, ppl={ppl0:.2f}")
        print(f"  Comp: time={dt1:.3f}s, ppl={ppl1:.2f}")
        print(f"  Δppl: {delta:+.1f}%, speedup: {dt0/dt1:.2f}×")
        print("-"*50)

def generate_and_compare(orig, comp, tok, prompt, max_length=50, device="cuda"):
    inp = tok(prompt, return_tensors="pt").to(device)
    gc.collect(); torch.cuda.empty_cache()

    o0 = orig.generate( input_ids=inp.input_ids, max_new_tokens=max_length,
                        do_sample=True, temperature=0.7, top_p=0.9 )
    o1 = comp.generate( input_ids=inp.input_ids, max_new_tokens=max_length,
                        do_sample=True, temperature=0.7, top_p=0.9 )
    print("\n=== Generation ===")
    print(f"Prompt: {prompt}")
    print("Original: ", tok.decode(o0[0], skip_special_tokens=True))
    print("Compressed:", tok.decode(o1[0], skip_special_tokens=True))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    orig, comp, tok = compress_llm_model(
        model_name, compression_rate=8, use_full_fft=False, device=device
    )

    prompts = [
        "The capital of France is Paris.",
        "Artificial intelligence is a field of study.",
        "The Earth orbits around the Sun."
    ]
    evaluate_models(orig, comp, tok, prompts, device=device)
    generate_and_compare(orig, comp, tok, "Once upon a time,", max_length=50, device=device)

if __name__=="__main__":
    main()
