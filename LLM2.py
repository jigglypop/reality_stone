import copy
import time
import gc
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
import reality_stone as rs 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GatePoincare(nn.Module):
    """
    원래 Linear(proj) + Poincaré 비선형을 결합한 모듈.
    입력 dim -> 출력 dim은 Linear가 결정하고,
    그 결과에 작은 shift를 곱한 뒤 poincare_ball_layer 적용.
    """
    def __init__(self, linear: nn.Linear, c=1e-4, t_init=1e-3):
        super().__init__()
        # 원래 가중치 재사용
        self.linear = linear
        self.c = c
        self.t = nn.Parameter(torch.tensor(t_init))
    def forward(self, x):
        # x: (B, L, in_dim)
        out_lin = self.linear(x)                     # (B, L, out_dim)
        B,L,D = out_lin.shape
        flat = out_lin.reshape(-1, D)                # (B*L, D)
        shift = flat * self.t                        # tangent shift
        hyp = rs.poincare_ball_layer(flat, shift, self.c, self.t)
        hyp = torch.nan_to_num(hyp)
        return hyp.reshape(B, L, D)

def replace_mlp_gate_up(layer: LlamaMLP):
    """
    LlamaMLP 내 gate_proj & up_proj을 GatePoincare로 감싸기.
    down_proj은 그대로 둡니다.
    """
    # wrap gate_proj
    orig_gate = layer.gate_proj
    layer.gate_proj = GatePoincare(orig_gate)

    # wrap up_proj
    orig_up = layer.up_proj
    layer.up_proj   = GatePoincare(orig_up)

def compress_llm_poincare_mlp_wrap(model_name: str):
    """
    CPU에서 LLaMA 로드 → 각 레이어의 gate_proj, up_proj만 poincare로 감싸기
    """
    device = torch.device("cpu")
    print(f"[+] Loading {model_name} on CPU")
    orig = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok  = AutoTokenizer.from_pretrained(model_name)
    print(f"[+] Original params: {count_parameters(orig):,}")

    comp = copy.deepcopy(orig).to(device)
    replaced = 0
    for lyr in comp.model.layers:
        if hasattr(lyr, "mlp") and isinstance(lyr.mlp, LlamaMLP):
            replace_mlp_gate_up(lyr.mlp)
            replaced += 1
            if replaced % 5 == 0:
                gc.collect()
    print(f"[+] Wrapped gate/up in {replaced} MLP layers")
    print(f"[+] Compressed params: {count_parameters(comp):,}")
    return orig, comp, tok

def evaluate(orig, comp, tok, prompts):
    """
    Perplexity & 추론 속도 비교
    """
    loss_f = nn.CrossEntropyLoss()
    print("\n=== Evaluation ===")
    for p in prompts:
        inp = tok(p, return_tensors="pt", padding=True).to("cpu")
        gc.collect()

        # original
        t0 = time.time()
        with torch.no_grad():
            out0 = orig(**inp)
        l0 = loss_f(
            out0.logits[:, :-1, :].reshape(-1, out0.logits.size(-1)),
            inp.input_ids[:, 1:].reshape(-1)
        ).item()
        dt0 = time.time() - t0

        # compressed
        t1 = time.time()
        with torch.no_grad():
            out1 = comp(**inp)
        l1 = loss_f(
            out1.logits[:, :-1, :].reshape(-1, out1.logits.size(-1)),
            inp.input_ids[:, 1:].reshape(-1)
        ).item()
        dt1 = time.time() - t1

        ppl0 = np.exp(l0)
        ppl1 = np.exp(l1)
        speedup = dt0 / dt1 if dt1 > 0 else float('inf')

        print(f"Prompt: {p}")
        print(f"  Orig: time={dt0:.3f}s, PPL={ppl0:.2f}")
        print(f"  Comp: time={dt1:.3f}s, PPL={ppl1:.2f}, speedup={speedup:.2f}×")
        print("-"*50)

def generate_compare(orig, comp, tok, prompt, max_new_tokens=20):
    """
    텍스트 생성 품질 & 속도 비교
    """
    inp = tok(prompt, return_tensors="pt").to("cpu")
    gc.collect()

    t0 = time.time()
    o0 = orig.generate(
        input_ids=inp.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9
    )
    dt0 = time.time() - t0

    t1 = time.time()
    o1 = comp.generate(
        input_ids=inp.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9
    )
    dt1 = time.time() - t1

    text0 = tok.decode(o0[0], skip_special_tokens=True)
    text1 = tok.decode(o1[0], skip_special_tokens=True)
    speedup = dt0 / dt1 if dt1 > 0 else float('inf')

    print("\n=== Generation Compare ===")
    print(f"Prompt: {prompt}")
    print(f"Original ({dt0:.2f}s): {text0}")
    print(f"Compressed({dt1:.2f}s): {text1}")
    print(f"Speedup: {speedup:.2f}×")

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    orig, comp, tok = compress_llm_poincare_mlp_wrap(model_name)

    prompts = [
        "The capital of France is Paris.",
        "Artificial intelligence is a field of study.",
        "The Earth orbits around the Sun."
    ]
    evaluate(orig, comp, tok, prompts)
    generate_compare(orig, comp, tok, "Once upon a time,", max_new_tokens=10)

if __name__ == "__main__":
    main()
