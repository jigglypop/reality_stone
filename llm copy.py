# complete_volterra_finetune_light.py

import copy, time, gc
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaMLP
import reality_stone as rs  # poincare_ball_layer 제공

# -----------------------------------------------------------------------------
# 1) Very Lightweight Volterra Branch (1st order only)
# -----------------------------------------------------------------------------
class FastVolterra1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(dim//2 + 1))
    def forward(self, hidden):
        B,L,D = hidden.shape
        flat = hidden.reshape(-1, D)
        Xf1  = torch.fft.rfft(flat, dim=1)
        w1   = F.softmax(self.w1, dim=0).unsqueeze(0)
        t1   = torch.fft.irfft(Xf1 * w1, n=D, dim=1)
        return t1.view(B, L, D)

# -----------------------------------------------------------------------------
# 2) PartialVolterraMLP: apply branch only on 1/4 of layers, residual weight small
# -----------------------------------------------------------------------------
class PartialVolterraMLP(nn.Module):
    def __init__(self, orig: LlamaMLP, apply_branch: bool):
        super().__init__()
        # keep original FFN projections
        self.gate_proj = orig.gate_proj
        self.up_proj   = orig.up_proj
        self.down_proj = orig.down_proj

        self.apply = apply_branch
        if apply_branch:
            D = orig.gate_proj.out_features
            self.branch = FastVolterra1(D)
            # very small residual coeff
            self.beta = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        # original FFN
        h        = F.silu(self.gate_proj(x)) * self.up_proj(x)
        orig_out = self.down_proj(h)
        # optionally add lightweight Volterra
        if self.apply:
            v       = self.branch(h)
            vol_out = self.down_proj(v)
            # residual mix
            orig_out = orig_out + self.beta.clamp(0,1) * vol_out
        return orig_out

# -----------------------------------------------------------------------------
# 3) Compression + JIT compile
# -----------------------------------------------------------------------------
def compress_and_compile(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading {model_name} on {device}")
    orig  = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok   = AutoTokenizer.from_pretrained(model_name)
    print(f"[+] Original params: {sum(p.numel() for p in orig.parameters() if p.requires_grad):,}")

    comp = copy.deepcopy(orig).to(device)
    replaced = 0
    # 4개 중 1개 레이어에만 경량 브랜치 적용
    for idx, lyr in enumerate(comp.model.layers):
        if hasattr(lyr, "mlp") and isinstance(lyr.mlp, LlamaMLP):
            apply_branch = (idx % 4 == 0)
            lyr.mlp = PartialVolterraMLP(lyr.mlp, apply_branch).to(device)
            replaced += 1
            if replaced % 8 == 0:
                gc.collect(); torch.cuda.empty_cache()

    print(f"[+] Applied PartialVolterraMLP to {replaced} layers")
    print(f"[+] Compressed params: {sum(p.numel() for p in comp.parameters() if p.requires_grad):,}")

    # torch.compile 으로 전체 모델 JIT 컴파일
    print("[+] JIT-compiling compressed model with torch.compile...")
    comp = torch.compile(comp, backend="inductor")
    return orig, comp, tok, device

# -----------------------------------------------------------------------------
# 4) Data preparation
# -----------------------------------------------------------------------------
def tokenize_fn(ex, tokenizer, block_size=512):
    return tokenizer(ex["text"], truncation=True, max_length=block_size)

def group_texts(examples, block_size=512):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // block_size) * block_size
    result = {
        k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# -----------------------------------------------------------------------------
# 5) Training & Evaluation
# -----------------------------------------------------------------------------
def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./volterra_finetuned_light"

    # 1) compress & compile
    orig, comp, tok, device = compress_and_compile(model_name)

    # 2) dataset
    print("[+] Loading and tokenizing dataset")
    ds_raw = load_dataset("wikitext", "wikitext-103-v1", split="train")
    ds_tok = ds_raw.map(lambda ex: tokenize_fn(ex, tok), batched=True, remove_columns=["text"])
    ds_lm  = ds_tok.map(group_texts, batched=True, batch_size=1000)

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # 3) training args
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=2000,
        bf16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=comp,
        args=args,
        train_dataset=ds_lm,
        data_collator=data_collator,
    )

    # 4) train
    print("[+] Starting fine-tuning")
    trainer.train()
    trainer.save_model(output_dir)

    # 5) quick eval
    print("[+] Quick evaluation on sample prompts")
    loss_f = nn.CrossEntropyLoss()
    prompts = [
        "The capital of France is Paris.",
        "Artificial intelligence is a field of study.",
        "The Earth orbits around the Sun."
    ]
    for p in prompts:
        inp = tok(p, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = comp(**inp)
        l   = loss_f(out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)),
                     inp.input_ids[:,1:].reshape(-1)).item()
        print(f"Prompt: {p} → PPL={np.exp(l):.2f}")

    # 6) generation
    prompt="Once upon a time,"
    inp   = tok(prompt, return_tensors="pt").to(device)
    print(f"\n=== Generation: {prompt} ===")
    gen = comp.generate(input_ids=inp.input_ids, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9)
    print("Comp model:", tok.decode(gen[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
