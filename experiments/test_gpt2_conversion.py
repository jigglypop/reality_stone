import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from reality_stone.models.transformer_converter import (
    RSULFTransformerConverter,
    RSULFModel,
    FFNPotential,
    StructuralRSULFModel,
)
import time

def analyze_layer_fidelity(original_model, rs_model, tokenizer, prompt, device):
    original_model.eval()
    rs_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = original_model.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = list(outputs.hidden_states)
    x = hidden_states[0]
    rs_outputs = []
    h = x
    for wrapper in rs_model.wrappers:
        h = wrapper(h)
        rs_outputs.append(h.detach())
    n_layers = min(len(hidden_states) - 1, len(rs_outputs))
    print("\n[RS-ULF] Layer-wise similarity")
    for i in range(n_layers):
        o = hidden_states[i + 1]
        r = rs_outputs[i]
        o_flat = o.view(-1, o.size(-1))
        r_flat = r.view(-1, r.size(-1))
        cos = F.cosine_similarity(o_flat, r_flat, dim=-1).mean().item()
        rel = (o_flat - r_flat).norm() / (o_flat.norm() + 1e-8)
        print(f"   layer {i:02d}: cos={cos:.4f}, rel_l2={rel:.4f}")


def build_structural_rsulf_model(original_model, hidden_dim: int = None):
    if hidden_dim is None:
        hidden_dim = original_model.config.n_embd
    blocks = list(original_model.transformer.h)
    d_model = original_model.config.n_embd
    model = StructuralRSULFModel(blocks, d_model=d_model, hidden_dim=hidden_dim)
    return model


def distill_structural_potentials(
    original_model,
    structural_model,
    tokenizer,
    device,
    steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 32,
    lr: float = 1e-4,
    lambda_energy: float = 0.1,
):
    original_model.eval()
    structural_model.train()
    for layer in structural_model.layers:
        layer.ln_1.eval()
        layer.ln_2.eval()
        layer.attn.eval()
        for p in layer.ln_1.parameters():
            p.requires_grad = False
        for p in layer.ln_2.parameters():
            p.requires_grad = False
        for p in layer.attn.parameters():
            p.requires_grad = False
    potentials_params = []
    for layer in structural_model.layers:
        potentials_params.extend(list(layer.potential.parameters()))
    optimizer = torch.optim.AdamW(potentials_params, lr=lr)
    vocab_size = tokenizer.vocab_size
    teacher_blocks = list(original_model.transformer.h)
    num_layers = min(len(structural_model.layers), len(teacher_blocks))
    for step in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            tok_emb = original_model.transformer.wte(input_ids)
            pos = torch.arange(seq_len, dtype=torch.long, device=device)
            pos_emb = original_model.transformer.wpe(pos)
            x0 = tok_emb + pos_emb
        x_s = x0
        force_loss = 0.0
        energy_loss = 0.0
        for i in range(num_layers):
            layer_s = structural_model.layers[i]
            block_t = teacher_blocks[i]
            u = layer_s.ln_1(x_s)
            attn_out_full = layer_s.attn(u)
            attn_out = attn_out_full[0] if isinstance(attn_out_full, (tuple, list)) else attn_out_full
            y = x_s + attn_out
            w = layer_s.ln_2(y)
            with torch.no_grad():
                f_teacher = block_t.mlp(w)
                e_teacher = 0.5 * (f_teacher ** 2).sum(dim=-1)
            f_student = -layer_s.potential.gradient(w)
            phi_student = layer_s.potential(w)
            force_loss = force_loss + F.mse_loss(f_student, f_teacher)
            energy_loss = energy_loss + F.mse_loss(phi_student, e_teacher)
            x_s = y + f_teacher.detach()
        loss = force_loss + lambda_energy * energy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(potentials_params, 1.0)
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"[structural_potential] step={step+1}/{steps} loss={loss.item():.4f}")
    structural_model.eval()

class PotentialFunction(nn.Module):
    def __init__(self, w1, w2, activation="gelu"):
        super().__init__()
        self.W1 = nn.Parameter(w1.clone().detach())
        self.W2 = nn.Parameter(w2.clone().detach())
        self.W1.requires_grad = False
        self.W2.requires_grad = False
        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        elif activation == "silu":
            self.act = F.silu
        else:
            self.act = F.gelu

    def forward(self, x):
        h = self.act(F.linear(x, self.W1))
        y = F.linear(h, self.W2)
        phi = 0.5 * (y ** 2).sum(dim=-1)
        return phi

    def gradient(self, x):
        x_in = x.detach().requires_grad_(True)
        phi = self.forward(x_in).sum()
        grad = torch.autograd.grad(phi, x_in, create_graph=False)[0]
        return grad

class RSULFStudentAdapter(nn.Module):
    def __init__(self, rs_model, d_model, hidden_dim: int = None):
        super().__init__()
        self.rs_model = rs_model
        for p in self.rs_model.parameters():
            p.requires_grad = False
        num_layers = len(self.rs_model.wrappers)
        if hidden_dim is None:
            hidden_dim = d_model
        self.projections = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)])
        self.logit_adapter = nn.Linear(d_model, d_model, bias=False)
        self.potentials = nn.ModuleList([FFNPotential(d_model, hidden_dim=hidden_dim) for _ in range(num_layers)])

    def forward_hidden(self, x):
        rs_hiddens = []
        h = x
        for i, wrapper in enumerate(self.rs_model.wrappers):
            with torch.no_grad():
                h = wrapper(h)
            h = h - self.potentials[i].gradient(h)
            rs_hiddens.append(h)
        return rs_hiddens

    def project_layers(self, rs_hiddens):
        proj = []
        for h, proj_layer in zip(rs_hiddens, self.projections):
            proj.append(proj_layer(h))
        return proj

def distill_gpt2_to_rsulf(original_model, rs_model, tokenizer, device, steps=100, batch_size=4, seq_len=32, lr=1e-4, layer_loss_weight=1.0, logit_loss_weight=1.0):
    adapter = RSULFStudentAdapter(rs_model, original_model.config.n_embd).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    original_model.eval()
    rs_model.eval()
    vocab_size = tokenizer.vocab_size
    for step in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            teacher_out = original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            teacher_hidden = list(teacher_out.hidden_states)
            teacher_logits = teacher_out.logits
            tok_emb = original_model.transformer.wte(input_ids)
            pos = torch.arange(seq_len, dtype=torch.long, device=device)
            pos_emb = original_model.transformer.wpe(pos)
            x0 = tok_emb + pos_emb
        rs_hiddens = adapter.forward_hidden(x0)
        proj_hiddens = adapter.project_layers(rs_hiddens)
        num_layers = min(len(proj_hiddens), len(teacher_hidden) - 1)
        layer_loss = 0.0
        force_loss = 0.0
        energy_loss = 0.0
        for i in range(num_layers):
            t = teacher_hidden[i + 1]
            s = proj_hiddens[i]
            t_flat = t.view(-1, t.size(-1))
            s_flat = s.view(-1, s.size(-1))
            layer_loss = layer_loss + F.mse_loss(s_flat, t_flat)
            x_i = rs_hiddens[i]
            with torch.no_grad():
                f_teacher = original_model.transformer.h[i].mlp(x_i)
                phi_teacher = 0.5 * (f_teacher ** 2).sum(dim=-1)
            force_student = -adapter.potentials[i].gradient(x_i)
            phi_student = adapter.potentials[i](x_i)
            force_loss = force_loss + F.mse_loss(force_student, f_teacher)
            energy_loss = energy_loss + F.mse_loss(phi_student, phi_teacher)
        last_student = proj_hiddens[num_layers - 1]
        ln_f = original_model.transformer.ln_f
        lm_head = original_model.lm_head
        student_logits = lm_head(ln_f(last_student))
        logit_loss = F.mse_loss(student_logits, teacher_logits)
        loss = layer_loss_weight * layer_loss + logit_loss_weight * logit_loss + force_loss + 0.1 * energy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"[distill] step={step+1}/{steps} loss={loss.item():.4f}")
    return adapter

def exact_lift_ffn_check(original_model, converter, device, num_layers=4, num_samples=4, seq_len=16):
    original_model.eval()
    layers = original_model.transformer.h
    results = []
    for layer_idx in range(min(num_layers, len(layers))):
        layer = layers[layer_idx]
        weights = converter.extract_weights(layer)
        w1 = torch.tensor(weights["W1"], device=device)
        w2 = torch.tensor(weights["W2"], device=device)
        pot = PotentialFunction(w1, w2, activation="gelu").to(device)
        cos_list = []
        rel_list = []
        for _ in range(num_samples):
            x = torch.randn(1, seq_len, w2.shape[0], device=device)
            with torch.no_grad():
                ffn_out = layer.mlp(x)
            grad_phi = pot.gradient(x)
            f_flat = ffn_out.view(-1, ffn_out.size(-1))
            g_flat = grad_phi.view(-1, grad_phi.size(-1))
            cos = F.cosine_similarity(f_flat, g_flat, dim=-1).mean().item()
            rel = (f_flat - g_flat).norm() / (f_flat.norm() + 1e-8)
            cos_list.append(cos)
            rel_list.append(rel.item())
        avg_cos = float(sum(cos_list) / len(cos_list))
        avg_rel = float(sum(rel_list) / len(rel_list))
        results.append({"layer": layer_idx, "cos": avg_cos, "rel_l2": avg_rel})
    return results

def exact_lift_metric_check(original_model, converter, device, num_layers=4, num_samples=4, seq_len=16):
    original_model.eval()
    layers = original_model.transformer.h
    results = []
    for layer_idx in range(min(num_layers, len(layers))):
        layer = layers[layer_idx]
        weights = converter.extract_weights(layer)
        wq = torch.tensor(weights["WQ"], device=device)
        wk = torch.tensor(weights["WK"], device=device)
        d_out, d_model = wq.shape
        g = wq.t().mm(wk)
        cos_list = []
        rel_list = []
        for _ in range(num_samples):
            x = torch.randn(1, seq_len, d_model, device=device)
            q = F.linear(x, wq)
            k = F.linear(x, wk)
            tf_dots = torch.matmul(q, k.transpose(-1, -2))
            gx = torch.matmul(x, g)
            rs_dots = torch.matmul(x, gx.transpose(-1, -2))
            tf_flat = tf_dots.reshape(-1)
            rs_flat = rs_dots.reshape(-1)
            diff = tf_flat - rs_flat
            rel = diff.norm() / (tf_flat.norm() + 1e-8)
            cos = F.cosine_similarity(tf_flat.unsqueeze(0), rs_flat.unsqueeze(0), dim=-1)[0].item()
            cos_list.append(cos)
            rel_list.append(rel.item())
        avg_cos = float(sum(cos_list) / len(cos_list))
        avg_rel = float(sum(rel_list) / len(rel_list))
        results.append({"layer": layer_idx, "cos": avg_cos, "rel_l2": avg_rel})
    return results

def rsulf_generate_text(original_model, rs_model_stack, tokenizer, device, text_prompt, max_tokens=30):
    curr_ids = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    lm_head = original_model.lm_head
    ln_f = original_model.transformer.ln_f
    generated = curr_ids
    start_gen = time.time()
    for _ in range(max_tokens):
        seq_len = generated.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = wte(generated)
            pos_emb = wpe(pos)
            x = tok_emb + pos_emb
            if x.is_cuda:
                torch.cuda.synchronize()
            h = x
            for wrapper in rs_model_stack.wrappers:
                h = wrapper(h)
            h_last = ln_f(h)
            logits = lm_head(h_last)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - start_gen
    return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time


def rsulf_generate_text_distilled(original_model, rs_model_stack, adapter, tokenizer, device, text_prompt, max_tokens=30):
    curr_ids = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    lm_head = original_model.lm_head
    ln_f = original_model.transformer.ln_f
    generated = curr_ids
    start_gen = time.time()
    for _ in range(max_tokens):
        seq_len = generated.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = wte(generated)
            pos_emb = wpe(pos)
            x = tok_emb + pos_emb
            if x.is_cuda:
                torch.cuda.synchronize()
            rs_hiddens = adapter.forward_hidden(x)
            proj_hiddens = adapter.project_layers(rs_hiddens)
            h_last = proj_hiddens[-1]
            h_last = ln_f(h_last)
            logits = lm_head(h_last)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - start_gen
    return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time

def test_gpt2_conversion():
    print("=== [Reality Stone] GPT-2 Conversion Test Start ===")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Device: cpu (CUDA not available)")
        print("WARNING: User requested CUDA but it is not available. Running on CPU.")
    # 1. Load Original GPT-2
    print("\n1. Loading Original GPT-2...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    original_model.eval()

    prompt = "The secret of the universe is"
    # Fix: Generate attention_mask
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2. Generate with Original
    print("   Generating with Original...")
    start = time.time()
    with torch.no_grad():
        # Fix: Pass attention_mask and set pad_token_id explicitly
        out_ids = original_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=30, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Prevent "unified, unified" loop
        )
    orig_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"   [Original]: {orig_text}")
    print(f"   Time: {time.time() - start:.4f}s")
    print("\n2. Building Structural RS-ULF (PyTorch, LN-Attn-FFN(-∇Φ))...")
    structural_model = build_structural_rsulf_model(original_model).to(device)
    analyze_layer_fidelity(original_model, structural_model, tokenizer, prompt, device)
    distill_structural_potentials(
        original_model,
        structural_model,
        tokenizer,
        device,
        steps=50,
        batch_size=4,
        seq_len=32,
        lr=1e-4,
        lambda_energy=0.1,
    )
    analyze_layer_fidelity(original_model, structural_model, tokenizer, prompt, device)
    struct_text, struct_time = rsulf_generate_text(
        original_model,
        structural_model,
        tokenizer,
        device,
        prompt,
    )

    print("\n3. Converting to RS-ULF (Rust, Structure Mapping Mode)...")
    config = {
        "d_model": original_model.config.n_embd,
        "r": max(256, original_model.config.n_embd // 3),
        "eta": 0.005,
        "alpha": 0.01,
        "beta": 0.0,
        "gamma": 0.99,
        "seq_len": 64,
        "window": 4,
        "verbose": True
    }
    full_rank_r = config["r"]

    converter = RSULFTransformerConverter(**config, exact=True)
    metric_stats = exact_lift_metric_check(original_model, converter, device)
    ffn_stats = exact_lift_ffn_check(original_model, converter, device)
    print("\n[Exact-Lift Metric check]")
    for s in metric_stats:
        print(f"   layer {s['layer']:02d}: cos={s['cos']:.4f}, rel_l2={s['rel_l2']:.4f}")
    print("\n[Exact-Lift FFN potential check]")
    for s in ffn_stats:
        print(f"   layer {s['layer']:02d}: cos={s['cos']:.4f}, rel_l2={s['rel_l2']:.4f}")
    rs_layers = converter.convert_model(original_model)
    rs_layers = rs_layers.to(device)

    analyze_layer_fidelity(original_model, rs_layers, tokenizer, prompt, device)

    adapter = distill_gpt2_to_rsulf(
        original_model,
        rs_layers,
        tokenizer,
        device,
        steps=30,
        batch_size=4,
        seq_len=32,
        lr=1e-4,
        layer_loss_weight=1.0,
        logit_loss_weight=1.0,
    )

    rs_text, rs_time = rsulf_generate_text_distilled(
        original_model,
        rs_layers,
        adapter,
        tokenizer,
        device,
        prompt,
    )
    print(f"   [RS-ULF Converted]: {rs_text}")
    print(f"   Time: {rs_time:.4f}s")

    # 5. Convert to RS-ULF (Compressed Mode: SVD Rank 64)
    print("\n4. Converting to RS-ULF (High Compression Mode - Rank 64)...")
    config["r"] = 64
    converter_svd = RSULFTransformerConverter(**config, exact=False)
    rs_layers_svd = converter_svd.convert_model(original_model)
    rs_layers_svd = rs_layers_svd.to(device)

    analyze_layer_fidelity(original_model, rs_layers_svd, tokenizer, prompt, device)
    
    rs_svd_text, rs_svd_time = rsulf_generate_text(
        original_model,
        rs_layers_svd,
        tokenizer,
        device,
        prompt,
    )
    print(f"   [RS-ULF SVD]: {rs_svd_text}")
    print(f"   Time: {rs_svd_time:.4f}s")

    # Summary
    print("\n=== Summary ===")
    print(f"Prompt: {prompt}")
    print(f"1. Original:                      {orig_text.strip()}")
    print(f"2. Structural RS-ULF(Py):         {struct_text.strip()}")
    print(f"3. RS-ULF Rust (r={full_rank_r}): {rs_text.strip()}")
    print(f"4. RS-ULF Rust (r=64):            {rs_svd_text.strip()}")

if __name__ == "__main__":
    try:
        test_gpt2_conversion()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

