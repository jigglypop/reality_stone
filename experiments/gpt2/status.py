import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reality_stone._rust import laplace_beltrami_matrix


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
    if len(hidden_states) == len(rs_outputs) + 2:
        n_layers = min(len(hidden_states) - 2, len(rs_outputs))
    print("\n[RS-ULF] Layer-wise similarity")
    for i in range(n_layers):
        o = hidden_states[i + 1]
        r = rs_outputs[i]
        o_flat = o.view(-1, o.size(-1))
        r_flat = r.view(-1, r.size(-1))
        cos = F.cosine_similarity(o_flat, r_flat, dim=-1).mean().item()
        rel = (o_flat - r_flat).norm() / (o_flat.norm() + 1e-8)
        print(f"   layer {i:02d}: cos={cos:.4f}, rel_l2={rel:.4f}")


def _laplace_beltrami_stats(x, name):
    x_np = x.detach().cpu().numpy()
    b, t, d = x_np.shape
    x_flat = x_np.reshape(b * t, d).astype(np.float32)
    l = laplace_beltrami_matrix(x_flat, "diagonal", 0.0, 0.5, 1e-6)
    row_sum = np.abs(l.sum(axis=1)).mean()
    frob = float(np.linalg.norm(l, ord="fro"))
    print(f"   [LB] {name}: n={x_flat.shape[0]}, dim={d}, fro={frob:.4e}, mean_row_sum_abs={row_sum:.4e}")


def analyze_layer_fidelity_blockwise(original_model, rs_model, tokenizer, prompt, device):
    original_model.eval()
    rs_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    pos = torch.arange(input_ids.size(1), dtype=torch.long, device=device)
    with torch.no_grad():
        x0 = wte(input_ids) + wpe(pos)
        teacher_blocks = list(original_model.transformer.h)
        teacher_states = []
        h = x0
        for block in teacher_blocks:
            out = block(h, attention_mask=attention_mask)
            h = out[0] if isinstance(out, (tuple, list)) else out
            teacher_states.append(h.detach())
        rs_states = []
        h_rs = x0
        for wrapper in rs_model.wrappers:
            h_rs = wrapper(h_rs)
            rs_states.append(h_rs.detach())
    n_layers = min(len(teacher_states), len(rs_states))
    print("\n[RS-ULF] Block-wise layer similarity")
    for i in range(n_layers):
        o = teacher_states[i]
        r = rs_states[i]
        o_flat = o.view(-1, o.size(-1))
        r_flat = r.view(-1, r.size(-1))
        cos = F.cosine_similarity(o_flat, r_flat, dim=-1).mean().item()
        rel = (o_flat - r_flat).norm() / (o_flat.norm() + 1e-8)
        print(f"   layer {i:02d}: cos={cos:.4f}, rel_l2={rel:.4f}")
    if n_layers > 0:
        ln_f = original_model.transformer.ln_f
        lm_head = original_model.lm_head
        with torch.no_grad():
            t_last = ln_f(teacher_states[n_layers - 1])
            s_last = ln_f(rs_states[n_layers - 1])
            t_flat = t_last.view(-1, t_last.size(-1))
            s_flat = s_last.view(-1, s_last.size(-1))
            cos_last = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
            rel_last = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)
            print(f"   final_hidden: cos={cos_last:.4f}, rel_l2={rel_last:.4f}")
            t_logits = lm_head(t_last)
            s_logits = lm_head(s_last)
            t_log_flat = t_logits.view(-1, t_logits.size(-1))
            s_log_flat = s_logits.view(-1, s_logits.size(-1))
            cos_log = F.cosine_similarity(t_log_flat, s_log_flat, dim=-1).mean().item()
            rel_log = (t_log_flat - s_log_flat).norm() / (t_log_flat.norm() + 1e-8)
            print(f"   final_logits: cos={cos_log:.4f}, rel_l2={rel_log:.4f}")
            _laplace_beltrami_stats(teacher_states[n_layers - 1], "teacher_last")
            _laplace_beltrami_stats(rs_states[n_layers - 1], "rsulf_last")


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
