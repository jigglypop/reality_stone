import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from reality_stone.models.transformer_converter import FFNPotential, StructuralRSULFModel

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def _use_amp(device):
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
    return torch.cuda.is_available() and device_type == "cuda"


def _compute_rank_top1_losses(teacher_logits, student_logits, k: int):
    b, t, v = teacher_logits.shape
    if v == 0 or k <= 0:
        return teacher_logits.new_tensor(0.0), teacher_logits.new_tensor(0.0)
    k = min(k, v)
    t_flat = teacher_logits.view(-1, v)
    s_flat = student_logits.view(-1, v)
    topk_vals, topk_idx = torch.topk(t_flat, k, dim=-1)
    s_top = torch.gather(s_flat, 1, topk_idx)
    t_i = topk_vals.unsqueeze(2)
    t_j = topk_vals.unsqueeze(1)
    s_i = s_top.unsqueeze(2)
    s_j = s_top.unsqueeze(1)
    t_diff = t_i - t_j
    s_diff = s_i - s_j
    prod = t_diff * s_diff
    margin = 1.0
    rank_loss_mat = torch.relu(margin - prod)
    mask = (t_diff != 0).float()
    denom = mask.sum()
    if denom > 0:
        rank_loss = (rank_loss_mat * mask).sum() / denom
    else:
        rank_loss = teacher_logits.new_tensor(0.0)
    t_top1 = topk_vals.argmax(dim=-1)
    s_top1 = s_top.argmax(dim=-1)
    top1_loss = (t_top1 != s_top1).float().mean()
    return rank_loss, top1_loss


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
    use_amp = _use_amp(device)
    if num_layers > 1:
        layer_denominator = float(num_layers - 1)
    else:
        layer_denominator = 1.0
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos_emb = wpe(pos)
    total_tokens = steps * batch_size * seq_len
    if _tqdm is not None:
        iterator = _tqdm(range(steps), desc="[structural_potential]", leave=False)
    else:
        iterator = range(steps)
    for step in iterator:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            tok_emb = wte(input_ids)
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
                with autocast(enabled=use_amp):
                    f_teacher = block_t.mlp(w)
                    e_teacher = 0.5 * (f_teacher ** 2).sum(dim=-1)
                f_teacher = f_teacher.to(dtype=x_s.dtype)
                e_teacher = e_teacher.to(dtype=x_s.dtype)
            # 학습과 추론의 일치를 위해 mlp 사용
            # f_student는 LowRankFFN의 출력 (고정된 weight)
            f_student = layer_s.mlp(w)
            
            # Potential은 에너지 제약 조건 학습 (Auxiliary)
            phi_student = layer_s.potential(w)
            if num_layers > 1:
                base_weight = 1.0 + 2.0 * (float(i) / layer_denominator)
                if i == num_layers - 1:
                    layer_weight = base_weight * 5.0
                else:
                    layer_weight = base_weight
            else:
                layer_weight = 1.0
            force_loss = force_loss + layer_weight * F.mse_loss(f_student, f_teacher)
            energy_loss = energy_loss + layer_weight * F.mse_loss(phi_student, e_teacher)
            x_s = y + f_student.detach()
        loss = force_loss + lambda_energy * energy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(potentials_params, 1.0)
        optimizer.step()
        if _tqdm is not None:
            processed_tokens = (step + 1) * batch_size * seq_len
            iterator.set_postfix(
                step=step + 1,
                total_steps=steps,
                tokens=f"{processed_tokens}/{total_tokens}",
                loss=float(loss.item()),
            )
    structural_model.eval()


class RSULFStudentAdapter(nn.Module):
    def __init__(self, rs_model, d_model, hidden_dim: int = None, use_potentials: bool = True):
        super().__init__()
        self.rs_model = rs_model
        for p in self.rs_model.parameters():
            p.requires_grad = False
        self.use_potentials = bool(use_potentials)
        num_layers = len(self.rs_model.wrappers)
        if hidden_dim is None:
            hidden_dim = d_model
        self.projections = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)])
        self.logit_adapter = nn.Linear(d_model, d_model, bias=False)
        self.potentials = nn.ModuleList([FFNPotential(d_model, hidden_dim=hidden_dim) for _ in range(num_layers)])
        self.last_adapter = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward_hidden(self, x):
        rs_hiddens = []
        h = x
        for i, wrapper in enumerate(self.rs_model.wrappers):
            with torch.no_grad():
                h = wrapper(h)
            if self.use_potentials:
                h = h - self.potentials[i].gradient(h)
            rs_hiddens.append(h)
        return rs_hiddens

    def project_layers(self, rs_hiddens):
        proj = []
        for h, proj_layer in zip(rs_hiddens, self.projections):
            proj.append(proj_layer(h))
        return proj


def distill_gpt2_to_rsulf(
    original_model,
    rs_model,
    tokenizer,
    device,
    steps=100,
    batch_size=4,
    seq_len=32,
    lr=1e-4,
    layer_loss_weight=1.0,
    logit_loss_weight=1.0,
    use_potentials: bool = True,
    force_loss_weight: float = 1.0,
    energy_loss_weight: float = 0.1,
    delta_loss_weight: float = 0.0,
    rank_loss_weight: float = 0.0,
    top1_loss_weight: float = 0.0,
    rank_k: int = 32,
    kl_loss_weight: float = 0.0,
    kl_temperature: float = 1.0,
    cos_last_weight: float = 0.0,
):
    adapter = RSULFStudentAdapter(rs_model, original_model.config.n_embd, use_potentials=bool(use_potentials)).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    original_model.eval()
    rs_model.eval()
    vocab_size = tokenizer.vocab_size
    use_amp = _use_amp(device)
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    ln_f = original_model.transformer.ln_f
    lm_head = original_model.lm_head
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos_emb = wpe(pos)
    total_tokens = steps * batch_size * seq_len
    if _tqdm is not None:
        iterator = _tqdm(range(steps), desc="[distill_rsulf]", leave=False)
    else:
        iterator = range(steps)
    for step in iterator:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            with autocast(enabled=use_amp):
                teacher_out = original_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                tok_emb = wte(input_ids)
                x0 = tok_emb + pos_emb
            teacher_hidden = [h.float() for h in teacher_out.hidden_states]
            teacher_logits = teacher_out.logits.float()
            x0 = x0.float()
        rs_hiddens = adapter.forward_hidden(x0)
        proj_hiddens = adapter.project_layers(rs_hiddens)
        num_layers = min(len(proj_hiddens), len(teacher_hidden) - 1)
        layer_loss = 0.0
        force_loss = 0.0
        energy_loss = 0.0
        delta_loss = 0.0
        cos_last_loss = 0.0
        for i in range(num_layers):
            t = teacher_hidden[i + 1]
            s = proj_hiddens[i]
            if i == num_layers - 1:
                s = s + adapter.last_adapter(s)
            t_flat = t.view(-1, t.size(-1))
            s_flat = s.view(-1, s.size(-1))
            scale = 5.0 if i == num_layers - 1 else 1.0
            layer_loss = layer_loss + scale * F.mse_loss(s_flat, t_flat)
            if cos_last_weight > 0.0 and i == num_layers - 1:
                t_norm = F.normalize(t_flat, dim=-1)
                s_norm = F.normalize(s_flat, dim=-1)
                cos_val = (t_norm * s_norm).sum(dim=-1).mean()
                cos_last_loss = cos_last_loss + (1.0 - cos_val)
            if adapter.use_potentials and (force_loss_weight > 0.0 or energy_loss_weight > 0.0):
                x_in = x0 if i == 0 else rs_hiddens[i - 1]
                wrapper = rs_model.wrappers[i]
                h_rs_raw = wrapper(x_in)
                if force_loss_weight > 0.0:
                    with torch.no_grad():
                        block_t = original_model.transformer.h[i]
                        with autocast(enabled=use_amp):
                            h_tea_raw = block_t(x_in.detach(), attention_mask=None)[0]
                        target_grad = (h_rs_raw.detach() - h_tea_raw).float()
                    force_student = adapter.potentials[i].gradient(h_rs_raw)
                    force_loss = force_loss + F.mse_loss(force_student, target_grad)
                if energy_loss_weight > 0.0:
                    phi_student = adapter.potentials[i](h_rs_raw)
                    energy_loss = energy_loss + F.mse_loss(phi_student, torch.zeros_like(phi_student))
        if delta_loss_weight > 0.0 and num_layers > 1:
            for i in range(num_layers - 1):
                t_high = teacher_hidden[i + 2]
                t_low = teacher_hidden[i + 1]
                s_high = proj_hiddens[i + 1]
                s_low = proj_hiddens[i]
                delta_t = (t_high - t_low).view(-1, t_high.size(-1))
                delta_s = (s_high - s_low).view(-1, s_high.size(-1))
                delta_loss = delta_loss + F.mse_loss(delta_s, delta_t)
        last_student = proj_hiddens[num_layers - 1]
        last_student = last_student + adapter.last_adapter(last_student)
        student_logits = lm_head(ln_f(last_student))
        logit_loss = F.mse_loss(student_logits, teacher_logits)
        rank_loss = student_logits.new_tensor(0.0)
        top1_loss = student_logits.new_tensor(0.0)
        kl_loss = student_logits.new_tensor(0.0)
        if kl_loss_weight > 0.0 and kl_temperature > 0.0:
            T = kl_temperature
            t_logits = teacher_logits / T
            s_logits = student_logits / T
            t_log_probs = F.log_softmax(t_logits, dim=-1)
            s_log_probs = F.log_softmax(s_logits, dim=-1)
            t_probs = torch.exp(t_log_probs)
            kl = t_probs * (t_log_probs - s_log_probs)
            kl_loss = kl.sum(dim=-1).mean() * (T * T)
        if rank_loss_weight > 0.0 or top1_loss_weight > 0.0:
            rank_loss, top1_loss = _compute_rank_top1_losses(teacher_logits, student_logits, rank_k)
        loss = (
            layer_loss_weight * layer_loss
            + logit_loss_weight * logit_loss
            + float(force_loss_weight) * force_loss
            + float(energy_loss_weight) * energy_loss
            + delta_loss_weight * delta_loss
            + rank_loss_weight * rank_loss
            + top1_loss_weight * top1_loss
            + kl_loss_weight * kl_loss
            + cos_last_weight * cos_last_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if _tqdm is not None:
            processed_tokens = (step + 1) * batch_size * seq_len
            iterator.set_postfix(
                step=step + 1,
                total_steps=steps,
                tokens=f"{processed_tokens}/{total_tokens}",
                loss=float(loss.item()),
            )
    return adapter


class SyntaxHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        y = self.fc2(h)
        return x + y


def forward_rsulf_core(rs_model_stack, x):
    h = x
    if hasattr(rs_model_stack, "wrappers"):
        for wrapper in rs_model_stack.wrappers:
            h = wrapper(h)
    elif hasattr(rs_model_stack, "layers"):
        for layer in rs_model_stack.layers:
            out, _ = layer(h, None)
            h = out
    return h


def distill_syntax_head(
    original_model,
    rs_model_stack,
    tokenizer,
    device,
    steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 32,
    lr: float = 1e-4,
    hidden_dim: int = None,
):
    d_model = original_model.config.n_embd
    if hidden_dim is None:
        hidden_dim = d_model
    head = SyntaxHead(d_model, hidden_dim).to(device)
    for p in original_model.parameters():
        p.requires_grad = False
    for layer in rs_model_stack.layers:
        for p in layer.parameters():
            p.requires_grad = False
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    vocab_size = tokenizer.vocab_size
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    ln_f = original_model.transformer.ln_f
    lm_head = original_model.lm_head
    original_model.eval()
    rs_model_stack.eval()
    use_amp = _use_amp(device)
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos_emb = wpe(pos)
    total_tokens = steps * batch_size * seq_len
    if _tqdm is not None:
        iterator = _tqdm(range(steps), desc="[syntax_head]", leave=False)
    else:
        iterator = range(steps)
    for step in iterator:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            with autocast(enabled=use_amp):
                teacher_out = original_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                teacher_logits = teacher_out.logits
                tok_emb = wte(input_ids)
                x0 = tok_emb + pos_emb
            teacher_logits = teacher_logits.float()
            x0 = x0.float()
        h_core = forward_rsulf_core(rs_model_stack, x0)
        h_syn = head(h_core)
        h_last = ln_f(h_syn)
        student_logits = lm_head(h_last)
        loss = F.mse_loss(student_logits, teacher_logits)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if _tqdm is not None:
            processed_tokens = (step + 1) * batch_size * seq_len
            iterator.set_postfix(
                step=step + 1,
                total_steps=steps,
                tokens=f"{processed_tokens}/{total_tokens}",
                loss=float(loss.item()),
            )
    head.eval()
    return head
