import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from reality_stone.models.transformer_converter import FFNPotential, StructuralRSULFModel
from reality_stone.utils.sampling import sample_next_token
from torch.amp import autocast as autocast_v2

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


@torch.no_grad()
def _extract_gpt2_ffn_weights(block: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    mlp = getattr(block, "mlp", None)
    if mlp is None or not hasattr(mlp, "c_fc") or not hasattr(mlp, "c_proj"):
        raise ValueError("Block does not look like GPT-2: missing mlp.c_fc/c_proj")
    w1_w = mlp.c_fc.weight.detach()
    w2_w = mlp.c_proj.weight.detach()
    if w1_w.dim() != 2 or w2_w.dim() != 2:
        raise ValueError("Unexpected GPT-2 FFN weight rank")
    ln2 = getattr(block, "ln_2", None)
    if ln2 is None or not hasattr(ln2, "weight"):
        raise ValueError("Block does not have ln_2.weight")
    d_model = int(ln2.weight.numel())
    if d_model <= 0:
        raise ValueError("Invalid d_model")
    if w1_w.size(0) == d_model:
        w1 = w1_w.t().contiguous()
    elif w1_w.size(1) == d_model:
        w1 = w1_w.contiguous()
    else:
        raise ValueError("Unexpected GPT-2 c_fc.weight shape")
    if w2_w.size(1) == d_model:
        w2 = w2_w.t().contiguous()
    elif w2_w.size(0) == d_model:
        w2 = w2_w.contiguous()
    else:
        raise ValueError("Unexpected GPT-2 c_proj.weight shape")
    return w1.to(dtype=torch.float32), w2.to(dtype=torch.float32)


class RSFFNPrototype(nn.Module):
    def __init__(self, d_model: int, k: int, tau: float = 1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.k = int(k)
        self.router = nn.Linear(self.d_model, 2 * self.k, bias=False)
        self.bias = nn.Parameter(torch.zeros(2 * self.k))
        self.U = nn.Parameter(torch.zeros(self.d_model, self.k))
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(float(tau))))))
        self.z_scale = nn.Parameter(torch.ones(self.d_model))
        self.eta_logit = nn.Parameter(torch.tensor(-1.5))
        self.out_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        tau = torch.exp(self.log_tau).clamp_min(1e-4)
        logits = (self.router(h) + self.bias) / tau
        w = torch.softmax(logits, dim=-1)
        a = w[..., : self.k] - w[..., self.k :]
        delta_u = torch.matmul(a, self.U.t())
        eta = torch.sigmoid(self.eta_logit)
        delta_z = h * self.z_scale
        return self.out_scale * ((1.0 - eta) * delta_z + eta * delta_u)

    @torch.no_grad()
    def init_from_gpt2_ffn(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        if w1.dim() != 2 or w2.dim() != 2:
            raise ValueError("w1 and w2 must be 2D tensors")
        d = int(self.d_model)
        if w1.size(1) != d:
            raise ValueError("w1 shape mismatch")
        if w2.size(0) != d:
            raise ValueError("w2 shape mismatch")
        u, s, vh = torch.linalg.svd(w2.to(dtype=torch.float32), full_matrices=False)
        k = int(min(self.k, u.size(1), s.size(0), vh.size(0)))
        uk = u[:, :k]
        sk = s[:k]
        vhk = vh[:k, :]
        self.U.zero_()
        self.U[:, :k].copy_(uk * sk.unsqueeze(0))
        r0 = vhk.matmul(w1.to(dtype=torch.float32))
        self.router.weight.zero_()
        self.router.weight[:k, :].copy_(r0)
        self.router.weight[self.k : self.k + k, :].copy_(-r0)
        self.bias.zero_()
        self.log_tau.copy_(torch.tensor(0.0))
        self.z_scale.fill_(0.0)
        self.eta_logit.copy_(torch.tensor(3.0))
        self.out_scale.copy_(torch.tensor(1.0))


class RSFFNBlock(nn.Module):
    def __init__(self, block: nn.Module, d_model: int, k: int, replace: bool):
        super().__init__()
        self.ln_1 = block.ln_1
        self.attn = block.attn
        self.ln_2 = block.ln_2
        self.mlp = block.mlp
        self.replace = bool(replace)
        self.rsffn = RSFFNPrototype(d_model=d_model, k=k)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        u = self.ln_1(x)
        attn_out_full = self.attn(u, attention_mask=attention_mask, use_cache=False)
        attn_out = attn_out_full[0] if isinstance(attn_out_full, (tuple, list)) else attn_out_full
        h1 = x + attn_out
        h = self.ln_2(h1)
        if self.replace:
            f = self.rsffn(h)
        else:
            f = self.mlp(h)
        return h1 + f

    @torch.no_grad()
    def init_rsffn_from_block(self, converter, device: torch.device) -> None:
        w1, w2 = _extract_gpt2_ffn_weights(self)
        w1 = w1.to(device=device, dtype=torch.float32)
        w2 = w2.to(device=device, dtype=torch.float32)
        self.rsffn.init_from_gpt2_ffn(w1=w1, w2=w2)


class RSFFNCausalLM(nn.Module):
    def __init__(self, original_model, k: int, replace_last_n: int = 1):
        super().__init__()
        self.wte = original_model.transformer.wte
        self.wpe = original_model.transformer.wpe
        self.drop = original_model.transformer.drop
        self.ln_f = original_model.transformer.ln_f
        self.lm_head = original_model.lm_head
        blocks = list(original_model.transformer.h)
        n = len(blocks)
        last_n = int(max(0, min(n, int(replace_last_n))))
        cut = n - last_n
        d_model = int(original_model.config.n_embd)
        self.blocks = nn.ModuleList(
            [RSFFNBlock(b, d_model=d_model, k=int(k), replace=(i >= cut)) for i, b in enumerate(blocks)]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = input_ids.device
        b, t = input_ids.shape
        pos = torch.arange(t, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(b, t)
        h = self.wte(input_ids) + self.wpe(pos)
        h = self.drop(h)
        attn_mask = None
        if attention_mask is not None:
            m = attention_mask.to(device=device)
            if m.dim() == 2:
                attn_mask = m[:, None, None, :].to(dtype=h.dtype)
                attn_mask = (1.0 - attn_mask) * -10000.0
            else:
                attn_mask = m
        for block in self.blocks:
            h = block(h, attention_mask=attn_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

    def forward_with_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        device = input_ids.device
        b, t = input_ids.shape
        pos = torch.arange(t, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(b, t)
        h = self.wte(input_ids) + self.wpe(pos)
        h = self.drop(h)
        attn_mask = None
        if attention_mask is not None:
            m = attention_mask.to(device=device)
            if m.dim() == 2:
                attn_mask = m[:, None, None, :].to(dtype=h.dtype)
                attn_mask = (1.0 - attn_mask) * -10000.0
            else:
                attn_mask = m
        hiddens = []
        for block in self.blocks:
            h = block(h, attention_mask=attn_mask)
            hiddens.append(h)
        h_last = self.ln_f(h)
        logits = self.lm_head(h_last)
        return logits, hiddens, h_last

    def rsffn_parameters(self):
        params = []
        for block in self.blocks:
            if block.replace and block.rsffn is not None:
                params.extend(list(block.rsffn.parameters()))
        return params

    def all_rsffn_parameters(self):
        params = []
        for block in self.blocks:
            if block.rsffn is not None:
                params.extend(list(block.rsffn.parameters()))
        return params

    @torch.no_grad()
    def set_replace_last_n(self, last_n: int) -> None:
        n = len(self.blocks)
        last_n = int(max(0, min(n, int(last_n))))
        cut = n - last_n
        for i, block in enumerate(self.blocks):
            block.replace = bool(i >= cut)

    @torch.no_grad()
    def set_trainable_last_n(self, last_n: int) -> None:
        n = len(self.blocks)
        last_n = int(max(0, min(n, int(last_n))))
        cut = n - last_n
        for i, block in enumerate(self.blocks):
            train = bool(i >= cut)
            for p in block.rsffn.parameters():
                p.requires_grad = train

    @torch.no_grad()
    def generate_sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(int(max_new_tokens)):
            logits = self.forward(generated)
            next_id = sample_next_token(
                logits[:, -1, :],
                generated_ids=generated,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )
            generated = torch.cat([generated, next_id], dim=1)
            if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
                break
        return generated


@torch.no_grad()
def build_gpt2_rsffn_model(original_model, device, k: int = 64, replace_last_n: int = 3):
    if isinstance(device, str):
        device = torch.device(device)
    student = RSFFNCausalLM(original_model, k=int(k), replace_last_n=int(replace_last_n)).to(device)
    for p in student.parameters():
        p.requires_grad = False
    original_model.eval()
    student.eval()
    for block in student.blocks:
        block.ln_1.eval()
        block.ln_2.eval()
        block.attn.eval()
        block.mlp.eval()
        block.init_rsffn_from_block(None, device=device)
    student.set_replace_last_n(int(replace_last_n))
    student.set_trainable_last_n(int(replace_last_n))
    if len(student.rsffn_parameters()) == 0:
        raise ValueError("No RSFFN parameters to train (replace_last_n is 0)")
    return student


class RSFFNTrainer:
    def __init__(self, original_model, student_lm: RSFFNCausalLM, tokenizer, device, lr: float):
        if isinstance(device, str):
            device = torch.device(device)
        self.original_model = original_model
        self.student_lm = student_lm
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = _use_amp(device)
        params = [p for p in self.student_lm.all_rsffn_parameters() if isinstance(p, torch.nn.Parameter)]
        self.optimizer = torch.optim.AdamW(params, lr=float(lr))
        self.vocab_size = tokenizer.vocab_size
        self._corpus_ids = None
        self._corpus_ptr = 0
        data_mode = str(os.environ.get("RSFFN_DATA", "random")).strip().lower()
        if data_mode in ("corpus", "mixed"):
            try:
                from reality_stone.utils.text_corpus import load_corpus, chunk_text
                roots_s = os.environ.get("RSFFN_CORPUS_ROOTS", "docs,README.md,rules.mdc")
                roots = [r.strip() for r in roots_s.split(",") if r.strip()]
                max_docs = int(os.environ.get("RSFFN_CORPUS_MAX_DOCS", "200"))
                chunk_chars = int(os.environ.get("RSFFN_CORPUS_CHUNK_CHARS", "4000"))
                overlap_chars = int(os.environ.get("RSFFN_CORPUS_OVERLAP_CHARS", "400"))
                docs = load_corpus(roots, max_docs=max_docs)
                chunks = []
                for d in docs:
                    for ch in chunk_text(d.text, chunk_chars=chunk_chars, overlap_chars=overlap_chars):
                        t = ch.strip().replace("\n", " ")
                        if len(t) >= 64:
                            chunks.append(t)
                        if len(chunks) >= max_docs * 4:
                            break
                    if len(chunks) >= max_docs * 4:
                        break
                if chunks:
                    ids_list = []
                    seq_len = int(os.environ.get("RSFFN_SEQ_LEN", "32"))
                    for t in chunks[: max(32, min(len(chunks), max_docs * 2))]:
                        enc = tokenizer(
                            t,
                            return_tensors="pt",
                            truncation=True,
                            max_length=seq_len,
                            padding="max_length",
                        )
                        ids_list.append(
                            (
                                enc["input_ids"][0].to("cpu"),
                                enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0].to("cpu"),
                            )
                        )
                    if ids_list:
                        self._corpus_ids = ids_list
            except Exception:
                self._corpus_ids = None

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = float(lr)

    def _sample_ids(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_mode = str(os.environ.get("RSFFN_DATA", "random")).strip().lower()
        use_corpus = self._corpus_ids is not None and data_mode in ("corpus", "mixed")
        if use_corpus and data_mode == "mixed":
            u = torch.rand((), device=self.device).item()
            p = float(os.environ.get("RSFFN_P_CORPUS", "0.7"))
            use_corpus = u < max(0.0, min(1.0, p))
        if use_corpus:
            bs = int(batch_size)
            buf = []
            n = len(self._corpus_ids)
            for _ in range(bs):
                idx = self._corpus_ptr % n
                self._corpus_ptr += 1
                buf.append(self._corpus_ids[idx])
            input_ids = torch.stack([x[0] for x in buf], dim=0).to(device=self.device)
            attention_mask = torch.stack([x[1] for x in buf], dim=0).to(device=self.device)
            if input_ids.size(1) != int(seq_len):
                input_ids = input_ids[:, : int(seq_len)]
                attention_mask = attention_mask[:, : int(seq_len)]
        else:
            input_ids = torch.randint(0, self.vocab_size, (int(batch_size), int(seq_len)), device=self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
        return input_ids, attention_mask

    def _to_attn_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        m = attention_mask.to(device=self.device)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.dim() == 3 and m.size(1) == 1:
            m = m[:, 0, :]
        if m.dim() == 2:
            am = m[:, None, None, :].to(dtype=dtype)
            return (1.0 - am) * -10000.0
        return m

    def _masked_mse(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(device=a.device, dtype=a.dtype)
        while m.dim() < a.dim():
            m = m.unsqueeze(-1)
        diff2 = (a - b) ** 2
        num = (diff2 * m).sum()
        den = m.sum().clamp_min(1.0)
        return num / den

    def train_ffn_only(self, steps: int, batch_size: int, seq_len: int) -> None:
        self.original_model.eval()
        self.student_lm.eval()
        wte = self.original_model.transformer.wte
        wpe = self.original_model.transformer.wpe
        pos = torch.arange(int(seq_len), dtype=torch.long, device=self.device)
        pos_emb = wpe(pos)
        if _tqdm is not None:
            iterator = _tqdm(range(int(steps)), desc="[rsffn_pipeline_ffn]", leave=False)
        else:
            iterator = range(int(steps))
        for _ in iterator:
            input_ids, attention_mask = self._sample_ids(batch_size=batch_size, seq_len=seq_len)
            with torch.no_grad():
                x = wte(input_ids) + pos_emb
            loss = x.new_tensor(0.0)
            attn_mask = self._to_attn_mask(attention_mask, dtype=x.dtype)
            for i, block_s in enumerate(self.student_lm.blocks):
                block_t = self.original_model.transformer.h[i]
                u = block_t.ln_1(x)
                attn_out_full = block_t.attn(u, attention_mask=attn_mask, use_cache=False)
                attn_out = attn_out_full[0] if isinstance(attn_out_full, (tuple, list)) else attn_out_full
                h1 = x + attn_out
                h = block_t.ln_2(h1).detach()
                with torch.no_grad():
                    with autocast(enabled=self.use_amp):
                        f_teacher = block_t.mlp(h)
                    f_teacher = f_teacher.to(dtype=h.dtype)
                if block_s.replace:
                    f_student = block_s.rsffn(h)
                    loss = loss + self._masked_mse(f_student, f_teacher, attention_mask)
                    x = h1 + f_student.detach()
                else:
                    x = h1 + f_teacher.detach()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_lm.all_rsffn_parameters(), 1.0)
            self.optimizer.step()
            if _tqdm is not None:
                iterator.set_postfix(loss=float(loss.item()))

    def train_e2e(
        self,
        steps: int,
        batch_size: int,
        seq_len: int,
        logits_mse_weight: float,
        kl_weight: float,
        kl_temperature: float,
        hidden_last_weight: float,
        hidden_layers_weight: float,
    ) -> None:
        self.original_model.eval()
        self.student_lm.eval()
        if _tqdm is not None:
            iterator = _tqdm(range(int(steps)), desc="[rsffn_pipeline_e2e]", leave=False)
        else:
            iterator = range(int(steps))
        for _ in iterator:
            input_ids, attention_mask = self._sample_ids(batch_size=batch_size, seq_len=seq_len)
            with torch.no_grad():
                with autocast_v2(device_type=self.device.type, enabled=self.use_amp):
                    t_out = self.original_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                t_logits = t_out.logits.float()
                t_last = t_out.hidden_states[-1].float()
                t_layers = [h.float() for h in t_out.hidden_states[1:]]
            with autocast_v2(device_type=self.device.type, enabled=self.use_amp):
                s_logits, s_layers, s_last = self.student_lm.forward_with_hiddens(input_ids, attention_mask=attention_mask)
                s_logits = s_logits.float()
                s_last = s_last.float()
                loss = s_logits.new_tensor(0.0)
                if float(logits_mse_weight) > 0.0:
                    loss = loss + float(logits_mse_weight) * self._masked_mse(s_logits, t_logits, attention_mask)
                if float(hidden_last_weight) > 0.0:
                    loss = loss + float(hidden_last_weight) * self._masked_mse(s_last, t_last, attention_mask)
                if float(hidden_layers_weight) > 0.0:
                    n_layers = min(len(s_layers), len(t_layers))
                    layer_loss = s_logits.new_tensor(0.0)
                    for i in range(n_layers):
                        if not bool(self.student_lm.blocks[i].replace):
                            continue
                        layer_loss = layer_loss + self._masked_mse(s_layers[i].float(), t_layers[i], attention_mask)
                    loss = loss + float(hidden_layers_weight) * layer_loss
                if float(kl_weight) > 0.0 and float(kl_temperature) > 0.0:
                    T = float(kl_temperature)
                    t_logp = F.log_softmax(t_logits / T, dim=-1)
                    s_logp = F.log_softmax(s_logits / T, dim=-1)
                    t_p = torch.exp(t_logp)
                    kl_tok = (t_p * (t_logp - s_logp)).sum(dim=-1)
                    m = attention_mask.to(device=kl_tok.device, dtype=kl_tok.dtype)
                    kl = (kl_tok * m).sum() / m.sum().clamp_min(1.0)
                    kl = kl * (T * T)
                    loss = loss + float(kl_weight) * kl
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_lm.all_rsffn_parameters(), 1.0)
            self.optimizer.step()
            if _tqdm is not None:
                iterator.set_postfix(loss=float(loss.item()))


def distill_gpt2_ffn_only(
    original_model,
    student_lm: RSFFNCausalLM,
    tokenizer,
    device,
    steps: int = 200,
    batch_size: int = 4,
    seq_len: int = 32,
    lr: float = 1e-3,
):
    if isinstance(device, str):
        device = torch.device(device)
    original_model.eval()
    student_lm.eval()
    params = student_lm.rsffn_parameters()
    optimizer = torch.optim.AdamW(params, lr=lr)
    vocab_size = tokenizer.vocab_size
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos_emb = wpe(pos)
    use_amp = _use_amp(device)
    if _tqdm is not None:
        iterator = _tqdm(range(steps), desc="[distill_rsffn_ffn_only]", leave=False)
    else:
        iterator = range(steps)
    for _step in iterator:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            x = wte(input_ids) + pos_emb
        loss = 0.0
        for i, block_s in enumerate(student_lm.blocks):
            block_t = original_model.transformer.h[i]
            u = block_t.ln_1(x)
            attn_out_full = block_t.attn(u, attention_mask=None, use_cache=False)
            attn_out = attn_out_full[0] if isinstance(attn_out_full, (tuple, list)) else attn_out_full
            h1 = x + attn_out
            h = block_t.ln_2(h1).detach()
            with torch.no_grad():
                with autocast(enabled=use_amp):
                    f_teacher = block_t.mlp(h)
                f_teacher = f_teacher.to(dtype=h.dtype)
            if block_s.replace and block_s.rsffn is not None:
                f_student = block_s.rsffn(h)
                loss = loss + F.mse_loss(f_student, f_teacher)
                x = h1 + f_student.detach()
            else:
                x = h1 + f_teacher.detach()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if _tqdm is not None:
            iterator.set_postfix(loss=float(loss.item()))
    student_lm.eval()
    return student_lm


def distill_gpt2_rsffn_e2e(
    original_model,
    student_lm: RSFFNCausalLM,
    tokenizer,
    device,
    steps: int = 200,
    batch_size: int = 2,
    seq_len: int = 32,
    lr: float = 5e-4,
    logits_mse_weight: float = 1.0,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    hidden_last_weight: float = 0.2,
    hidden_layers_weight: float = 0.0,
):
    if isinstance(device, str):
        device = torch.device(device)
    original_model.eval()
    student_lm.eval()
    params = student_lm.rsffn_parameters()
    optimizer = torch.optim.AdamW(params, lr=lr)
    vocab_size = tokenizer.vocab_size
    use_amp = _use_amp(device)
    if _tqdm is not None:
        iterator = _tqdm(range(steps), desc="[distill_rsffn_e2e]", leave=False)
    else:
        iterator = range(steps)
    for _step in iterator:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            with autocast_v2(device_type=device.type, enabled=use_amp):
                t_out = original_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            t_logits = t_out.logits.float()
            t_last = t_out.hidden_states[-1].float()
            t_layers = [h.float() for h in t_out.hidden_states[1:]]
        with autocast_v2(device_type=device.type, enabled=use_amp):
            s_logits, _s_hiddens, s_last = student_lm.forward_with_hiddens(input_ids, attention_mask=attention_mask)
            s_logits = s_logits.float()
            s_last = s_last.float()
            loss = s_logits.new_tensor(0.0)
            if logits_mse_weight > 0.0:
                loss = loss + float(logits_mse_weight) * F.mse_loss(s_logits, t_logits)
            if hidden_last_weight > 0.0:
                loss = loss + float(hidden_last_weight) * F.mse_loss(s_last, t_last)
            if hidden_layers_weight > 0.0:
                n_layers = min(len(_s_hiddens), len(t_layers))
                layer_loss = s_logits.new_tensor(0.0)
                for i in range(n_layers):
                    if not bool(student_lm.blocks[i].replace):
                        continue
                    layer_loss = layer_loss + F.mse_loss(_s_hiddens[i].float(), t_layers[i])
                loss = loss + float(hidden_layers_weight) * layer_loss
            if kl_weight > 0.0 and kl_temperature > 0.0:
                T = float(kl_temperature)
                t_logp = F.log_softmax(t_logits / T, dim=-1)
                s_logp = F.log_softmax(s_logits / T, dim=-1)
                t_p = torch.exp(t_logp)
                kl = (t_p * (t_logp - s_logp)).sum(dim=-1).mean() * (T * T)
                loss = loss + float(kl_weight) * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if _tqdm is not None:
            iterator.set_postfix(loss=float(loss.item()))
    student_lm.eval()
    return student_lm


def distill_gpt2_rsffn_curriculum(
    original_model,
    student_lm: RSFFNCausalLM,
    tokenizer,
    device,
    stages: list[int],
    stage_steps: int = 100,
    batch_size: int = 2,
    seq_len: int = 32,
    lr: float = 5e-4,
    logits_mse_weight: float = 1.0,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    hidden_last_weight: float = 0.2,
    hidden_layers_weight: float = 0.0,
):
    for last_n in stages:
        student_lm.set_replace_last_n(int(last_n))
        student_lm.set_trainable_last_n(int(last_n))
        distill_gpt2_rsffn_e2e(
            original_model,
            student_lm,
            tokenizer,
            device,
            steps=int(stage_steps),
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            lr=float(lr),
            logits_mse_weight=float(logits_mse_weight),
            kl_weight=float(kl_weight),
            kl_temperature=float(kl_temperature),
            hidden_last_weight=float(hidden_last_weight),
            hidden_layers_weight=float(hidden_layers_weight),
        )
    return student_lm


def distill_gpt2_rsffn_curriculum_two_phase(
    original_model,
    student_lm: RSFFNCausalLM,
    tokenizer,
    device,
    stages: list[int],
    stage_ffn_steps: int = 200,
    stage_e2e_steps: int = 100,
    batch_size: int = 2,
    seq_len: int = 32,
    lr: float = 5e-4,
    logits_mse_weight: float = 1.0,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    hidden_last_weight: float = 0.2,
    hidden_layers_weight: float = 0.0,
    ffn_lr: float | None = None,
):
    if ffn_lr is None:
        ffn_lr = lr
    for last_n in stages:
        student_lm.set_replace_last_n(int(last_n))
        student_lm.set_trainable_last_n(int(last_n))
        if int(stage_ffn_steps) > 0:
            distill_gpt2_ffn_only(
                original_model,
                student_lm,
                tokenizer,
                device,
                steps=int(stage_ffn_steps),
                batch_size=int(batch_size),
                seq_len=int(seq_len),
                lr=float(ffn_lr),
            )
        if int(stage_e2e_steps) > 0:
            distill_gpt2_rsffn_e2e(
                original_model,
                student_lm,
                tokenizer,
                device,
                steps=int(stage_e2e_steps),
                batch_size=int(batch_size),
                seq_len=int(seq_len),
                lr=float(lr),
                logits_mse_weight=float(logits_mse_weight),
                kl_weight=float(kl_weight),
                kl_temperature=float(kl_temperature),
                hidden_last_weight=float(hidden_last_weight),
                hidden_layers_weight=float(hidden_layers_weight),
            )
    return student_lm


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
