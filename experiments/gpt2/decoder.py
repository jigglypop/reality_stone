import time
import string
import numpy as np
import torch
from reality_stone._rust import PyHumanDecoder
from reality_stone.utils.sampling import sample_next_token

_SKELETON_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "or",
    "if",
    "then",
    "else",
}
_RELATION_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "because",
    "since",
    "therefore",
    "leads",
    "makes",
    "causes",
}

def _sample_next_token(logits, generated, repetition_penalty=1.2, temperature=1.0, top_k=50, top_p=0.95):
    step_logits = logits[:, -1, :]
    next_id = sample_next_token(
        step_logits,
        generated_ids=generated,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )
    return next_id

def _classify_token(token):
    plain = token.replace("Ġ", "").strip()
    if not plain:
        return "s"
    lower = plain.lower()
    if all(ch in string.punctuation for ch in plain):
        return "s"
    if lower in _SKELETON_WORDS:
        return "s"
    if lower in _RELATION_WORDS or lower.endswith("ing") or lower.endswith("ed"):
        return "r"
    return "o"


def build_human_stage_sets(tokenizer):
    skeleton = []
    relation = []
    objects = []
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        group = _classify_token(token)
        if group == "s":
            skeleton.append(token_id)
        elif group == "r":
            relation.append(token_id)
        else:
            objects.append(token_id)
    if not skeleton and tokenizer.eos_token_id is not None:
        skeleton.append(tokenizer.eos_token_id)
    if not relation and skeleton:
        relation.extend(skeleton[:8])
    if not objects:
        objects.extend(relation or skeleton)
    return skeleton, relation, objects


def build_stage_sets_tensors(tokenizer, device):
    skeleton, relation, objects = build_human_stage_sets(tokenizer)
    skel_ids = torch.tensor(skeleton, dtype=torch.long, device=device) if skeleton else torch.empty(0, dtype=torch.long, device=device)
    rel_ids = torch.tensor(relation, dtype=torch.long, device=device) if relation else torch.empty(0, dtype=torch.long, device=device)
    obj_ids = torch.tensor(objects, dtype=torch.long, device=device) if objects else torch.empty(0, dtype=torch.long, device=device)
    return skel_ids, rel_ids, obj_ids


def build_stage_bias(tokenizer, device, skel_bias: float = -0.4, obj_bias: float = 0.4):
    skeleton, relation, objects = build_human_stage_sets(tokenizer)
    vocab_size = tokenizer.vocab_size
    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    if skeleton:
        ids = torch.tensor(skeleton, dtype=torch.long, device=device)
        bias[ids] += skel_bias
    if objects:
        ids = torch.tensor(objects, dtype=torch.long, device=device)
        bias[ids] += obj_bias
    return bias


def build_human_decoder(tokenizer, model, curvature=1e-3):
    embeddings = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float32)
    skeleton, relation, objects = build_human_stage_sets(tokenizer)
    decoder = PyHumanDecoder(
        embeddings,
        skeleton,
        relation,
        objects,
        alpha_logit=1.0,
        alpha_cos=0.45,
        beta_logit=1.0,
        beta_cos=0.8,
        beta_geo=0.35,
        curvature=curvature,
    )
    return decoder


def _human_decode_token(human_decoder, logits, relation_ctx, object_ctx, device, topk_relation=8, topk_object=24):
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    rel_np = relation_ctx.detach().cpu().numpy().astype(np.float32)
    obj_np = object_ctx.detach().cpu().numpy().astype(np.float32)
    token_ids = human_decoder.decode(logits_np, rel_np, obj_np, topk_relation, topk_object)
    if not token_ids:
        raise RuntimeError("human decoder returned empty selection")
    next_id = int(token_ids[0])
    return torch.tensor([[next_id]], dtype=torch.long, device=device)


def rsulf_generate_text(
    original_model,
    rs_model_stack,
    tokenizer,
    device,
    text_prompt,
    max_tokens=30,
    top_k_original=0,
    decoder=None,
    human_decoder=None,
    human_cfg=None,
    syntax_head=None,
    adapter=None,
    rank=None,
    pfc_curvature=1e-4,
    pfc_window=4,
    entity_memory: int = 32,
    entity_beta: float = 0.25,
    entity_temp: float = 0.35,
    entity_min_sim: float = 0.35,
    vocab_ascii_only: bool = True,
    teacher_guidance: float = 0.0,
    teacher_topk_mask: int = 0,
    entity_warmup: int = 8,
    entity_beta_max: float | None = None,
    entity_gate_sigma: float = 0.08,
    entity_ctx_determiner: bool = True,
    entity_ctx_pronoun: bool = True,
    entity_ctx_prep: bool = True,
):
    # PFC-only mode (commanded): generation is strictly
    # RS-ULF forward -> PFC(last token) -> (decoder if provided else ln_f+lm_head).
    # All other experimental modules (syntax/adapter/human-stage) are ignored.
    human_decoder = None
    human_cfg = None
    syntax_head = None
    adapter = None

    _PRONOUN_TOKENS = {
        "he","she","it","they","him","her","them","his","hers","their","theirs",
        "this","that","these","those","who","whom","which",
    }
    _DETERMINER_TOKENS = {"the","a","an","my","your","our","their","its","his","her"}
    _PREP_TOKENS = {
        "to","of","in","on","for","with","at","by","from","into","onto","over","under","about","as",
    }

    def _clean_tok(t: str) -> str:
        return t.replace("Ġ", " ").strip()

    def _is_safe_ascii_token(t: str) -> bool:
        # allow basic ASCII text + whitespace; disallow weird unicode that causes "��/龍/к" spam
        s = t.replace("Ġ", " ")
        for ch in s:
            o = ord(ch)
            if ch == "\n" or ch == "\t":
                continue
            if o < 32 or o > 126:
                return False
        return True

    def _is_entity_like(t: str) -> bool:
        s = _clean_tok(t)
        if not s:
            return False
        low = s.lower()
        if low in {"i", "the", "a", "an"}:
            return False
        if vocab_ascii_only and not _is_safe_ascii_token(t):
            return False
        # Prefer word-boundary tokens for "proper noun" anchors
        # GPT2 BPE typically uses "Ġ" to indicate a preceding space.
        if not t.startswith("Ġ") and not any(ch.isdigit() for ch in s):
            return False
        if any(ch.isdigit() for ch in s):
            return True
        # avoid pure punctuation
        if all(ch in string.punctuation for ch in s):
            return False
        # capitalized wordpiece
        return len(s) >= 2 and s[0].isupper() and any(ch.islower() for ch in s[1:])

    def _is_binding_ctx(prev_tok: str) -> bool:
        s = _clean_tok(prev_tok).lower()
        ok = False
        if entity_ctx_pronoun and s in _PRONOUN_TOKENS:
            ok = True
        if entity_ctx_determiner and s in _DETERMINER_TOKENS:
            ok = True
        if entity_ctx_prep and s in _PREP_TOKENS:
            ok = True
        return ok

    def _entity_retrieve(
        h_t: torch.Tensor,
        mem: list[torch.Tensor],
        beta: float,
        temp: float,
        min_sim: float,
        sigma: float,
        warmup_scale: float,
    ) -> torch.Tensor:
        if not mem:
            return h_t
        keys = torch.stack(mem, dim=0)
        # Normalize key shape to (M, D)
        if keys.dim() == 3 and keys.size(1) == 1:
            keys = keys[:, 0, :]
        elif keys.dim() == 1:
            keys = keys.unsqueeze(0)
        keys = keys.to(device=h_t.device, dtype=h_t.dtype)  # (M, D)
        h = h_t  # (B,D)
        # cosine sim
        h_n = h / (h.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        k_n = keys / (keys.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        sim = torch.matmul(h_n, k_n.transpose(0, 1))  # (B,M)
        max_sim = sim.max(dim=-1, keepdim=True).values  # (B,1)
        # Smooth confidence gate in [0,1] around min_sim
        s = float(max(1e-6, sigma))
        gate = torch.sigmoid((max_sim - float(min_sim)) / s)
        gate = gate * float(max(0.0, warmup_scale))
        if gate.max().item() < 1e-4:
            return h_t
        w = torch.softmax(sim / max(float(temp), 1e-6), dim=-1)
        retrieve = torch.matmul(w, keys)  # (B,D)
        return h + (float(beta) * gate) * retrieve

    def _get_last_ffn_mats(model_stack):
        # RSULFModel has `.layers` = ModuleList[RSULFLayerCUDA]
        if hasattr(model_stack, "layers") and len(getattr(model_stack, "layers")) > 0:
            last = model_stack.layers[-1]
            v1 = getattr(last, "ffn_v1", None)
            u2 = getattr(last, "ffn_u2", None)
            return v1, u2
        # Fallback: RSULFWrapperCUDA holds `.rsulf`
        if hasattr(model_stack, "wrappers") and len(getattr(model_stack, "wrappers")) > 0:
            w = model_stack.wrappers[-1]
            inner = getattr(w, "rsulf", None)
            if inner is not None:
                v1 = getattr(inner, "ffn_v1", None)
                u2 = getattr(inner, "ffn_u2", None)
                return v1, u2
        return None, None

    def _pfc_tail(h_seq, v1, u2, c, w):
        if not c:
            return h_seq
        if h_seq.size(1) < 2:
            return h_seq
        c = float(c)
        if c == 0.0:
            return h_seq
        w = int(max(1, w))
        w = min(w, int(h_seq.size(1) - 1))
        r = int(v1.shape[1])
        if r <= 0:
            return h_seq
        h_seq = h_seq.clone()
        v1_f = v1.float()
        u2_f = u2.float()
        for j in range(w):
            idx = -1 - j
            h_t = h_seq[:, idx, :].float()
            h_prev = h_seq[:, idx - 1, :].float()
            v_t = h_t - h_prev
            vv1 = v_t @ v1_f
            hu2 = h_t @ u2_f
            corr = (vv1 * hu2) @ v1_f.T
            corr = corr * (c / float(r))
            hh = (h_t * h_t).mean(dim=-1, keepdim=True).clamp(min=1e-8)
            ch = (corr * h_t).mean(dim=-1, keepdim=True)
            corr = corr - (ch / hh) * h_t
            h_rms = (h_t * h_t).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            c_rms = (corr * corr).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            scale = (0.02 * h_rms / c_rms).clamp(max=1.0)
            h_seq[:, idx, :] = (h_t + corr * scale).to(dtype=h_seq.dtype)
        h_seq = h_seq.clone()
        return h_seq
    curr_ids = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    lm_head = original_model.lm_head
    ln_f = original_model.transformer.ln_f
    generated = curr_ids
    start_gen = time.time()
    v1_last, u2_last = _get_last_ffn_mats(rs_model_stack)
    entity_mem: list[torch.Tensor] = []

    # Precompute a "safe token" mask once (CPU -> GPU) to suppress garbage tokens.
    safe_mask = None
    if vocab_ascii_only:
        safe = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        for tid in range(tokenizer.vocab_size):
            tok = tokenizer.convert_ids_to_tokens(tid)
            if _is_safe_ascii_token(tok):
                safe[tid] = True
        # Always allow EOS
        if tokenizer.eos_token_id is not None:
            safe[int(tokenizer.eos_token_id)] = True
        safe_mask = safe.to(device=device)

    # Teacher-guided decoding state (logits prior + KV cache)
    teacher_past = None
    teacher_next_logits = None
    tg = float(teacher_guidance)
    tkm = int(max(0, teacher_topk_mask))
    if tg > 0.0 or tkm > 0:
        with torch.no_grad():
            t_out = original_model(input_ids=generated, use_cache=True)
            teacher_past = t_out.past_key_values
            teacher_next_logits = t_out.logits[:, -1, :].detach()

    for step_i in range(max_tokens):
        seq_len = generated.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = wte(generated)
            pos_emb = wpe(pos)
            x = tok_emb + pos_emb
            # RS-ULF core forward (no hybrid/original paths)
            h = x
            for wrapper in rs_model_stack.wrappers:
                h = wrapper(h)

            # PFC last-token correction only
            if pfc_curvature and v1_last is not None and u2_last is not None:
                h = _pfc_tail(h, v1_last, u2_last, pfc_curvature, pfc_window)

            # IMPORTANT: if you want readable text, pass a decoder that maps RS manifold
            # coordinates back to the teacher's hidden space.
            if decoder is not None:
                h_token = h[:, -1, :]
                h_np = h_token.detach().cpu().numpy().astype(np.float32)
                hidden_np = decoder.forward(h_np)
                hidden_t = torch.from_numpy(hidden_np).to(device=device)
                # entity anchor memory in teacher hidden space
                prev_tok = tokenizer.convert_ids_to_tokens(int(generated[0, -1].item())) if generated.size(1) > 0 else ""
                if entity_memory > 0 and entity_beta and _is_binding_ctx(prev_tok):
                    warm = int(max(0, entity_warmup))
                    warm_scale = 1.0
                    if warm > 0:
                        warm_scale = min(1.0, float(step_i + 1) / float(warm))
                    beta_eff = float(entity_beta)
                    if entity_beta_max is not None:
                        bmax = float(entity_beta_max)
                        beta_eff = min(bmax, beta_eff + (bmax - beta_eff) * warm_scale)
                    hidden_t = _entity_retrieve(
                        hidden_t,
                        entity_mem,
                        beta=beta_eff,
                        temp=entity_temp,
                        min_sim=entity_min_sim,
                        sigma=entity_gate_sigma,
                        warmup_scale=warm_scale,
                    )
                hidden = hidden_t.unsqueeze(1)
                logits = lm_head(ln_f(hidden))
            else:
                logits = lm_head(ln_f(h))
            logits_step = logits[:, -1, :]

            # Teacher-guided logit blending / masking (prevents rare junk tokens from dominating).
            if (tg > 0.0 or tkm > 0) and teacher_next_logits is not None:
                t_logits = teacher_next_logits.to(dtype=logits_step.dtype, device=logits_step.device)
                if tg > 0.0:
                    logits_step = (1.0 - tg) * logits_step + tg * t_logits
                if tkm > 0 and tkm < t_logits.size(-1):
                    topk_ids = torch.topk(t_logits, k=tkm, dim=-1).indices  # (B,tkm)
                    mask = torch.ones_like(logits_step, dtype=torch.bool)
                    mask.scatter_(1, topk_ids, False)
                    logits_step = logits_step.masked_fill(mask, -float("inf"))

            if safe_mask is not None:
                # mask out unsafe tokens
                logits_step = logits_step.masked_fill(~safe_mask, -float("inf"))
                # keep logits tensor shape consistent for sampler (expects logits[0,-1,:])
                logits = logits.clone()
                logits[:, -1, :] = logits_step
            else:
                logits = logits.clone()
                logits[:, -1, :] = logits_step
            next_token = _sample_next_token(
                logits,
                generated,
                repetition_penalty=1.2,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
            )
            generated = torch.cat([generated, next_token], dim=1)
            # update entity memory after generating token
            if decoder is not None and entity_memory > 0:
                tok_str = tokenizer.convert_ids_to_tokens(int(next_token.item()))
                if _is_entity_like(tok_str):
                    # store teacher-space anchor(s) as (D,)
                    ht = hidden_t.detach().to(device=device, dtype=torch.float32)
                    if ht.dim() == 2:
                        # batch may be >1, store all rows
                        for bi in range(ht.size(0)):
                            entity_mem.append(ht[bi])
                    else:
                        entity_mem.append(ht.reshape(-1))
                    if len(entity_mem) > int(entity_memory):
                        entity_mem = entity_mem[-int(entity_memory):]

            # advance teacher cache (one-step) to keep teacher_next_logits aligned
            if (tg > 0.0 or tkm > 0) and teacher_past is not None:
                with torch.no_grad():
                    t_out = original_model(input_ids=next_token, past_key_values=teacher_past, use_cache=True)
                    teacher_past = t_out.past_key_values
                    teacher_next_logits = t_out.logits[:, -1, :].detach()
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - start_gen
    return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time


def rsulf_generate_text_pure(rs_lm, tokenizer, device, text_prompt, max_tokens=30):
    start = time.time()
    if isinstance(device, str):
        device = torch.device(device)
    input_ids = tokenizer(text_prompt, return_tensors="pt")["input_ids"].to(device)
    rs_lm = rs_lm.to(device)
    rs_lm.eval()
    with torch.no_grad():
        out_ids = rs_lm.generate_sample(
            input_ids=input_ids,
            max_new_tokens=int(max_tokens),
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.15,
            eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        )
    text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    return text, (time.time() - start)