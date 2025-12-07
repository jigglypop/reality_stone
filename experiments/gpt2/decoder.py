import time
import numpy as np
import torch
import torch.nn.functional as F
from reality_stone._rust import laplace_beltrami_matrix, PyRiemannianDecoder


def _collect_decoder_data(
    original_model,
    rs_model,
    tokenizer,
    device,
    num_batches: int = 16,
    batch_size: int = 4,
    seq_len: int = 32,
):
    original_model.eval()
    rs_model.eval()
    vocab_size = tokenizer.vocab_size
    ln_f = original_model.transformer.ln_f
    lm_head = original_model.lm_head
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    h_rs_list = []
    logits_t_list = []
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
            outputs = original_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            teacher_hidden_last = outputs.hidden_states[-1]
            t_last = ln_f(teacher_hidden_last)
            t_logits = lm_head(t_last)
            pos = torch.arange(seq_len, dtype=torch.long, device=device)
            x0 = wte(input_ids) + wpe(pos)
            h_rs = x0
            for wrapper in rs_model.wrappers:
                h_rs = wrapper(h_rs)
            h_rs_list.append(h_rs.reshape(-1, h_rs.size(-1)).cpu())
            logits_t_list.append(t_logits.reshape(-1, t_logits.size(-1)).cpu())
    H_rs = torch.cat(h_rs_list, dim=0).numpy().astype(np.float32)
    L_t = torch.cat(logits_t_list, dim=0).numpy().astype(np.float32)
    return H_rs, L_t


def fit_riemannian_decoder(
    original_model,
    rs_model,
    tokenizer,
    device,
    global_basis: dict,
    target_rank: int,
    num_batches: int = 16,
    batch_size: int = 4,
    seq_len: int = 32,
):
    if global_basis is None or "u" not in global_basis or "rank" not in global_basis:
        return None
    H_rs, L_t = _collect_decoder_data(
        original_model,
        rs_model,
        tokenizer,
        device,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    U = global_basis["u"].astype(np.float32)
    d_model, k_all = U.shape
    k_basis = int(global_basis.get("rank", k_all))
    k_basis = max(1, min(k_basis, k_all))
    U_k = U[:, :k_basis]
    C = H_rs @ U_k
    L_mean = L_t.mean(axis=0, keepdims=True)
    L_centered = L_t - L_mean
    lambda_lb = 1e-2
    try:
        C_float = C.astype(np.float32)
        lb_mat = laplace_beltrami_matrix(C_float, "diagonal", 0.0, 0.5, 1e-6)
        lb_mat = np.asarray(lb_mat, dtype=np.float32)
        A_main = C
        B_main = L_centered
        A_reg = lb_mat @ C
        B_reg = np.zeros_like(L_centered)
        A_aug = np.vstack([A_main, np.sqrt(lambda_lb) * A_reg])
        B_aug = np.vstack([B_main, B_reg])
        W_T, _, _, _ = np.linalg.lstsq(A_aug, B_aug, rcond=None)
    except Exception:
        W_T, _, _, _ = np.linalg.lstsq(C, L_centered, rcond=None)
    W_star = W_T.T.astype(np.float32)
    U_svd, S_svd, Vt_svd = np.linalg.svd(W_star, full_matrices=False)
    max_rank = min(target_rank, U_svd.shape[1], Vt_svd.shape[0])
    r = max_rank
    U_r = U_svd[:, :r]
    S_r = S_svd[:r]
    V_r = Vt_svd[:r, :].T
    sqrt_S = np.sqrt(S_r).astype(np.float32)
    Bt = U_r * sqrt_S[None, :]
    A = sqrt_S[:, None] * V_r.T
    bias = L_mean.reshape(-1).astype(np.float32)
    decoder = PyRiemannianDecoder(
        U_k.astype(np.float32),
        A.astype(np.float32),
        Bt.astype(np.float32),
        bias.astype(np.float32),
    )
    with torch.no_grad():
        H_rs_t = torch.from_numpy(H_rs[:1024]).to(device)
        logits_t_ref = torch.from_numpy(L_t[:1024]).to(device)
        logits_s_np = decoder.forward(H_rs_t.cpu().numpy().astype(np.float32))
        logits_s = torch.from_numpy(logits_s_np).to(device)
        t_flat = logits_t_ref.view(-1, logits_t_ref.size(-1))
        s_flat = logits_s.view(-1, logits_s.size(-1))
        cos = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
        rel = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)
        print(f"[Decoder fit] final_logits (train-sample) cos={cos:.4f}, rel_l2={rel:.4f}")
    return decoder


def _sample_next_token(logits, generated, repetition_penalty=1.2, temperature=1.0, top_k=50, top_p=0.95):
    scores = logits[0, -1, :].clone()
    seen = set(generated[0].tolist())
    for token_id in seen:
        val = scores[token_id]
        if val < 0:
            scores[token_id] = val * repetition_penalty
        else:
            scores[token_id] = val / repetition_penalty
    if temperature != 1.0:
        scores = scores / max(temperature, 1e-8)
    if top_k > 0 and top_k < scores.size(-1):
        _, topk_idx = torch.topk(scores, top_k)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[topk_idx] = False
        scores[mask] = -float("inf")
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > top_p
        if cutoff.any():
            first = int(cutoff.nonzero(as_tuple=False)[0])
            sorted_scores[first + 1 :] = -float("inf")
            scores = torch.full_like(scores, -float("inf")
            )
            scores[sorted_idx] = sorted_scores
    probs = torch.softmax(scores, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.unsqueeze(0)


def rsulf_generate_text(original_model, rs_model_stack, tokenizer, device, text_prompt, max_tokens=30, top_k_original=0, decoder=None):
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
        attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
        with torch.no_grad():
            tok_emb = wte(generated)
            pos_emb = wpe(pos)
            x = tok_emb + pos_emb
            h = x
            for wrapper in rs_model_stack.wrappers:
                if hasattr(wrapper, "original_block") and wrapper.original_block is not None:
                    outputs = wrapper.original_block(h, attention_mask=attention_mask)
                    h = outputs[0] if isinstance(outputs, tuple) else outputs
                elif hasattr(wrapper, "rsulf") and hasattr(wrapper.rsulf, "original_block") and wrapper.rsulf.original_block is not None:
                    outputs = wrapper.rsulf.original_block(h, attention_mask=attention_mask)
                    h = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    h = wrapper(h)
            if decoder is not None:
                h_token = h[:, -1, :]
                h_np = h_token.detach().cpu().numpy().astype(np.float32)
                logits_np = decoder.forward(h_np)
                logits_t = torch.from_numpy(logits_np).to(device=device)
                logits = logits_t.unsqueeze(1)
            else:
                h_last = ln_f(h)
                logits = lm_head(h_last)
            next_token = _sample_next_token(logits, generated)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - start_gen
    return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time
