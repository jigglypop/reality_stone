import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from reality_stone._rust import laplace_beltrami_matrix, PyRiemannianDecoder

try:
    from reality_stone.utils.text_corpus import load_corpus, chunk_text
    _HAS_TEXT_CORPUS = True
except Exception:
    _HAS_TEXT_CORPUS = False

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


def _generate_binary_curriculum_prompts(
    original_model,
    tokenizer,
    device,
    num_batches: int,
    batch_size: int,
    seq_len: int,
):
    original_model.eval()
    vocab_size = tokenizer.vocab_size
    wte = original_model.transformer.wte
    seed_inputs = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), device=device)
    embeds = wte(seed_inputs).detach().requires_grad_(True)
    use_amp = _use_amp(device)
    with autocast(enabled=use_amp):
        outputs = original_model.transformer(inputs_embeds=embeds)
    hidden_last = outputs.last_hidden_state.float()
    activation_norm = hidden_last.norm(dim=-1).mean()
    grad = torch.autograd.grad(activation_norm, embeds, create_graph=False)[0]
    epsilon = 0.1 * embeds.std().item()
    delta = epsilon * torch.sign(grad)
    inputs_high = embeds + delta
    inputs_low = embeds - delta
    print(f"[BACD] Seed Norm: {hidden_last.norm(dim=-1).mean().item():.4f}")
    print(f"[BACD] Gradient Norm: {grad.norm().item():.4f}")
    print(f"[BACD] Perturbation Epsilon: {epsilon:.4f}")
    return inputs_high.detach(), inputs_low.detach()


def _collect_decoder_data(
    original_model,
    rs_model,
    tokenizer,
    device,
    num_batches: int = 64,
    batch_size: int = 4,
    seq_len: int = 32,
    prompt_text: str | None = None,
    data_mode: str = "random",
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    corpus_roots: list[str] | None = None,
    corpus_max_docs: int = 2000,
    corpus_chunk_chars: int = 8000,
    corpus_overlap_chars: int = 1000,
    mix_p_corpus: float = 0.6,
    mix_p_random: float = 0.3,
    mix_p_teacher_sample: float = 0.1,
    prompt_pool: list[str] | None = None,
):
    if isinstance(device, str):
        device = torch.device(device)
    original_model.eval()
    rs_model.eval()
    lm_head = original_model.lm_head
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    h_rs_list = []
    h_t_list = []
    vocab_size = tokenizer.vocab_size
    total_samples = num_batches * batch_size
    start = time.time()
    total_batches = (total_samples + batch_size - 1) // batch_size
    total_tokens = total_samples * seq_len
    indices = range(0, total_samples, batch_size)
    if _tqdm is not None:
        iterator = _tqdm(indices, total=total_batches, desc="[BACD] collect", leave=False)
    else:
        iterator = indices
    use_amp = _use_amp(device)
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos_emb = wpe(pos)

    corpus_chunks: list[str] | None = None
    if data_mode in ("corpus", "mixed") and float(mix_p_corpus) > 0.0:
        if not _HAS_TEXT_CORPUS:
            raise RuntimeError("data_mode='corpus' requires python/reality_stone/utils/text_corpus.py")
        if not corpus_roots:
            corpus_roots = ["docs", "README.md", "rules.mdc"]
        docs = load_corpus(corpus_roots, max_docs=int(corpus_max_docs))
        if len(docs) == 0:
            raise RuntimeError(f"No corpus docs found from roots={corpus_roots}")
        chunks: list[str] = []
        for d in docs:
            for ch in chunk_text(d.text, chunk_chars=int(corpus_chunk_chars), overlap_chars=int(corpus_overlap_chars)):
                t = ch.strip()
                if len(t) >= 64:
                    chunks.append(t)
        if len(chunks) == 0:
            raise RuntimeError("Corpus loaded but produced 0 usable chunks.")
        # Pre-tokenize only what we actually need (avoid tokenizing the whole corpus).
        needed = int(total_samples)
        # small buffer for variety
        cap = min(len(chunks), max(needed, int(batch_size) * 4))
        ids_list: list[torch.Tensor] = []
        for t in chunks[:cap]:
            enc = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=int(seq_len),
                padding="max_length",
            )
            ids_list.append(enc["input_ids"][0].to("cpu"))
        if len(ids_list) == 0:
            raise RuntimeError("Corpus tokenization produced 0 sequences.")
        corpus_chunks = None
        corpus_ids = ids_list
    else:
        corpus_ids = None

    # Mixed-mode prompt pool (if not provided, reuse corpus chunks as prompts)
    if prompt_pool is None:
        prompt_pool = []
        if data_mode in ("corpus", "mixed") and _HAS_TEXT_CORPUS:
            try:
                roots = corpus_roots if corpus_roots else ["docs", "README.md", "rules.mdc"]
                docs_for_prompts = load_corpus(roots, max_docs=50)
                for d in docs_for_prompts:
                    for ch in chunk_text(d.text, chunk_chars=1200, overlap_chars=0):
                        t = ch.strip().replace("\n", " ")
                        if len(t) >= 32:
                            prompt_pool.append(t[:256])
                        if len(prompt_pool) >= 200:
                            break
                    if len(prompt_pool) >= 200:
                        break
            except Exception:
                prompt_pool = []
        if not prompt_pool:
            prompt_pool = ["The", "Once upon a time", "In summary", "Q: ", "Answer: "]

    def _pick_mode() -> str:
        # deterministic-ish mixture by batch index (no RNG dependency)
        p_c = float(max(0.0, mix_p_corpus))
        p_r = float(max(0.0, mix_p_random))
        p_t = float(max(0.0, mix_p_teacher_sample))
        s = p_c + p_r + p_t
        if s <= 0:
            return "random"
        p_c, p_r, p_t = p_c / s, p_r / s, p_t / s
        u = (float(b) * 0.6180339887498949) % 1.0
        if u < p_c:
            return "corpus"
        if u < p_c + p_r:
            return "random"
        return "teacher_sample"

    with torch.no_grad():
        for b, i in enumerate(iterator, 1):
            # Important: each batch is an independent sequence set.
            # RSULF wrappers keep KV/memory (v_mem) for autoregressive generation;
            # if we don't reset here, batches contaminate each other and ruin distillation quality.
            if hasattr(rs_model, "wrappers"):
                for w in rs_model.wrappers:
                    if hasattr(w, "reset_memory"):
                        w.reset_memory()
                    elif hasattr(w, "v_mem"):
                        w.v_mem = None
            mode = data_mode
            if data_mode == "mixed":
                mode = _pick_mode()

            if mode == "teacher_sample":
                # Fast path: use HF generate() with KV-cache instead of per-token transformer forward.
                ptxt = prompt_text if prompt_text else prompt_pool[(b - 1) % len(prompt_pool)]
                prompt_ids = tokenizer.encode(ptxt, return_tensors="pt").to(device)
                prompt_ids = prompt_ids[:, : max(1, min(prompt_ids.size(1), seq_len))]
                prompt_ids = prompt_ids.repeat(batch_size, 1)
                attn_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
                max_new = int(seq_len - prompt_ids.size(1))
                if max_new > 0:
                    gen_ids = original_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                else:
                    gen_ids = prompt_ids
                if gen_ids.size(1) < seq_len:
                    pad = torch.full((batch_size, seq_len - gen_ids.size(1)), int(tokenizer.eos_token_id), device=device, dtype=torch.long)
                    input_ids = torch.cat([gen_ids, pad], dim=1)
                else:
                    input_ids = gen_ids[:, :seq_len]
            elif mode == "corpus":
                if corpus_ids is None or len(corpus_ids) == 0:
                    # Fallback (no corpus prepared): behave like random
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                else:
                    base = (b - 1) * batch_size
                    batch_ids = [corpus_ids[(base + j) % len(corpus_ids)] for j in range(batch_size)]
                    input_ids = torch.stack(batch_ids, dim=0).to(device)
            else:
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            tok_emb = wte(input_ids)
            x = tok_emb + pos_emb
            with autocast(enabled=use_amp):
                outputs = original_model.transformer(input_ids=input_ids, use_cache=False)
                teacher_hidden_last = outputs.last_hidden_state
            h_rs = x
            for wrapper in rs_model.wrappers:
                h_rs = wrapper(h_rs)
            h_rs_list.append(h_rs.reshape(-1, h_rs.size(-1)).cpu())
            h_t_list.append(teacher_hidden_last.reshape(-1, teacher_hidden_last.size(-1)).cpu())
            current_norm = h_rs.norm(dim=-1).mean().item()
            if _tqdm is not None:
                processed_samples = min(i + batch_size, total_samples)
                processed_tokens = processed_samples * seq_len
                iterator.set_postfix(
                    batch=b,
                    total_batches=total_batches,
                    tokens=f"{processed_tokens}/{total_tokens}",
                    norm=current_norm,
                )
    elapsed = time.time() - start
    H_rs = torch.cat(h_rs_list, dim=0).numpy().astype(np.float32)
    H_t = torch.cat(h_t_list, dim=0).numpy().astype(np.float32)
    print(f"[BACD] Collected tokens={H_rs.shape[0]} dim={H_rs.shape[1]} time={elapsed:.2f}s")
    return H_rs, H_t


def fit_riemannian_decoder(
    original_model,
    rs_model,
    tokenizer,
    device,
    global_basis: dict,
    target_rank: int,
    num_batches: int = 64,
    batch_size: int = 4,
    seq_len: int = 32,
    prompt_text: str | None = None,
    data_mode: str = "random",
    corpus_roots: list[str] | None = None,
    corpus_max_docs: int = 2000,
    corpus_chunk_chars: int = 8000,
    corpus_overlap_chars: int = 1000,
    mix_p_corpus: float = 0.6,
    mix_p_random: float = 0.3,
    mix_p_teacher_sample: float = 0.1,
    prompt_pool: list[str] | None = None,
):
    if isinstance(device, str):
        device = torch.device(device)
    H_rs, H_t = _collect_decoder_data(
        original_model,
        rs_model,
        tokenizer,
        device,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_text=prompt_text,
        data_mode=data_mode,
        corpus_roots=corpus_roots,
        corpus_max_docs=corpus_max_docs,
        corpus_chunk_chars=corpus_chunk_chars,
        corpus_overlap_chars=corpus_overlap_chars,
        mix_p_corpus=mix_p_corpus,
        mix_p_random=mix_p_random,
        mix_p_teacher_sample=mix_p_teacher_sample,
        prompt_pool=prompt_pool,
    )
    print(f"[BACD] Start fitting Riemannian Decoder (samples={H_rs.shape[0]})...")
    start_time = time.time()
    if global_basis is not None and "u" in global_basis and "rank" in global_basis:
        U = global_basis["u"].astype(np.float32)
        d_model, k_all = U.shape
        k_basis = int(global_basis.get("rank", k_all))
        k_basis = max(1, min(k_basis, k_all))
        U_k = U[:, :k_basis]
        C = H_rs @ U_k
    else:
        d_model = H_rs.shape[1]
        k_basis = d_model
        U_k = np.eye(d_model, k_basis, dtype=np.float32)
        C = H_rs
    C_float = C.astype(np.float32)
    C_mean = C_float.mean(axis=0, keepdims=True)
    C_centered = C_float - C_mean
    H_mean = H_t.mean(axis=0, keepdims=True)
    H_centered = H_t - H_mean
    lambda_lb = 0.0
    rank_factor = float(H_rs.shape[1]) / float(max(1, int(target_rank)))
    lambda_ridge = 1e-5 * max(1.0, rank_factor)
    if lambda_ridge > 1e-2:
        lambda_ridge = 1e-2
    use_gpu = device.type == "cuda" or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and device.type == "mps"
    )

    try:
        use_lb = float(lambda_lb) > 0.0
        lb_mat_np = None
        if use_lb:
            print(f"[BACD] Computing Laplace-Beltrami matrix (samples={C_float.shape[0]})...")
            lb_done = {"v": False, "start": time.time()}

            def _lb_spinner():
                while not lb_done["v"]:
                    elapsed = time.time() - lb_done["start"]
                    print(f"[BACD] LB running... {elapsed:.1f}s", flush=True)
                    time.sleep(5.0)

            spinner = threading.Thread(target=_lb_spinner, daemon=True)
            spinner.start()
            lb_mat_np = laplace_beltrami_matrix(C_float, "diagonal", 0.0, 0.5, 1e-6)
            lb_done["v"] = True
            spinner.join(timeout=0.1)
            print("[BACD] LB matrix computed.")

        if use_gpu:
            print(f"[BACD] Switching to GPU ({device}) for Solver...")
            C_torch = torch.from_numpy(C_centered).to(device)
            H_centered_torch = torch.from_numpy(H_centered).to(device)

            A_main = C_torch
            B_main = H_centered_torch

            A_ridge = np.sqrt(lambda_ridge) * torch.eye(C.shape[1], device=device)
            B_ridge = torch.zeros((C.shape[1], H_centered.shape[1]), device=device)

            if use_lb and lb_mat_np is not None:
                lb_mat = torch.from_numpy(lb_mat_np).to(device)
                A_reg = torch.matmul(lb_mat, C_torch)
                B_reg = torch.zeros_like(H_centered_torch)
                A_aug = torch.cat([A_main, np.sqrt(lambda_lb) * A_reg, A_ridge], dim=0)
                B_aug = torch.cat([B_main, B_reg, B_ridge], dim=0)
            else:
                A_aug = torch.cat([A_main, A_ridge], dim=0)
                B_aug = torch.cat([B_main, B_ridge], dim=0)

            print(f"[BACD] Solving Linear System (A={A_aug.shape}, B={B_aug.shape})...")
            # torch.linalg.lstsq returns (solution, residuals, rank, singular_values)
            W_T_torch = torch.linalg.lstsq(A_aug, B_aug).solution
            W_T = W_T_torch.cpu().numpy()
            print("[BACD] Linear System solved.")
        else:
            A_main = C_centered
            B_main = H_centered
            A_ridge = np.sqrt(lambda_ridge) * np.eye(C.shape[1], dtype=np.float32)
            B_ridge = np.zeros((C.shape[1], H_centered.shape[1]), dtype=np.float32)
            if use_lb and lb_mat_np is not None:
                lb_mat = np.asarray(lb_mat_np, dtype=np.float32)
                A_reg = lb_mat @ C_float
                B_reg = np.zeros_like(H_centered)
                A_aug = np.vstack([A_main, np.sqrt(lambda_lb) * A_reg, A_ridge])
                B_aug = np.vstack([B_main, B_reg, B_ridge])
            else:
                A_aug = np.vstack([A_main, A_ridge])
                B_aug = np.vstack([B_main, B_ridge])
            print(f"[BACD] Solving Linear System (A={A_aug.shape}, B={B_aug.shape})...")
            W_T, resid, _, _ = np.linalg.lstsq(A_aug, B_aug, rcond=None)
            print("[BACD] Linear System solved.")
            if len(resid) > 0:
                print(f"[BACD] LSTSQ Residual: {resid.sum():.4e}")

    except Exception as e:
        print(f"[BACD] LSTSQ Fallback (Error: {e})")
        W_T, _, _, _ = np.linalg.lstsq(C, H_centered, rcond=None)
    
    W_star = W_T.T.astype(np.float32)
    print(f"[BACD] Computing SVD (W_star shape: {W_star.shape})...")
    
    if use_gpu:
        W_star_torch = torch.from_numpy(W_star).to(device)
        U_t, S_t, V_t = torch.linalg.svd(W_star_torch, full_matrices=False)
        U_svd = U_t.cpu().numpy()
        S_svd = S_t.cpu().numpy()
        Vt_svd = V_t.cpu().numpy()
    else:
        U_svd, S_svd, Vt_svd = np.linalg.svd(W_star, full_matrices=False)

    max_rank = min(target_rank, U_svd.shape[1], Vt_svd.shape[0])
    r = max_rank
    U_r = U_svd[:, :r]
    S_r = S_svd[:r]
    V_r = Vt_svd[:r, :].T
    sqrt_S = np.sqrt(S_r).astype(np.float32)
    Bt = U_r * sqrt_S[None, :]
    A = sqrt_S[:, None] * V_r.T
    bias = (H_mean - (C_mean @ W_T)).reshape(-1).astype(np.float32)
    decoder_state = {
        "u": U_k.astype(np.float32),
        "a": A.astype(np.float32),
        "bt": Bt.astype(np.float32),
        "bias": bias.astype(np.float32),
    }
    decoder = PyRiemannianDecoder(
        decoder_state["u"],
        decoder_state["a"],
        decoder_state["bt"],
        decoder_state["bias"],
    )
    with torch.no_grad():
        lm_head = original_model.lm_head
        H_rs_t = torch.from_numpy(H_rs[:1024]).to(device)
        logits_t_ref = lm_head(torch.from_numpy(H_t[:1024]).to(device))
        hidden_s_np = decoder.forward(H_rs_t.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32))
        hidden_s = torch.from_numpy(hidden_s_np).to(device=device, dtype=torch.float32)
        logits_s = lm_head(hidden_s)
        t_flat = logits_t_ref.view(-1, logits_t_ref.size(-1))
        s_flat = logits_s.view(-1, logits_s.size(-1))
        cos = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
        rel = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)
        print(f"[Decoder fit] final_logits (train-sample) cos={cos:.4f}, rel_l2={rel:.4f}")
    end_time = time.time()
    print(f"[BACD] Decoder fitting finished in {end_time - start_time:.2f}s")
    return decoder, decoder_state

