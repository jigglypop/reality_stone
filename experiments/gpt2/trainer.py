import time
import numpy as np
import torch
import torch.nn.functional as F
from reality_stone._rust import laplace_beltrami_matrix, PyRiemannianDecoder

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


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
    wpe = original_model.transformer.wpe
    seed_inputs = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), device=device)
    embeds = wte(seed_inputs).detach().requires_grad_(True)
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos_emb = wpe(pos)
    x = embeds + pos_emb
    outputs = original_model.transformer(inputs_embeds=x)
    hidden_last = outputs.last_hidden_state
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
):
    if isinstance(device, str):
        device = torch.device(device)
    original_model.eval()
    rs_model.eval()
    ln_f = original_model.transformer.ln_f
    lm_head = original_model.lm_head
    wpe = original_model.transformer.wpe
    h_rs_list = []
    h_t_list = []
    logits_t_list = []
    half_batches = max(1, num_batches // 2)
    inputs_high, inputs_low = _generate_binary_curriculum_prompts(
        original_model, tokenizer, device, half_batches, batch_size, seq_len
    )
    all_inputs = torch.cat([inputs_high, inputs_low], dim=0)
    total_samples = all_inputs.size(0)
    start = time.time()
    total_batches = (total_samples + batch_size - 1) // batch_size
    indices = range(0, total_samples, batch_size)
    if _tqdm is not None:
        iterator = _tqdm(indices, total=total_batches, desc="[BACD] collect", leave=False)
    else:
        iterator = indices
    with torch.no_grad():
        for b, i in enumerate(iterator, 1):
            batch_embeds = all_inputs[i : i + batch_size]
            pos = torch.arange(seq_len, dtype=torch.long, device=device)
            pos_emb = wpe(pos)
            x = batch_embeds + pos_emb
            outputs = original_model.transformer(inputs_embeds=x, output_hidden_states=True)
            teacher_hidden_last = outputs.hidden_states[-1]
            t_last = ln_f(teacher_hidden_last)
            t_logits = lm_head(t_last)
            h_rs = x
            for wrapper in rs_model.wrappers:
                h_rs = wrapper(h_rs)
            h_rs_list.append(h_rs.reshape(-1, h_rs.size(-1)).cpu())
            h_t_list.append(teacher_hidden_last.reshape(-1, teacher_hidden_last.size(-1)).cpu())
            logits_t_list.append(t_logits.reshape(-1, t_logits.size(-1)).cpu())
            current_norm = h_rs.norm(dim=-1).mean().item()
            if _tqdm is not None:
                iterator.set_postfix(batch=b, total=total_batches, norm=current_norm)
            print(f"[BACD] batch={b}/{total_batches} norm={current_norm:.2f}")
    elapsed = time.time() - start
    H_rs = torch.cat(h_rs_list, dim=0).numpy().astype(np.float32)
    H_t = torch.cat(h_t_list, dim=0).numpy().astype(np.float32)
    L_t = torch.cat(logits_t_list, dim=0).numpy().astype(np.float32)
    print(f"[BACD] Collected tokens={H_rs.shape[0]} dim={H_rs.shape[1]} time={elapsed:.2f}s")
    return H_rs, H_t, L_t


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
):
    if isinstance(device, str):
        device = torch.device(device)
    H_rs, H_t, L_t = _collect_decoder_data(
        original_model,
        rs_model,
        tokenizer,
        device,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
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
    H_mean = H_t.mean(axis=0, keepdims=True)
    H_centered = H_t - H_mean
    lambda_lb = 0.1
    lambda_ridge = 1e-3
    use_gpu = device.type == "cuda" or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and device.type == "mps"
    )

    try:
        C_float = C.astype(np.float32)
        print(f"[BACD] Computing Laplace-Beltrami matrix (samples={C_float.shape[0]})...")
        lb_mat_np = laplace_beltrami_matrix(C_float, "diagonal", 0.0, 0.5, 1e-6)
        print("[BACD] LB matrix computed.")

        if use_gpu:
            print(f"[BACD] Switching to GPU ({device}) for Solver...")
            lb_mat = torch.from_numpy(lb_mat_np).to(device)
            C_torch = torch.from_numpy(C_float).to(device)
            H_centered_torch = torch.from_numpy(H_centered).to(device)

            A_main = C_torch
            B_main = H_centered_torch

            # Matrix multiplication on GPU
            A_reg = torch.matmul(lb_mat, C_torch)
            B_reg = torch.zeros_like(H_centered_torch)

            A_ridge = np.sqrt(lambda_ridge) * torch.eye(C.shape[1], device=device)
            B_ridge = torch.zeros((C.shape[1], H_centered.shape[1]), device=device)

            A_aug = torch.cat([A_main, np.sqrt(lambda_lb) * A_reg, A_ridge], dim=0)
            B_aug = torch.cat([B_main, B_reg, B_ridge], dim=0)

            print(f"[BACD] Solving Linear System (A={A_aug.shape}, B={B_aug.shape})...")
            # torch.linalg.lstsq returns (solution, residuals, rank, singular_values)
            W_T_torch = torch.linalg.lstsq(A_aug, B_aug).solution
            W_T = W_T_torch.cpu().numpy()
            print("[BACD] Linear System solved.")
        else:
            lb_mat = np.asarray(lb_mat_np, dtype=np.float32)
            A_main = C_float
            B_main = H_centered
            A_reg = lb_mat @ C_float
            B_reg = np.zeros_like(H_centered)
            A_ridge = np.sqrt(lambda_ridge) * np.eye(C.shape[1], dtype=np.float32)
            B_ridge = np.zeros((C.shape[1], H_centered.shape[1]), dtype=np.float32)
            A_aug = np.vstack([A_main, np.sqrt(lambda_lb) * A_reg, A_ridge])
            B_aug = np.vstack([B_main, B_reg, B_ridge])
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
    bias = H_mean.reshape(-1).astype(np.float32)
    decoder = PyRiemannianDecoder(
        U_k.astype(np.float32),
        A.astype(np.float32),
        Bt.astype(np.float32),
        bias.astype(np.float32),
    )
    with torch.no_grad():
        ln_f = original_model.transformer.ln_f
        lm_head = original_model.lm_head
        H_rs_t = torch.from_numpy(H_rs[:1024]).to(device)
        logits_t_ref = torch.from_numpy(L_t[:1024]).to(device)
        hidden_s_np = decoder.forward(H_rs_t.cpu().numpy().astype(np.float32))
        hidden_s = torch.from_numpy(hidden_s_np).to(device)
        hidden_s = hidden_s.view_as(logits_t_ref[:, : hidden_s.size(-1)])
        hidden_s = hidden_s.view(-1, hidden_s.size(-1))
        hidden_s = hidden_s.view(-1, 1, hidden_s.size(-1))
        logits_s = lm_head(ln_f(hidden_s))
        logits_s = logits_s.view_as(logits_t_ref)
        t_flat = logits_t_ref.view(-1, logits_t_ref.size(-1))
        s_flat = logits_s.view(-1, logits_s.size(-1))
        cos = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
        rel = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)
        print(f"[Decoder fit] final_logits (train-sample) cos={cos:.4f}, rel_l2={rel:.4f}")
    end_time = time.time()
    print(f"[BACD] Decoder fitting finished in {end_time - start_time:.2f}s")
    return decoder
