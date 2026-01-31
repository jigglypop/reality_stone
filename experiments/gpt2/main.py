import os
import sys

import torch
from transformers import GPT2Tokenizer

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
PY_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "python"))
if PY_ROOT not in sys.path:
    sys.path.append(PY_ROOT)

from encoder import (
    build_structural_rsulf_model,
    distill_structural_potentials,
    distill_gpt2_to_rsulf,
    distill_syntax_head,
    build_gpt2_rsffn_model,
    distill_gpt2_ffn_only,
    distill_gpt2_rsffn_e2e,
    distill_gpt2_rsffn_curriculum,
    distill_gpt2_rsffn_curriculum_two_phase,
    RSFFNTrainer,
)
from eval_distill import eval_teacher_vs_rsulf


def _safe_print(text: str) -> None:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)


def _parse_int_list(value: str) -> list[int]:
    raw = (value or "").strip()
    if not raw:
        return []
    for ch in [",", "/", "|", ";", "\n", "\t"]:
        raw = raw.replace(ch, " ")
    parts = []
    for p in raw.split(" "):
        s = p.strip()
        if not s:
            continue
        try:
            parts.append(int(s))
        except Exception:
            continue
    out: list[int] = []
    seen = set()
    for x in parts:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def select_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Device: cpu (CUDA not available)")
        print("WARNING: User requested CUDA but it is not available. Running on CPU.")
    return device


def load_gpt2_components(model_name, device):
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def run_structural_rsulf_experiment(original_model, tokenizer, device, prompt, human_decoder=None):
    print("\n1. RS-ULF 빌드")
    structural_model = build_structural_rsulf_model(original_model).to(device)
    return "", 0.0


def run_rsulf_rank_sweep(original_model, tokenizer, device, prompt, human_decoder=None):
    from reality_stone.models.transformer_converter import (
        RSULFTransformerConverter,
        wrap_rsulf_as_causal_lm,
        save_rsulf_causal_lm,
    )
    from trainer import fit_riemannian_decoder
    from decoder import rsulf_generate_text_pure
    print("\n2. RS-ULF 변환 테스트")
    full_rank_r = original_model.config.n_embd
    ranks_env = os.environ.get("RSULF_RANKS", "").strip()
    ranks = _parse_int_list(ranks_env)
    if ranks:
        orig_ranks = os.environ.pop("RSULF_RANKS", None)
        orig_start = os.environ.get("RSULF_START_R", None)
        orig_min = os.environ.get("RSULF_MIN_R", None)
        last = None
        for r in ranks:
            r_int = int(r)
            if r_int < 1:
                r_int = 1
            print(f"\n=== Testing RS-ULF with rank r={r_int} ===")
            os.environ["RSULF_START_R"] = str(r_int)
            os.environ["RSULF_MIN_R"] = str(r_int)
            last = run_rsulf_rank_sweep(original_model, tokenizer, device, prompt, human_decoder=human_decoder)
        if orig_ranks is not None:
            os.environ["RSULF_RANKS"] = orig_ranks
        else:
            os.environ.pop("RSULF_RANKS", None)
        if orig_start is not None:
            os.environ["RSULF_START_R"] = orig_start
        else:
            os.environ.pop("RSULF_START_R", None)
        if orig_min is not None:
            os.environ["RSULF_MIN_R"] = orig_min
        else:
            os.environ.pop("RSULF_MIN_R", None)
        return last
    start_r = int(os.environ.get("RSULF_START_R", str(full_rank_r)))
    if start_r < 1:
        start_r = 1
    if start_r > full_rank_r:
        start_r = full_rank_r
    min_rank = int(os.environ.get("RSULF_MIN_R", "32"))
    if min_rank < 1:
        min_rank = 1
    current_r = start_r
    while current_r >= 1:
        print(f"\n--- Testing with rank r={current_r} ---")
        geo_flow = os.environ.get("RSULF_GEO_FLOW", "").strip().lower() in ("1", "true", "yes")
        geo_blend = float(os.environ.get("RSULF_GEO_BLEND", "0.0"))
        config = {
            "d_model": original_model.config.n_embd,
            "r": current_r,
            "eta": 0.005,
            "alpha": 0.01,
            "beta": 0.01,
            "gamma": 0.99,
            "seq_len": 32,
            "window": 4,
            "calibration_samples": 0,
            # Universal PFC option:
            # - pfc_mode="accel": trajectory-only (no dependence on RS-ULF factorization)
            # - pfc_mode="bilinear": legacy (uses FFN low-rank factors)
            "pfc_mode": "accel",
            "pfc_curvature": 6e-4,
            "pfc_max_rel": 0.05,
            "verbose": True,
            "exact": True,
            "pfc_window": 8,
            "pfc_layers": 3,
            "pfc_speed_gate": 1.25,
            "use_geodesic_flow": geo_flow,
            "geodesic_blend": geo_blend,
        }
        converter = RSULFTransformerConverter(**config)
        rs_layers = converter.convert_model(original_model)
        rs_layers = rs_layers.to(device)
        if hasattr(rs_layers, "reset_memory"):
            rs_layers.reset_memory()
        basis = getattr(converter, "global_basis", None)
        tokens_target = int(os.environ.get("RSULF_DECODER_TOKENS", "16384"))
        if tokens_target < 1024:
            tokens_target = 1024
        batch_size = int(os.environ.get("RSULF_DECODER_BS", "4"))
        if batch_size < 1:
            batch_size = 1
        seq_len = int(config.get("seq_len", 32))
        num_batches = int(max(4, tokens_target // max(1, (batch_size * seq_len))))
        decoder, decoder_state = fit_riemannian_decoder(
            original_model,
            rs_layers,
            tokenizer,
            device,
            global_basis=basis,
            target_rank=full_rank_r,
            num_batches=num_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            prompt_text=None,
            data_mode="mixed",
            corpus_roots=["docs", "README.md", "rules.mdc"],
            corpus_max_docs=2000,
            corpus_chunk_chars=8000,
            corpus_overlap_chars=1000,
            mix_p_corpus=0.6,
            mix_p_random=0.3,
            mix_p_teacher_sample=0.1,
        )
        rs_lm_plain = wrap_rsulf_as_causal_lm(original_model, rs_layers).to(device)
        rs_lm_plain.decoder = None
        rs_lm_plain.apply_final_norm = True

        rs_lm = wrap_rsulf_as_causal_lm(original_model, rs_layers).to(device)
        rs_lm.decoder = decoder
        rs_lm.apply_final_norm = False

        steps_head = int(os.environ.get("RSULF_SYNTAX_STEPS", "0"))
        rs_lm_head = None
        if steps_head > 0:
            head = distill_syntax_head(
                original_model,
                rs_layers,
                tokenizer,
                device,
                steps=steps_head,
                batch_size=batch_size,
                seq_len=seq_len,
                lr=float(os.environ.get("RSULF_SYNTAX_LR", "1e-4")),
                hidden_dim=None,
            )
            rs_lm_head = wrap_rsulf_as_causal_lm(original_model, rs_layers).to(device)
            rs_lm_head.decoder = None
            rs_lm_head.apply_final_norm = True
            rs_lm_head.syntax_head.load_state_dict(head.state_dict())
        save_dir = os.environ.get("RSULF_SAVE_DIR", "").strip()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"gpt2_rsulf_r{current_r}.pt")

        # Held-out sanity check: fixed prompt batch
        with torch.no_grad():
            held_enc = tokenizer(
                [prompt, "In summary, the key idea is", "Q: What is the answer?\nA:"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
            held = held_enc["input_ids"].to(device)
            held_mask = held_enc.get("attention_mask")
            ev_plain = eval_teacher_vs_rsulf(original_model, rs_lm_plain, tokenizer, device, held, attention_mask=held_mask)
            ev_dec = eval_teacher_vs_rsulf(original_model, rs_lm, tokenizer, device, held, attention_mask=held_mask)
            ev_best = ev_plain
            best_model = rs_lm_plain
            best_decoder_state = None
            if ev_dec.rsulf_ppl < ev_best.rsulf_ppl:
                ev_best = ev_dec
                best_model = rs_lm
                best_decoder_state = decoder_state
            if rs_lm_head is not None:
                ev_head = eval_teacher_vs_rsulf(original_model, rs_lm_head, tokenizer, device, held, attention_mask=held_mask)
                if ev_head.rsulf_ppl < ev_best.rsulf_ppl:
                    ev_best = ev_head
                    best_model = rs_lm_head
                    best_decoder_state = None
            ev = ev_best
            print(
                f"   [EVAL r={current_r}] ppl_t={ev.teacher_ppl:.2f} ppl_rs={ev.rsulf_ppl:.2f} "
                f"logits_cos={ev.logits_cos:.4f} rel_l2={ev.logits_rel_l2:.4f}"
            )
            rs_lm = best_model
            decoder_state = best_decoder_state
            if save_dir:
                save_rsulf_causal_lm(save_path, rs_lm, decoder_state=decoder_state)

        rs_text, rs_time = rsulf_generate_text_pure(
            rs_lm,
            tokenizer,
            device,
            prompt,
            max_tokens=30,
        )
        _safe_print(f"   [RS-ULF r={current_r}]: {rs_text}")
        print(f"   Time: {rs_time:.4f}s")
        if current_r <= min_rank:
            break
        next_r = current_r // 2
        if next_r < min_rank:
            next_r = min_rank
        current_r = max(1, int(next_r))


def run_rsffn_only_experiment(original_model, tokenizer, device, prompt):
    ks_env = os.environ.get("RSFFN_KS", "").strip()
    ks = _parse_int_list(ks_env)
    if ks:
        orig_ks = os.environ.pop("RSFFN_KS", None)
        orig_k = os.environ.get("RSFFN_K", None)
        load_tpl = os.environ.get("RSFFN_LOAD_PATH", "")
        save_tpl = os.environ.get("RSFFN_SAVE_PATH", "")
        last = None
        if str(save_tpl).strip() and "{k}" not in str(save_tpl):
            os.environ["RSFFN_SAVE_PATH"] = ""
        for k0 in ks:
            k_int = int(k0)
            if k_int < 1:
                k_int = 1
            print(f"\n--- Testing RSFFN with k={k_int} ---")
            os.environ["RSFFN_K"] = str(k_int)
            if "{k}" in str(load_tpl):
                os.environ["RSFFN_LOAD_PATH"] = str(load_tpl).format(k=k_int)
            if "{k}" in str(save_tpl):
                os.environ["RSFFN_SAVE_PATH"] = str(save_tpl).format(k=k_int)
            last = run_rsffn_only_experiment(original_model, tokenizer, device, prompt)
        if orig_ks is not None:
            os.environ["RSFFN_KS"] = orig_ks
        else:
            os.environ.pop("RSFFN_KS", None)
        if orig_k is not None:
            os.environ["RSFFN_K"] = orig_k
        else:
            os.environ.pop("RSFFN_K", None)
        os.environ["RSFFN_LOAD_PATH"] = str(load_tpl)
        os.environ["RSFFN_SAVE_PATH"] = str(save_tpl)
        return last
    k = int(os.environ.get("RSFFN_K", "64"))
    replace_last_n = int(os.environ.get("RSFFN_LAST_N", "3"))
    steps = int(os.environ.get("RSFFN_STEPS", "200"))
    lr = float(os.environ.get("RSFFN_LR", "1e-3"))
    batch_size = int(os.environ.get("RSFFN_BS", "4"))
    seq_len = int(os.environ.get("RSFFN_SEQ_LEN", "32"))
    mode = os.environ.get("RSFFN_MODE", "ffn").strip().lower()
    student = build_gpt2_rsffn_model(original_model, device=device, k=k, replace_last_n=replace_last_n)
    load_path = os.environ.get("RSFFN_LOAD_PATH", "").strip()
    if load_path:
        state = torch.load(load_path, map_location=device)
        if isinstance(state, dict):
            student.load_state_dict(state, strict=False)
    if mode in ("pipeline", "fullpipe"):
        stages_s = os.environ.get("RSFFN_STAGES", "1,3,6,12")
        stages = []
        for part in stages_s.split(","):
            p = part.strip()
            if not p:
                continue
            stages.append(int(p))
        if not stages:
            stages = [1, 3, 6, 12]
        stage_ffn_steps = int(os.environ.get("RSFFN_STAGE_FFN_STEPS", "200"))
        stage_e2e_steps = int(os.environ.get("RSFFN_STAGE_E2E_STEPS", "100"))
        ffn_lr = float(os.environ.get("RSFFN_FFN_LR", "2e-3"))
        e2e_lr = float(os.environ.get("RSFFN_LR", str(lr)))
        logits_mse_w = float(os.environ.get("RSFFN_LOGITS_MSE_W", "0.2"))
        hidden_last_w = float(os.environ.get("RSFFN_HLAST_W", "0.1"))
        hidden_layers_w = float(os.environ.get("RSFFN_HLAYERS_W", "0.02"))
        kl_w = float(os.environ.get("RSFFN_KL_W", "1.0"))
        kl_t = float(os.environ.get("RSFFN_KL_T", "2.0"))
        trainer = RSFFNTrainer(original_model, student, tokenizer, device=device, lr=e2e_lr)
        for n in stages:
            student.set_replace_last_n(int(n))
            student.set_trainable_last_n(int(n))
            trainer.set_lr(ffn_lr)
            trainer.train_ffn_only(steps=stage_ffn_steps, batch_size=batch_size, seq_len=seq_len)
            trainer.set_lr(e2e_lr)
            trainer.train_e2e(
                steps=stage_e2e_steps,
                batch_size=batch_size,
                seq_len=seq_len,
                logits_mse_weight=logits_mse_w,
                kl_weight=kl_w,
                kl_temperature=kl_t,
                hidden_last_weight=hidden_last_w,
                hidden_layers_weight=hidden_layers_w,
            )
            with torch.no_grad():
                held_enc = tokenizer(
                    [prompt, "In summary, the key idea is", "Q: What is the answer?\nA:"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=seq_len,
                )
                held = held_enc["input_ids"].to(device)
                held_mask = held_enc.get("attention_mask")
                ev_s = eval_teacher_vs_rsulf(original_model, student, tokenizer, device, held, attention_mask=held_mask)
                print(
                    f"   [PIPE stage={n}] ppl_t={ev_s.teacher_ppl:.2f} ppl_rs={ev_s.rsulf_ppl:.2f} "
                    f"logits_cos={ev_s.logits_cos:.4f} rel_l2={ev_s.logits_rel_l2:.4f}"
                )
            save_dir = os.environ.get("RSFFN_SAVE_DIR", "").strip()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(student.state_dict(), os.path.join(save_dir, f"rsffn_k{k}_n{n}.pt"))
        replace_last_n = int(max(stages))
    elif mode in ("curriculum2", "two", "two_phase", "two-phase"):
        stages_s = os.environ.get("RSFFN_STAGES", "1,3,6,12")
        stages = []
        for part in stages_s.split(","):
            p = part.strip()
            if not p:
                continue
            stages.append(int(p))
        if not stages:
            stages = [1, 3, 6, 12]
        stage_ffn_steps = int(os.environ.get("RSFFN_STAGE_FFN_STEPS", "50"))
        stage_e2e_steps = int(os.environ.get("RSFFN_STAGE_E2E_STEPS", "20"))
        logits_mse_w = float(os.environ.get("RSFFN_LOGITS_MSE_W", "0.2"))
        hidden_last_w = float(os.environ.get("RSFFN_HLAST_W", "0.1"))
        hidden_layers_w = float(os.environ.get("RSFFN_HLAYERS_W", "0.02"))
        kl_w = float(os.environ.get("RSFFN_KL_W", "1.0"))
        kl_t = float(os.environ.get("RSFFN_KL_T", "2.0"))
        ffn_lr = float(os.environ.get("RSFFN_FFN_LR", str(lr)))
        student = distill_gpt2_rsffn_curriculum_two_phase(
            original_model,
            student,
            tokenizer,
            device,
            stages=stages,
            stage_ffn_steps=stage_ffn_steps,
            stage_e2e_steps=stage_e2e_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            lr=lr,
            ffn_lr=ffn_lr,
            logits_mse_weight=logits_mse_w,
            hidden_last_weight=hidden_last_w,
            hidden_layers_weight=hidden_layers_w,
            kl_weight=kl_w,
            kl_temperature=kl_t,
        )
        replace_last_n = int(max(stages))
    elif mode in ("curriculum", "sched", "schedule"):
        stages_s = os.environ.get("RSFFN_STAGES", "1,3,6,12")
        stage_steps = int(os.environ.get("RSFFN_STAGE_STEPS", str(max(1, steps))))
        stages = []
        for part in stages_s.split(","):
            p = part.strip()
            if not p:
                continue
            stages.append(int(p))
        if not stages:
            stages = [1, 3, 6, 12]
        logits_mse_w = float(os.environ.get("RSFFN_LOGITS_MSE_W", "1.0"))
        hidden_last_w = float(os.environ.get("RSFFN_HLAST_W", "0.2"))
        hidden_layers_w = float(os.environ.get("RSFFN_HLAYERS_W", "0.05"))
        kl_w = float(os.environ.get("RSFFN_KL_W", "0.5"))
        kl_t = float(os.environ.get("RSFFN_KL_T", "2.0"))
        student = distill_gpt2_rsffn_curriculum(
            original_model,
            student,
            tokenizer,
            device,
            stages=stages,
            stage_steps=stage_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            lr=lr,
            logits_mse_weight=logits_mse_w,
            hidden_last_weight=hidden_last_w,
            hidden_layers_weight=hidden_layers_w,
            kl_weight=kl_w,
            kl_temperature=kl_t,
        )
        replace_last_n = int(max(stages))
    elif mode in ("e2e", "logits", "full"):
        logits_mse_w = float(os.environ.get("RSFFN_LOGITS_MSE_W", "1.0"))
        hidden_last_w = float(os.environ.get("RSFFN_HLAST_W", "0.2"))
        hidden_layers_w = float(os.environ.get("RSFFN_HLAYERS_W", "0.0"))
        kl_w = float(os.environ.get("RSFFN_KL_W", "0.0"))
        kl_t = float(os.environ.get("RSFFN_KL_T", "1.0"))
        student = distill_gpt2_rsffn_e2e(
            original_model,
            student,
            tokenizer,
            device,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            lr=lr,
            logits_mse_weight=logits_mse_w,
            hidden_last_weight=hidden_last_w,
            hidden_layers_weight=hidden_layers_w,
            kl_weight=kl_w,
            kl_temperature=kl_t,
        )
    else:
        student = distill_gpt2_ffn_only(
            original_model,
            student,
            tokenizer,
            device,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            lr=lr,
        )
    save_path = os.environ.get("RSFFN_SAVE_PATH", "").strip()
    if save_path:
        torch.save(student.state_dict(), save_path)

    student.set_replace_last_n(int(replace_last_n))
    student.set_trainable_last_n(0)
    with torch.no_grad():
        held_enc = tokenizer(
            [prompt, "In summary, the key idea is", "Q: What is the answer?\nA:"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        held = held_enc["input_ids"].to(device)
        held_mask = held_enc.get("attention_mask")
        ev = eval_teacher_vs_rsulf(original_model, student, tokenizer, device, held, attention_mask=held_mask)
        print(
            f"   [EVAL RSFFN k={k} last_n={replace_last_n}] ppl_t={ev.teacher_ppl:.2f} ppl_rs={ev.rsulf_ppl:.2f} "
            f"logits_cos={ev.logits_cos:.4f} rel_l2={ev.logits_rel_l2:.4f}"
        )
    gen_tokens = int(os.environ.get("RSFFN_GEN_TOKENS", "30"))
    if gen_tokens > 0:
        import time
        start = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out_ids = student.generate_sample(
            input_ids=input_ids,
            max_new_tokens=int(gen_tokens),
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.15,
            eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        )
        text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        tsec = time.time() - start
        _safe_print(f"   [RSFFN gen]: {text}")
        print(f"   Time: {tsec:.4f}s")
    return student


def test_gpt2_conversion():
    print("=== [Reality Stone] GPT-2 변환 테스트 ===")
    device = select_device()
    model_name = "gpt2"
    prompt = "The secret of the universe is"
    load_path = os.environ.get("RSULF_LOAD_PATH", "").strip()
    if load_path:
        from reality_stone.models.transformer_converter import load_rsulf_causal_lm
        from decoder import rsulf_generate_text_pure
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        rs_lm = load_rsulf_causal_lm(load_path, device=device)
        rs_text, rs_time = rsulf_generate_text_pure(rs_lm, tokenizer, device, prompt, max_tokens=30)
        _safe_print(f"   [RS-ULF load]: {rs_text}")
        print(f"   Time: {rs_time:.4f}s")
        print("\n=== 요약 ===")
        print(f"프롬프트 : {prompt}")
        print("done")
        return

    print("\n1. Loading Original GPT-2...")
    tokenizer, original_model = load_gpt2_components(model_name, device)
    if os.environ.get("RSFFN_ONLY", "").strip():
        run_rsffn_only_experiment(original_model, tokenizer, device, prompt)
    else:
        run_rsulf_rank_sweep(original_model, tokenizer, device, prompt, human_decoder=None)
    print("\n=== 요약 ===")
    print(f"프롬프트 : {prompt}")
    print("done")


if __name__ == "__main__":
    try:
        test_gpt2_conversion()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()

