import os
import sys

import torch
from transformers import GPT2Tokenizer
from reality_stone.models.transformer_converter import (
    RSULFTransformerConverter,
    wrap_rsulf_as_causal_lm,
    save_rsulf_causal_lm,
    load_rsulf_causal_lm,
)

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
)
from decoder import rsulf_generate_text_pure
from trainer import fit_riemannian_decoder
from eval_distill import eval_teacher_vs_rsulf


def _safe_print(text: str) -> None:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)


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
    print("\n2. RS-ULF 변환 테스트")
    full_rank_r = original_model.config.n_embd
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
        config = {
            "d_model": original_model.config.n_embd,
            "r": current_r,
            "eta": 0.005,
            "alpha": 0.01,
            "beta": 0.0,
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


def test_gpt2_conversion():
    print("=== [Reality Stone] GPT-2 변환 테스트 ===")
    device = select_device()
    model_name = "gpt2"
    prompt = "The secret of the universe is"
    load_path = os.environ.get("RSULF_LOAD_PATH", "").strip()
    if load_path:
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

