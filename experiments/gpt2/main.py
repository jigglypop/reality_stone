import os
import sys

import torch
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from reality_stone.models.transformer_converter import RSULFTransformerConverter

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
from decoder import rsulf_generate_text, build_human_decoder
from trainer import fit_riemannian_decoder
from status import analyze_layer_fidelity, analyze_layer_fidelity_blockwise


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
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def run_structural_rsulf_experiment(original_model, tokenizer, device, prompt, human_decoder=None):
    print("\n1. RS-ULF 빌드")
    structural_model = build_structural_rsulf_model(original_model).to(device)
    analyze_layer_fidelity(original_model, structural_model, tokenizer, prompt, device)
    analyze_layer_fidelity_blockwise(original_model, structural_model, tokenizer, prompt, device)
    struct_text, struct_time = rsulf_generate_text(
        original_model,
        structural_model,
        tokenizer,
        device,
        prompt,
    )
    return struct_text, struct_time


def run_rsulf_rank_sweep(original_model, tokenizer, device, prompt, human_decoder=None):
    print("\n2. RS-ULF 변환 테스트")
    full_rank_r = original_model.config.n_embd
    start_r = min(full_rank_r, 356)
    current_r = start_r
    decoder_cache = {}
    while current_r >= 1:
        print(f"\n--- Testing with rank r={current_r} ---")
        config = {
            "d_model": original_model.config.n_embd,
            "r": current_r,
            "eta": 0.005,
            "alpha": 0.01,
            "beta": 0.0,
            "gamma": 0.99,
            "seq_len": 64,
            "window": 4,
            "calibration_samples": 0,
            # Universal PFC option:
            # - pfc_mode="accel": trajectory-only (no dependence on RS-ULF factorization)
            # - pfc_mode="bilinear": legacy (uses FFN low-rank factors)
            "pfc_mode": "accel",
            "pfc_curvature": 3e-4,
            "verbose": True,
            "exact": True,
            "pfc_window": 8,
            "pfc_layers": 6,
            "pfc_speed_gate": 1.0,
        }
        converter = RSULFTransformerConverter(**config)
        rs_layers = converter.convert_model(original_model)
        rs_layers = rs_layers.to(device)
        basis = getattr(converter, "global_basis", None)
        analyze_layer_fidelity(original_model, rs_layers, tokenizer, prompt, device)
        decoder = decoder_cache.get(current_r)
        if decoder is None:
            decoder = fit_riemannian_decoder(
                original_model,
                rs_layers,
                tokenizer,
                device,
                global_basis=basis,
                target_rank=current_r,
                num_batches=16,
                batch_size=4,
                seq_len=32,
                prompt_text=prompt,
                data_mode="teacher_sample",
            )
            decoder_cache[current_r] = decoder
        rs_text, rs_time = rsulf_generate_text(
            original_model,
            rs_layers,
            tokenizer,
            device,
            prompt,
            rank=current_r,
            decoder=decoder,
            pfc_curvature=0.0,
            entity_memory=48,
            entity_beta=0.20,
            entity_temp=0.35,
            entity_min_sim=0.40,
            entity_warmup=10,
            entity_beta_max=0.45,
            entity_gate_sigma=0.08,
            entity_ctx_determiner=True,
            entity_ctx_pronoun=True,
            entity_ctx_prep=True,
            # teacher-guided decoding to suppress rare junk tokens and stabilize nouns/objects
            teacher_guidance=0.25,
            teacher_topk_mask=200,
        )
        print(f"   [RS-ULF r={current_r}]: {rs_text}")
        print(f"   Time: {rs_time:.4f}s")
        if current_r == 1:
            break
        current_r = current_r // 2
        if current_r < 1:
            current_r = 1


def test_gpt2_conversion():
    print("=== [Reality Stone] GPT-2 변환 테스트 ===")
    device = select_device()
    print("\n1. Loading Original GPT-2...")
    model_name = "gpt2"
    tokenizer, original_model = load_gpt2_components(model_name, device)
    prompt = "The secret of the universe is"
    struct_text, struct_time = run_structural_rsulf_experiment(
        original_model,
        tokenizer,
        device,
        prompt,
    )
    run_rsulf_rank_sweep(original_model, tokenizer, device, prompt, human_decoder=None)
    print("\n=== 요약 ===")
    print(f"프롬프트 : {prompt}")
    print(f"1. RS-ULF(Py):         {struct_text.strip()}")


if __name__ == "__main__":
    try:
        test_gpt2_conversion()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()

