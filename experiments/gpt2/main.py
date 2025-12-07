import os
import sys

import torch
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from reality_stone.models.transformer_converter import RSULFTransformerConverter

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from encoder import build_structural_rsulf_model, distill_structural_potentials
from decoder import fit_riemannian_decoder, rsulf_generate_text
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


def run_structural_rsulf_experiment(original_model, tokenizer, device, prompt):
    print("\n1. RS-ULF 빌드")
    structural_model = build_structural_rsulf_model(original_model).to(device)
    analyze_layer_fidelity(original_model, structural_model, tokenizer, prompt, device)
    analyze_layer_fidelity_blockwise(original_model, structural_model, tokenizer, prompt, device)
    distill_structural_potentials(
        original_model,
        structural_model,
        tokenizer,
        device,
        steps=50,
        batch_size=4,
        seq_len=32,
        lr=1e-4,
        lambda_energy=0.1,
    )
    analyze_layer_fidelity(original_model, structural_model, tokenizer, prompt, device)
    struct_text, struct_time = rsulf_generate_text(
        original_model,
        structural_model,
        tokenizer,
        device,
        prompt,
    )
    return struct_text, struct_time


def run_rsulf_rank_sweep(original_model, tokenizer, device, prompt):
    print("\n2. RS-ULF 변환 테스트")
    full_rank_r = original_model.config.n_embd
    start_r = min(full_rank_r, 356)
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
            "seq_len": 64,
            "window": 4,
            "verbose": True,
            "exact": True,
        }
        converter = RSULFTransformerConverter(**config)
        rs_layers = converter.convert_model(original_model)
        rs_layers = rs_layers.to(device)
        basis = getattr(converter, "global_basis", None)
        decoder = fit_riemannian_decoder(
            original_model,
            rs_layers,
            tokenizer,
            device,
            basis,
            target_rank=current_r,
            num_batches=4,
            batch_size=4,
            seq_len=32,
        )
        analyze_layer_fidelity(original_model, rs_layers, tokenizer, prompt, device)
        rs_text, rs_time = rsulf_generate_text(
            original_model,
            rs_layers,
            tokenizer,
            device,
            prompt,
            decoder=decoder,
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
    run_rsulf_rank_sweep(original_model, tokenizer, device, prompt)
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
