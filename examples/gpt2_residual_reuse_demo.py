"""Compare baseline GPT-2 generation with residual-reuse MLP caching.

Example:
    python reality_stone/examples/gpt2_residual_reuse_demo.py --model gpt2 --tokens 64
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reality_stone.models.residual_reuse import (
    install_gpt2_residual_reuse,
    reset_residual_reuse,
    residual_reuse_report,
)


def generate_once(model, tokenizer, prompt: str, tokens: int, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_mb = 0.0
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    new_tokens = max(0, int(out.shape[1] - inputs["input_ids"].shape[1]))
    return {
        "elapsed": elapsed,
        "tok_s": new_tokens / max(elapsed, 1e-9),
        "peak_mb": peak_mb,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The weather residual graph shows")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--match-metric", choices=["cosine", "relative_l2"], default="cosine")
    parser.add_argument("--distance-tolerance", type=float, default=None)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-return-full", action="store_true")
    parser.add_argument("--entries", type=int, default=128)
    parser.add_argument("--signature-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    baseline = generate_once(model, tokenizer, args.prompt, args.tokens, device)

    wrappers = install_gpt2_residual_reuse(
        model,
        similarity_threshold=args.threshold,
        distance_tolerance=args.distance_tolerance,
        match_metric=args.match_metric,
        audit=args.audit,
        audit_return_cached=not args.audit_return_full,
        max_entries=args.entries,
        signature_dim=args.signature_dim,
        enabled=True,
    )
    model.eval()
    reset_residual_reuse(wrappers)
    reuse = generate_once(model, tokenizer, args.prompt, args.tokens, device)
    report = residual_reuse_report(wrappers)

    print("baseline:")
    print(f"  elapsed={baseline['elapsed']:.4f}s  tok/s={baseline['tok_s']:.2f}  peak_mb={baseline['peak_mb']:.1f}")
    print("residual-reuse:")
    print(f"  elapsed={reuse['elapsed']:.4f}s  tok/s={reuse['tok_s']:.2f}  peak_mb={reuse['peak_mb']:.1f}")
    print(
        "  hits={hits}  misses={misses}  hit_rate={hit_rate:.2%}  disabled={disabled}".format(
            **report
        )
    )
    if report.get("audit_hits", 0):
        print(
            "  audit_hits={audit_hits}  audit_rel_error_mean={audit_rel_error_mean:.6f}  "
            "audit_rel_error_max={audit_rel_error_max:.6f}".format(**report)
        )
    if reuse["elapsed"] > 0:
        print(f"speedup={baseline['elapsed'] / reuse['elapsed']:.3f}x")
    print("output:")
    print(reuse["text"])


if __name__ == "__main__":
    main()
