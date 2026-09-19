"""Measure low-rank structure of GPT-2 MLP residuals.

This collects each block's MLP output ``r_l = mlp(ln_2(y_l))`` on a small prompt
set and reports how much residual energy is preserved by rank-k SVD.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "The weather residual graph shows",
    "A low pressure system moved across the coast",
    "The model predicts rainfall from graph residuals",
    "Temperature anomalies can be decomposed into regional modes",
    "A transformer block stores syntax and semantics in residual streams",
    "Low rank compression works when activations lie near a subspace",
    "The quick experiment compares speed memory and accuracy",
    "Climate data often contains repeated seasonal structure",
    "Neural networks reuse patterns across many positions",
    "Forecast uncertainty rises when the pressure gradient changes",
]


def parse_ranks(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def energy_curve(x: torch.Tensor, ranks: list[int]) -> dict[int, float]:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return {rank: 1.0 for rank in ranks}
    singular_values = torch.linalg.svdvals(x)
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-12)
    cumsum = energy.cumsum(dim=0) / total
    out: dict[int, float] = {}
    for rank in ranks:
        idx = min(max(int(rank), 1), int(cumsum.numel())) - 1
        out[int(rank)] = float(cumsum[idx].item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--ranks", default="8,16,32,64,128,256,384")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ranks = parse_ranks(args.ranks)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    residuals: dict[int, list[torch.Tensor]] = defaultdict(list)
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            residuals[layer_idx].append(out.detach().cpu().reshape(-1, out.shape[-1]))

        return hook

    for idx, block in enumerate(model.transformer.h):
        hooks.append(block.mlp.register_forward_hook(make_hook(idx)))

    corpus = ("\n".join(DEFAULT_PROMPTS) + "\n") * int(max(1, args.repeat))
    ids = tokenizer(corpus, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    max_length = int(max(2, args.max_length))
    chunks = []
    for start in range(0, max(0, ids.numel() - max_length + 1), max_length):
        chunks.append(ids[start : start + max_length])
        if len(chunks) >= int(max(1, args.max_chunks)):
            break
    if not chunks:
        chunks = [ids[:max_length]]
    input_ids = torch.stack(chunks, dim=0).to(device)
    encoded = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    with torch.no_grad():
        _ = model(**encoded, use_cache=False)

    for hook in hooks:
        hook.remove()

    print(f"model={args.model} layers={len(model.transformer.h)} hidden={model.config.n_embd}")
    print(
        f"prompts={len(DEFAULT_PROMPTS)} repeat={args.repeat} chunks={len(chunks)} "
        f"max_length={args.max_length}"
    )
    print("layer samples " + " ".join(f"r{rank}" for rank in ranks))

    per_rank = {rank: [] for rank in ranks}
    all_residuals = []
    for idx in range(len(model.transformer.h)):
        x = torch.cat(residuals[idx], dim=0)
        all_residuals.append(x)
        curve = energy_curve(x, ranks)
        for rank in ranks:
            per_rank[rank].append(curve[rank])
        vals = " ".join(f"{curve[rank]:.4f}" for rank in ranks)
        print(f"{idx:02d} {x.shape[0]:7d} {vals}")

    print("avg " + " ".join(f"{sum(per_rank[rank]) / len(per_rank[rank]):.4f}" for rank in ranks))
    global_curve = energy_curve(torch.cat(all_residuals, dim=0), ranks)
    print("global " + " ".join(f"{global_curve[rank]:.4f}" for rank in ranks))


if __name__ == "__main__":
    main()
