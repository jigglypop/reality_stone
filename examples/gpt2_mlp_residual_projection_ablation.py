"""Ablate GPT-2 generation after projecting MLP residuals to rank-k subspaces.

This is a quality gate for residual compression. It still computes the original
MLP, then projects its residual output, so it is not a speed benchmark.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2_mlp_residual_svd import DEFAULT_PROMPTS


class ResidualProjectionMLP(nn.Module):
    def __init__(self, mlp: nn.Module, mean: torch.Tensor, basis: torch.Tensor):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("mean", mean.detach().float(), persistent=False)
        self.register_buffer("basis", basis.detach().float(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.mlp(hidden_states)
        dtype = out.dtype
        flat = out.float().reshape(-1, out.shape[-1])
        centered = flat - self.mean
        projected = centered @ self.basis.T @ self.basis + self.mean
        return projected.to(dtype=dtype).reshape_as(out)


def collect_residuals(model, tokenizer, device, max_length: int, repeat: int, max_chunks: int):
    residuals: dict[int, list[torch.Tensor]] = defaultdict(list)
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            residuals[layer_idx].append(out.detach().cpu().reshape(-1, out.shape[-1]))

        return hook

    for idx, block in enumerate(model.transformer.h):
        hooks.append(block.mlp.register_forward_hook(make_hook(idx)))

    corpus = ("\n".join(DEFAULT_PROMPTS) + "\n") * int(max(1, repeat))
    ids = tokenizer(corpus, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    chunks = []
    for start in range(0, max(0, ids.numel() - max_length + 1), max_length):
        chunks.append(ids[start : start + max_length])
        if len(chunks) >= int(max(1, max_chunks)):
            break
    input_ids = torch.stack(chunks, dim=0).to(device)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)

    for hook in hooks:
        hook.remove()

    return {idx: torch.cat(parts, dim=0).float() for idx, parts in residuals.items()}


def build_bases(residuals: dict[int, torch.Tensor], rank: int):
    bases = {}
    for idx, x in residuals.items():
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        bases[idx] = (mean.squeeze(0), vh[: int(rank)].contiguous())
    return bases


def install_projection(model, bases):
    for idx, block in enumerate(model.transformer.h):
        mean, basis = bases[idx]
        block.mlp = ResidualProjectionMLP(block.mlp, mean, basis)


def generate(model, tokenizer, prompt: str, tokens: int, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The weather residual graph shows")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    baseline = generate(model, tokenizer, args.prompt, args.tokens, device)
    residuals = collect_residuals(
        model, tokenizer, device, args.max_length, args.repeat, args.max_chunks
    )
    bases = build_bases(residuals, args.rank)
    install_projection(model, bases)
    model.eval()
    projected = generate(model, tokenizer, args.prompt, args.tokens, device)

    print(f"model={args.model} rank={args.rank}")
    print(f"same={baseline == projected} prefix={prefix_len(baseline, projected)}")
    print("baseline:")
    print(baseline)
    print("projected:")
    print(projected)


if __name__ == "__main__":
    main()
