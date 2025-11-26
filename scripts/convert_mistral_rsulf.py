import os
import argparse
from pathlib import Path

import torch

from reality_stone.models.llm_adapter import MistralRSULFAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--save_dir", type=str, default="checkpoints/mistral-rsulf-7b")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--hf_cache", type=str, default=os.environ.get("HF_HOME", "E:/hf-cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prompt", type=str, default="리얼리티 스톤 RS-ULF 모델의 핵심 아이디어를 한국어로 설명해줘.")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.98)
    args = parser.parse_args()

    cache_dir = Path(args.hf_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    adapter = MistralRSULFAdapter(
        model_name=args.model_id,
        device=device,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    base_params = sum(p.numel() for p in adapter.base.parameters())
    rsulf_params = sum(p.numel() for p in adapter.rsulf_stack.parameters())
    total_params = base_params + rsulf_params
    print("base params:", base_params)
    print("rsulf params:", rsulf_params)
    print("total params:", total_params)

    text = adapter.generate(args.prompt, max_length=args.max_length)
    print("prompt:", args.prompt)
    print("generated:", text)

    save_path = Path(args.save_dir)
    adapter.save_pretrained(save_path)
    print("saved rsulf-converted mistral to", str(save_path))


if __name__ == "__main__":
    main()


