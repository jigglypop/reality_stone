import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.metric_extraction import extract_riemannian_metric
from reality_stone.hyper_compression import apply_hyper_compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--target_dim", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="checkpoints/mistral-7b-compressed-rs")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--offline", action="store_true", help="Use local cache only (no downloads)")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "E:/hf-cache"))
    args = parser.parse_args()

    # Cache and offline guard
    os.environ.setdefault("HF_HOME", args.cache_dir)
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=args.offline)
    if getattr(tok, "pad_token", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if hasattr(model, "resize_token_embeddings"):
        try:
            model.resize_token_embeddings(len(tok))
        except Exception:
            pass

    model = extract_riemannian_metric(model, target_dim=args.target_dim)
    model = apply_hyper_compression(model, target_dim=args.target_dim)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"Saved compressed model to {args.save_dir}")


if __name__ == "__main__":
    main()


