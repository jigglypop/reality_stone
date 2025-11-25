import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.conversion import convert_to_full_riemannian
from reality_stone.metric_extraction import extract_riemannian_metric


def main():
    parser = argparse.ArgumentParser(description="Teacher 리만 변환 -> 메트릭 추출 -> 저장")
    parser.add_argument("--teacher", type=str, default="microsoft/phi-2")
    parser.add_argument("--target_dim", type=int, default=64)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/phi2-riemannian")
    parser.add_argument("--cache_dir", type=str, default="E:/hf-cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/5] Teacher 로드: {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir
    )
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"   Original params: {orig_params:,}")

    print(f"[2/5] 리만 기하학 변환 (curvature={args.curvature})...")
    model = convert_to_full_riemannian(model, curvature=args.curvature)

    print(f"[3/5] 리만 메트릭 추출 (dim={args.target_dim})...")
    model = extract_riemannian_metric(
        model,
        target_dim=args.target_dim,
        calibration_data=None,
        num_steps=1,
        curvature=args.curvature,
        lr=0.0
    )

    new_params = sum(p.numel() for p in model.parameters())
    reduction = (1 - new_params / orig_params) * 100
    print(f"   Final params: {new_params:,}")
    print(f"   Reduction: {reduction:.2f}%")

    print(f"[4/5] 저장: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print(f"[5/5] 생성 테스트...")
    model.to(device)
    model.eval()

    prompts = [
        "The theory of relativity",
        "Artificial intelligence will",
        "The fundamental laws of physics"
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        print(f"\n[{prompt}]")
        print(tokenizer.decode(out[0], skip_special_tokens=True))

    print(f"\nDone. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
