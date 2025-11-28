#!/usr/bin/env python3
"""
RS-ULF 한국어 챗 모듈

- finetune된 HF LM + RS-ULF 체크포인트를 사용해
  한글 질문에 대해 RS-ULF 경로로만 응답합니다.

사용 예시:
    uv run scripts/rsulf_ko_chat.py \
      --model_dir checkpoints/rsulf_lm_finetuned \
      --rsulf_ckpt checkpoints/rsulf_metric_opt \
      --device cuda
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import reality_stone as rs  # noqa: F401
from reality_stone.models.transformer_converter import (
    load_rsulf_model_checkpoint,
    rsulf_generate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="finetune된 HF 모델 디렉토리")
    parser.add_argument("--rsulf_ckpt", required=True, help="RS-ULF 체크포인트 디렉토리")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    print("\n" + "=" * 70)
    print("RS-ULF 한국어 챗 모듈")
    print("=" * 70)

    # 1. finetuned HF 모델 + 토크나이저 로드
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # pad_token 설정
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(torch_device)

    # 2. RS-ULF 체크포인트 로드
    rs_model = load_rsulf_model_checkpoint(args.rsulf_ckpt)

    print(f"  모델 디렉토리: {model_dir}")
    print(f"  RS-ULF 체크포인트: {args.rsulf_ckpt}")
    print(f"  디바이스: {torch_device}")

    print("\n한글 질문을 입력하세요. (종료: 빈 줄 또는 Ctrl+C)")

    while True:
        try:
            prompt = input("\n[질문] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not prompt:
            print("종료합니다.")
            break

        try:
            answer = rsulf_generate(
                model,
                rs_model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            print(f"[RS-ULF] {answer}")
        except Exception as e:
            print(f"[오류] RS-ULF 응답 생성 실패: {e}")


if __name__ == "__main__":
    main()


