#!/usr/bin/env python3
"""
RS-ULF LLM lm_head 미세조정 스크립트

- 이미 변환/최적화된 RS-ULF 체크포인트(checkpoints/rsulf_... )와
  HF Transformer 가중치를 사용해,
  RS-ULF hidden state 위에서 lm_head만 LM loss로 학습합니다.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from transformers import AutoModelForCausalLM, AutoTokenizer

import reality_stone as rs  # noqa: F401
from reality_stone.models.transformer_converter import (
    load_rsulf_model_checkpoint,
    finetune_rsulf_lm_head,
    cache_rsulf_hidden_states,
    finetune_lm_head_from_cache,
)


class TextFileDataset(Dataset):
    """
    간단한 텍스트 파일 기반 데이터셋.
    - 한 줄당 하나의 시퀀스.
    """

    def __init__(self, path: str, tokenizer, max_len: int = 128):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.lines[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default="E:/hf-cache")
    parser.add_argument("--checkpoint_dir", required=True, help="RS-ULF 체크포인트 디렉토리")
    parser.add_argument("--text_path", required=True, help="학습용 텍스트 파일 (한 줄당 한 문장)")
    parser.add_argument("--output_dir", default="checkpoints/rsulf_lm_finetuned")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_cache", action="store_true", help="캐싱 모드: 1회 캐시 후 빠른 학습")
    parser.add_argument("--cache_path", default="data/rsulf_hidden_cache.pt", help="캐시 파일 경로")
    parser.add_argument("--cache_samples", type=int, default=10000, help="캐시할 샘플 수")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    print("\n" + "=" * 70)
    print("1. HF 모델 + 토크나이저 로딩")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    # pad_token 설정 (필수)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    print(f"  모델: {args.model_name}")

    print("\n" + "=" * 70)
    print("2. RS-ULF 체크포인트 로딩")
    print("=" * 70)

    rs_model = load_rsulf_model_checkpoint(args.checkpoint_dir)
    print(f"  RS-ULF 레이어 수: {len(rs_model.layers)}")

    print("\n" + "=" * 70)
    print("3. 학습 데이터 로딩")
    print("=" * 70)

    dataset = TextFileDataset(args.text_path, tokenizer, max_len=args.max_len)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  샘플 수: {len(dataset)}")

    print("\n" + "=" * 70)
    print("4. RS-ULF lm_head 미세조정")
    print("=" * 70)

    if args.use_cache:
        cache_file = args.cache_path
        if not os.path.exists(cache_file):
            print(f"  [캐시 생성 중] {cache_file}")
            cache_rsulf_hidden_states(
                model,
                rs_model,
                train_loader,
                cache_path=cache_file,
                device=device,
                max_samples=args.cache_samples,
            )
        else:
            print(f"  [캐시 로드] {cache_file}")

        finetune_lm_head_from_cache(
            model,
            cache_path=cache_file,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
    else:
        finetune_rsulf_lm_head(
            model,
            rs_model,
            tokenizer,
            train_loader,
            num_steps=args.num_steps,
            lr=args.lr,
            device=device,
        )

    print("\n" + "=" * 70)
    print("5. 결과 저장")
    print("=" * 70)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"  finetuned HF 모델 저장: {out_dir}")
    print("  (RS-ULF 체크포인트는 기존 디렉토리에서 그대로 사용)")


if __name__ == "__main__":
    main()


