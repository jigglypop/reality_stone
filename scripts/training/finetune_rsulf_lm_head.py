#!/usr/bin/env python3
"""
RS-ULF LLM lm_head 미세조정 스크립트

이 스크립트는 변환/최적화된 RS-ULF 체크포인트와 HF Transformer 가중치를 사용하여,
RS-ULF의 잠재 상태(Hidden State) 위에서 언어 모델 헤드(lm_head)만을 미세조정합니다.

주요 기능:
1. HF 모델 및 RS-ULF 모델 로드
2. 텍스트 데이터셋 로드
3. RS-ULF forward pass 결과 캐싱 (선택 사항, 학습 속도 향상)
4. lm_head 미세조정 학습 (Loss: CrossEntropy)
5. 결과 모델 저장

사용법:
    python scripts/training/finetune_rsulf_lm_head.py \
        --checkpoint_dir checkpoints/rsulf-converted \
        --text_path data/train_corpus.txt \
        --use_cache
"""

import os
import argparse
import sys
from pathlib import Path

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from reality_stone.models.transformer_converter import (
    load_rsulf_model_checkpoint,
    cache_rsulf_hidden_states,
    finetune_lm_head_from_cache,
    finetune_rsulf_lm_head,
)
from reality_stone.utils.misc import get_device


class TextLineDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 10:
                    self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def main():
    parser = argparse.ArgumentParser(description="RS-ULF lm_head Fine-tuning")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="RS-ULF checkpoint directory (layer_*.npz)")
    parser.add_argument("--hf_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Original HF model ID")
    parser.add_argument("--text_path", type=str, required=True, help="Text corpus file path (line by line)")
    parser.add_argument("--use_cache", action="store_true", help="Enable hidden state caching")
    parser.add_argument("--cache_file", type=str, default="data/rsulf_hidden_cache.pt", help="Cache file path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.device is None:
        device = get_device()
    else:
        device = args.device
        
    print(f"Using device: {device}")

    # 1. 모델 로드
    print(f"Loading original model: {args.hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else "cpu",
    )
    
    print(f"Loading RS-ULF checkpoint from: {args.checkpoint_dir}")
    rs_model = load_rsulf_model_checkpoint(args.checkpoint_dir, hf_model=hf_model, device=device)
    
    # 2. 데이터셋 로드
    print(f"Loading dataset: {args.text_path}")
    dataset = TextLineDataset(args.text_path, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. 학습 (캐시 사용 여부 분기)
    if args.use_cache:
        if not os.path.exists(args.cache_file):
            print("Generating cache...")
            cache_rsulf_hidden_states(
                hf_model, rs_model, loader, 
                cache_path=args.cache_file, 
                device=device,
                max_samples=5000 # 데모용 제한
            )
        
        print("Fine-tuning from cache...")
        finetune_lm_head_from_cache(
            hf_model,
            cache_path=args.cache_file,
            num_steps=args.epochs * 100, # 간단한 스텝 계산
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )
    else:
        print("Fine-tuning end-to-end (no cache)...")
        finetune_rsulf_lm_head(
            hf_model,
            rs_model,
            tokenizer,
            loader,
            num_steps=args.epochs * 100,
            lr=args.lr,
            device=device
        )
        
    # 4. 저장
    save_path = os.path.join(args.checkpoint_dir, "lm_head_finetuned.pt")
    torch.save(hf_model.lm_head.state_dict(), save_path)
    print(f"Saved finetuned lm_head to {save_path}")


if __name__ == "__main__":
    main()
