#!/usr/bin/env python3
"""
모델 압축 및 미세조정 (Compress and Finetune)

이 스크립트는 다음 5단계 파이프라인을 수행합니다:
1. 모델 로드
2. 교정(Calibration) 데이터 생성
3. 리만 메트릭 추출 (CUDA 최적화)
4. 초압축 (Hyper Compression) 적용
5. 미세조정 (Fine-tuning) - 선택 사항

사용법:
    python scripts/compress_and_finetune.py --model_id gpt2 --finetune
"""

import os
import argparse
import json
import torch

from reality_stone.metric_extraction import extract_riemannian_metric
from reality_stone.hyper_compression import apply_hyper_compression
from reality_stone.data import SimpleTextDataset as TextDataset
from reality_stone.utils.misc import get_device
from reality_stone.utils.training import load_model_and_tokenizer, train_model_simple, generate_text


def generate_calibration_data(tokenizer, num_samples=32, seq_len=128):
    """임의의 토큰 ID로 구성된 교정용 데이터를 생성합니다."""
    vocab_size = tokenizer.vocab_size
    random_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    return random_ids.float()

def main():
    parser = argparse.ArgumentParser(description="Compress and Finetune Model")
    
    # 모델 설정
    parser.add_argument("--model_id", type=str, default="gpt2", help="Target model ID")
    parser.add_argument("--target_dim", type=int, default=64, help="Target compression dimension")
    parser.add_argument("--save_dir", type=str, default="checkpoints/gpt2-compressed-rs", help="Output directory")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "E:/hf-cache"))
    
    # 압축 설정
    parser.add_argument("--num_steps", type=int, default=100, help="Metric extraction steps (CUDA)")
    parser.add_argument("--curvature", type=float, default=-1.0, help="Target curvature")
    parser.add_argument("--lr", type=float, default=0.01, help="Extraction learning rate")
    
    # 미세조정 설정
    parser.add_argument("--finetune", action="store_true", help="Enable fine-tuning")
    parser.add_argument("--finetune_epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--finetune_lr", type=float, default=5e-5, help="Fine-tuning learning rate")
    parser.add_argument("--data_path", type=str, default=None, help="Path to JSONL training data")
    
    args = parser.parse_args()

    # dtype 설정
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"[1/5] Loading model: {args.model_id}")
    model, tok = load_model_and_tokenizer(
        args.model_id, 
        cache_dir=args.cache_dir, 
        dtype=dtype
    )
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"   Original params: {orig_params:,}")

    print(f"[2/5] Generating calibration data...")
    calib_data = generate_calibration_data(tok, num_samples=32, seq_len=128)
    print(f"   Calibration shape: {calib_data.shape}")

    print(f"[3/5] Extracting Riemannian Metric (CUDA, {args.num_steps} steps)...")
    model = extract_riemannian_metric(
        model, 
        target_dim=args.target_dim,
        calibration_data=calib_data,
        num_steps=args.num_steps,
        curvature=args.curvature,
        lr=args.lr
    )

    print(f"[4/5] Applying Hyper Compression...")
    model = apply_hyper_compression(model, target_dim=args.target_dim)
    
    new_params = sum(p.numel() for p in model.parameters())
    reduction = (1 - new_params / orig_params) * 100
    print(f"   Compressed params: {new_params:,}")
    print(f"   Reduction: {reduction:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.finetune:
        print(f"[5/5] Fine-tuning compressed model...")
        
        if args.data_path and os.path.exists(args.data_path):
            train_texts = []
            with open(args.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "text" in obj:
                        train_texts.append(obj["text"])
                    elif "paragraph" in obj:
                        train_texts.append(obj["paragraph"])
            print(f"   Loaded {len(train_texts)} samples from {args.data_path}")
        else:
            # 더미 데이터 사용 (테스트용)
            train_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks can learn complex patterns from data.",
                "Deep learning models require large amounts of training data.",
                "Transformers have revolutionized natural language processing.",
                "Attention mechanisms allow models to focus on relevant parts.",
                "GPT models are trained on massive text corpora.",
                "Language models can generate human-like text.",
            ] * 10
            print(f"   Using {len(train_texts)} synthetic samples")
        
        device = get_device()
        train_dataset = TextDataset(train_texts, tok, max_len=256)
        model = train_model_simple(
            model, tok, train_dataset,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            device=device
        )
    else:
        print("[5/5] Skipping fine-tuning (use --finetune to enable)")

    # 저장
    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"Saved to {args.save_dir}")

    print("\n[Test] Generating sample text...")
    generate_text(model, tok, ["The future of AI is"], device=get_device())


if __name__ == "__main__":
    main()
