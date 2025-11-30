#!/usr/bin/env python3
"""
Transformer → RS-ULF 완전 변환 스크립트

이 스크립트는 사전 학습된 Huggingface Transformer 모델을 
Reality Stone의 Unified Latent Field (RS-ULF) 구조로 변환합니다.

주요 기능:
1. Transformer 모델 로드 (HF AutoModelForCausalLM)
2. 리만 메트릭 추출 및 안정화 (Riemannian Metric Extraction)
3. 차원 축소 (Folding) 및 기하학적 압축
4. 그래프 라플라시안(Graph Laplacian) 기반의 잠재 공간 구조화
5. RS-ULF 모델로 변환 및 저장

사용법:
    python scripts/convert_transformer_rsulf.py \
        --model_name mistralai/Mistral-7B-v0.1 \
        --save_dir checkpoints/mistral-7b-rsulf \
        --folding_ratio 0.5 \
        --device cuda
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from reality_stone.models.transformer_converter import (
    TransformerToRSULFConverter,
    cache_rsulf_hidden_states,
    finetune_lm_head_from_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Transformer to RS-ULF")
    
    # 모델 설정
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Huggingface 모델 ID 또는 경로 (기본값: 로컬 Mistral 자동 탐색)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/rsulf-converted",
        help="변환된 모델을 저장할 디렉토리"
    )
    
    # 하드웨어 설정
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="사용할 디바이스 (기본값: auto - 가능한 경우 CUDA 사용)"
    )
    
    # 변환 설정
    parser.add_argument(
        "--metric_strategy",
        type=str,
        default="diagonal",
        choices=["diagonal", "symmetric", "sym_abs"],
        help="메트릭 안정화 전략 (diagonal: 대각 성분만 사용, symmetric: 대칭화)"
    )
    parser.add_argument(
        "--folding_ratio",
        type=float,
        default=None,
        help="차원 축소 비율 (예: 0.5는 차원을 절반으로 줄임)"
    )
    
    parser.add_argument("--lr", type=float, default=0.02, help="지오데식 학습률 (Learning rate for geodesic)")
    parser.add_argument("--alpha", type=float, default=0.04, help="라플라시안 가중치 (Laplacian weight)")
    parser.add_argument("--beta", type=float, default=0.01, help="그래프 확산 가중치 (Graph diffusion weight)")
    parser.add_argument("--gamma", type=float, default=0.98, help="DP 메모리 감쇠율 (DP memory decay)")
    parser.add_argument("--graph_window", type=int, default=8, help="그래프 윈도우 크기 (Graph window size)")
    parser.add_argument("--graph_decay", type=float, default=0.9, help="그래프 엣지 감쇠율 (Graph edge decay)")
    parser.add_argument("--skip_tests", action="store_true", help="일관성 테스트 건너뛰기")
    parser.add_argument("--test_generation", action="store_true", help="변환 후 텍스트 생성 테스트 수행")
    parser.add_argument("--test_prompt", type=str, default="Reality Stone은", help="테스트용 프롬프트")
    parser.add_argument("--max_length", type=int, default=50, help="최대 생성 길이")
    parser.add_argument("--fast_mode", action="store_true", help="RS-ULF 빠른 변환 모드 (근사 SVD 사용)")
    parser.add_argument("--max_layers", type=int, default=None, help="변환할 최대 레이어 수 (디버깅/테스트용)")
    parser.add_argument("--do_finetune", action="store_true", help="RS-ULF 변환 후 lm_head 파인튜닝 수행")
    parser.add_argument("--finetune_text", type=str, default=None, help="lm_head 파인튜닝용 텍스트 파일 경로")
    parser.add_argument("--finetune_cache_file", type=str, default=None, help="RS-ULF hidden cache 파일 경로")
    parser.add_argument("--finetune_epochs", type=int, default=1, help="lm_head 파인튜닝 epoch 수")
    parser.add_argument("--finetune_batch_size", type=int, default=4, help="lm_head 파인튜닝 배치 크기")
    parser.add_argument("--finetune_lr", type=float, default=1e-4, help="lm_head 파인튜닝 학습률")
    parser.add_argument("--finetune_max_samples", type=int, default=5000, help="캐시 생성 시 최대 샘플 수")
    
    # 캐시 및 기타 설정
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("HF_HOME", "E:/hf-cache"),
        help="Huggingface 캐시 디렉토리"
    )
    parser.add_argument("--offline", action="store_true", help="오프라인 모드 (로컬 파일만 사용)")
    parser.add_argument("--quiet", action="store_true", help="출력 최소화 (Quiet mode)")
    
    return parser.parse_args()


def find_local_mistral_path() -> str:
    base = Path("data/models--mistralai--Mistral-7B-v0.1/snapshots")
    if base.exists():
        for snapshot in base.iterdir():
            if snapshot.is_dir() and (snapshot / "config.json").exists():
                return str(snapshot.resolve())
    base = Path("data/qwen-tuning-serious")
    if base.exists():
        checkpoints = sorted(
            [d for d in base.iterdir() if d.is_dir() and "checkpoint-" in d.name],
            key=lambda x: int(x.name.split("-")[-1]),
            reverse=True,
        )
        if checkpoints:
            return str(checkpoints[0].resolve())
    return "mistralai/Mistral-7B-v0.1"


class _TextLineDataset(Dataset):
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
    args = parse_args()
    
    # 캐시 설정
    os.environ.setdefault("HF_HOME", args.cache_dir)
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_name = args.model_name
    if model_name is None:
        model_name = find_local_mistral_path()
        print(f"선택된 모델: {model_name}")
    if args.device is None or args.device == "auto":
        from reality_stone.utils.misc import get_device
        device = get_device()
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if not args.quiet:
        print(f"사용 디바이스: {device}")
    # Transformer 모델 로드
    if not args.quiet:
        print(f"\n모델 로딩 중: {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else "cpu",
        cache_dir=args.cache_dir,
        local_files_only=args.offline
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.offline
    )
    
    if not args.quiet:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"모델 로드 완료: {num_params:,} 파라미터")
    
    converter_config = {
        'metric_strategy': args.metric_strategy,
        'lr': args.lr,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'folding_ratio': args.folding_ratio,
        'graph_window_size': args.graph_window,
        'graph_directed': True,
        'graph_decay': args.graph_decay,
        'run_consistency_tests': not args.skip_tests,
        'consistency_tolerance': 1e-2,
        'verbose': not args.quiet,
        'fast_mode': args.fast_mode,
        'max_layers': args.max_layers,
    }
    
    converter = TransformerToRSULFConverter(config=converter_config)
    rs_model = converter.convert_model(model, device=device)
    
    if not args.quiet:
        if hasattr(rs_model, "param_count"):
            stats = rs_model.param_count()
            rs_params = int(stats.get("compressed", 0))
        else:
            rs_params = None
        print(f"\n✓ 변환 완료!")
        if rs_params is not None and rs_params > 0:
            print(f"RS-ULF 파라미터: {rs_params:,}")
            if args.folding_ratio is not None:
                reduction = (1 - rs_params / num_params) * 100
                print(f"파라미터 감소율: {reduction:.2f}%")
    
    # 생성 테스트 (옵션)
    if args.test_generation:
        if not args.quiet:
            print(f"\n생성 테스트 수행 (프롬프트: '{args.test_prompt}')")
        
        test_generation(
            rs_model,
            model,  # 임베딩 및 lm_head 용 원본 모델
            tokenizer,
            args.test_prompt,
            args.max_length,
            device
        )

    if args.do_finetune or args.finetune_text is not None:
        if args.finetune_text is None:
            raise ValueError("finetune_text가 필요합니다")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        text_path = args.finetune_text
        dataset = _TextLineDataset(text_path, tokenizer, max_length=args.max_length)
        loader = DataLoader(dataset, batch_size=args.finetune_batch_size, shuffle=True)
        if args.finetune_cache_file is None:
            cache_path = str(Path(args.save_dir) / "rsulf_hidden_cache.pt")
        else:
            cache_path = args.finetune_cache_file
        if not os.path.exists(cache_path):
            cache_rsulf_hidden_states(
                model,
                rs_model,
                loader,
                cache_path=cache_path,
                device=device,
                max_samples=args.finetune_max_samples,
            )
        finetune_lm_head_from_cache(
            model,
            cache_path=cache_path,
            num_steps=args.finetune_epochs * 100,
            batch_size=args.finetune_batch_size,
            lr=args.finetune_lr,
            device=device,
        )
        lm_head_path = Path(args.save_dir) / "lm_head_finetuned.pt"
        torch.save(model.lm_head.state_dict(), lm_head_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    can_save_model = hasattr(rs_model, "state_dict")
    if can_save_model:
        save_dict = {
            'model_state_dict': rs_model.state_dict(),
            'converter_config': converter.config,
            'original_model_name': model_name,
            'tokenizer_name': model_name,
        }
        if hasattr(converter, "stats"):
            save_dict['conversion_stats'] = converter.stats
        save_path = save_dir / 'rsulf_model.pt'
        torch.save(save_dict, save_path)
    import json
    config_path = save_dir / 'converter_config.json'
    meta = dict(converter_config)
    meta['original_model_name'] = model_name
    with open(config_path, 'w') as f:
        json.dump(meta, f, indent=2)
    if not args.quiet:
        if can_save_model:
            print(f"\n✓ 저장 완료: {save_dir}/")
            print(f"  - rsulf_model.pt")
            print(f"  - converter_config.json")
        else:
            print(f"\n✓ 설정만 저장됨: {save_dir}/")
            print(f"  - converter_config.json")
    
    return rs_model


def test_generation(
    rs_model,
    original_model,
    tokenizer,
    prompt: str,
    max_length: int,
    device: str
):
    """
    간단한 생성 테스트 함수
    
    Note: 실제 텍스트 생성을 위해서는 원본 모델의 embedding 레이어와 lm_head가 필요합니다.
    이 함수는 RS-ULF 모델의 forward pass가 정상 작동하는지 확인하는 용도입니다.
    """
    with torch.no_grad():
        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        # 원본 모델에서 임베딩 추출
        embeds = original_model.model.embed_tokens(input_ids)
        seq_len = embeds.size(1)
        # 시퀀스 길이에 맞춰 그래프 라플라시안 업데이트
        rs_model.update_graph_laplacians(seq_len, device=device)
        # RS-ULF 순전파 (Forward)
        output, V_list = rs_model(embeds)
        print(f"  입력 형태 (Input shape): {embeds.shape}")
        print(f"  출력 형태 (Output shape): {output.shape}")
        print(f"  메모리 상태 수 (Memory states): {len(V_list)}")
        print("  ✓ Forward pass 성공")


if __name__ == "__main__":
    main()
