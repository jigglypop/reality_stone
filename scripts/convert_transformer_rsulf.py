#!/usr/bin/env python3
"""
Transformer → RS-ULF 완전 변환 스크립트

Usage:
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from reality_stone.models.transformer_converter import (
    TransformerToRSULFConverter,
    create_graph_laplacian
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Transformer to RS-ULF")
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/rsulf-converted",
        help="Output directory"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Device (default: auto)"
    )
    
    # Conversion config
    parser.add_argument(
        "--metric_strategy",
        type=str,
        default="diagonal",
        choices=["diagonal", "symmetric", "sym_abs"],
        help="Metric stabilization strategy"
    )
    parser.add_argument(
        "--folding_ratio",
        type=float,
        default=None,
        help="Dimension folding ratio (e.g., 0.5 for half)"
    )
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate for geodesic")
    parser.add_argument("--alpha", type=float, default=0.04, help="Laplacian weight")
    parser.add_argument("--beta", type=float, default=0.01, help="Graph diffusion weight")
    parser.add_argument("--gamma", type=float, default=0.98, help="DP memory decay")
    
    # Graph
    parser.add_argument("--graph_window", type=int, default=8, help="Graph window size")
    parser.add_argument("--graph_decay", type=float, default=0.9, help="Graph edge decay")
    
    # Testing
    parser.add_argument("--skip_tests", action="store_true", help="Skip consistency tests")
    parser.add_argument("--test_generation", action="store_true", help="Test generation after conversion")
    parser.add_argument("--test_prompt", type=str, default="Reality Stone은", help="Test prompt")
    parser.add_argument("--max_length", type=int, default=50, help="Max generation length")
    
    # Cache
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("HF_HOME", "E:/hf-cache"),
        help="Huggingface cache directory"
    )
    parser.add_argument("--offline", action="store_true", help="Offline mode")
    
    # Misc
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup cache
    os.environ.setdefault("HF_HOME", args.cache_dir)
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Device
    if args.device is None or args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if not args.quiet:
        print(f"Using device: {device}")
    
    # Load Transformer model
    if not args.quiet:
        print(f"\nLoading {args.model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else "cpu",
        cache_dir=args.cache_dir,
        local_files_only=args.offline
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.offline
    )
    
    if not args.quiet:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {num_params:,} parameters")
    
    # Converter config
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
        'verbose': not args.quiet
    }
    
    # Convert
    converter = TransformerToRSULFConverter(config=converter_config)
    rs_model = converter.convert_model(model, device=device)
    
    if not args.quiet:
        rs_params = sum(p.numel() for p in rs_model.parameters())
        print(f"\n✓ Conversion complete!")
        print(f"RS-ULF parameters: {rs_params:,}")
        
        if args.folding_ratio is not None:
            reduction = (1 - rs_params / num_params) * 100
            print(f"Parameter reduction: {reduction:.2f}%")
    
    # Test generation (optional)
    if args.test_generation:
        if not args.quiet:
            print(f"\nTesting generation with prompt: '{args.test_prompt}'")
        
        test_generation(
            rs_model,
            model,  # Original for embedding/lm_head
            tokenizer,
            args.test_prompt,
            args.max_length,
            device
        )
    
    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': rs_model.state_dict(),
        'converter_config': converter.config,
        'conversion_stats': converter.stats,
        'original_model_name': args.model_name,
        'tokenizer_name': args.model_name
    }
    
    save_path = save_dir / 'rsulf_model.pt'
    torch.save(save_dict, save_path)
    
    # Also save converter config as JSON
    import json
    config_path = save_dir / 'converter_config.json'
    with open(config_path, 'w') as f:
        json.dump(converter_config, f, indent=2)
    
    if not args.quiet:
        print(f"\n✓ Saved to {save_dir}/")
        print(f"  - rsulf_model.pt")
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
    간단한 생성 테스트
    
    Note: 실제 생성을 위해서는 original model의 embedding과 lm_head 필요
    여기서는 forward pass만 테스트
    """
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        
        # Get embeddings from original model
        embeds = original_model.model.embed_tokens(input_ids)
        
        seq_len = embeds.size(1)
        
        # Update graph Laplacian for this sequence length
        rs_model.update_graph_laplacians(seq_len, device=device)
        
        # Forward through RS-ULF
        output, V_list = rs_model(embeds)
        
        print(f"  Input shape: {embeds.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Memory states: {len(V_list)}")
        print("  ✓ Forward pass successful")


if __name__ == "__main__":
    main()

