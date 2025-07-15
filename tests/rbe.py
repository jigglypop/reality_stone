import os
import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from reality_stone.layers import EquivalentHyperbolicLinear, RBELinear
from reality_stone.layers.rbe import calculate_compression_stats, encode_model_to_seeds
import time
import sys
from tqdm import tqdm
import ctypes
import numpy as np
import copy

def get_model_size(model):
    """모델의 파라미터 크기를 계산합니다 (MB 단위)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def save_model(model, tokenizer, save_path):
    """모델의 state_dict와 tokenizer, config를 저장합니다."""
    os.makedirs(save_path, exist_ok=True)
    
    # 모델 저장 (가중치, 설정, 토크나이저 등)
    model.save_pretrained(save_path, safe_serialization=True)
    if tokenizer:
        tokenizer.save_pretrained(save_path)
    
    # 디스크 크기 계산
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(save_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    return total_size / 1024 / 1024  # MB 단위

def convert_to_equivalent_hyperbolic(model: nn.Module, c: float = 1.0):
    """모델의 모든 선형 레이어를 EquivalentHyperbolicLinear로 교체합니다."""
    # 먼저 모든 변환 대상 레이어를 찾음
    layers_to_convert = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or 'Conv1D' in str(type(module)):
            layers_to_convert.append((name, module))
    
    # tqdm으로 진행 상황 표시
    for name, module in tqdm(layers_to_convert, desc="Converting to Hyperbolic"):
        # 부모 모듈과 속성명 찾기
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # EquivalentHyperbolicLinear로 변환
        equiv_layer = EquivalentHyperbolicLinear.from_linear(module, c=c)
        setattr(parent, parts[-1], equiv_layer)
    
    return len(layers_to_convert)

def convert_to_rbe(model, use_fast_compression=True):
    num_converted = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, RBELinear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # 레이어 크기에 따른 블록 크기 결정
            in_features = module.in_features
            out_features = module.out_features
            total_params = in_features * out_features
            
            # 레이어 크기에 따른 블록 크기 조정
            if total_params > 1_000_000:
                block_size = 32  # 대형 레이어는 작은 블록
            elif total_params > 100_000:
                block_size = 64  # 중형 레이어
            else:
                block_size = 128  # 소형 레이어는 큰 블록
                
            # RBELinear로 교체 - 빠른 압축 모드 사용
            rbe_layer = RBELinear(
                in_features, 
                out_features, 
                bias=module.bias is not None, 
                block_size=block_size,
                use_fast_compression=use_fast_compression
            )
            
            # 블록 정보 출력
            if hasattr(rbe_layer, 'block_info') and rbe_layer.block_info:
                info = rbe_layer.block_info
                num_blocks = info['out_blocks'] * info['in_blocks']
                print(f"   Block size: {info['block_size']}×{info['block_size']}")
                print(f"   Number of blocks: {num_blocks}")
            
            # 기존 가중치와 편향 복사는 이미 __init__에서 처리됨
            if module.bias is not None:
                rbe_layer.bias.data = module.bias.data.clone()
                
            setattr(parent, child_name, rbe_layer)
            num_converted += 1
            
            # RMSE 계산 (빠른 압축 모드에서는 근사치)
            if use_fast_compression:
                print(f"   Fast compression mode enabled")
            else:
                rmse = rbe_layer.get_rmse()
                print(f"   RMSE: {rmse:.6f}")
    
    return num_converted

def benchmark_generation(model, tokenizer, prompts, num_runs=3):
    """생성 속도를 벤치마크합니다"""
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    # 환경 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 모델 로드
    print("Loading model...")
    try:
        config = AutoConfig.from_pretrained('hyperbolic_model/')
        tokenizer = AutoTokenizer.from_pretrained('hyperbolic_model/')
        original_model = AutoModelForCausalLM.from_pretrained(
            'hyperbolic_model/',
            torch_dtype=torch.float32  # float16 대신 float32 사용
        )
        if torch.cuda.is_available():
            original_model = original_model.cuda()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 1. 원본 모델 테스트 (생략 가능)
    print("\n📊 Original model info:")
    total_params = sum(p.numel() for p in original_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 2. 빠른 RBE 압축
    print("\n🔄 Converting to RBE with fast compression...")
    start_time = time.time()
    
    # 모델을 CPU로 이동하여 압축
    rbe_model = copy.deepcopy(original_model).cpu()
    
    # Linear 레이어를 직접 찾아서 변환
    linear_count = 0
    for name, module in rbe_model.named_modules():
        if isinstance(module, nn.Linear):
            linear_count += 1
    
    print(f"Found {linear_count} Linear layers to compress")
    
    # 진행률 표시와 함께 변환
    converted = 0
    for name, module in tqdm(list(rbe_model.named_modules()), desc="Compressing layers"):
        if isinstance(module, nn.Linear) and not isinstance(module, RBELinear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = rbe_model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # 레이어 크기에 따른 블록 크기 결정
            total_params = module.in_features * module.out_features
            if total_params > 1_000_000:
                block_size = 32
            elif total_params > 100_000:
                block_size = 64
            else:
                block_size = 128
            
            # RBELinear로 교체
            rbe_layer = RBELinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                block_size=block_size,
                use_fast_compression=True  # 빠른 압축 사용
            )
            
            # 편향 복사
            if module.bias is not None:
                rbe_layer.bias.data = module.bias.data.clone()
            
            setattr(parent, child_name, rbe_layer)
            converted += 1
    
    compression_time = time.time() - start_time
    print(f"\n✅ Converted {converted} layers in {compression_time:.1f}s")
    print(f"Average time per layer: {compression_time/converted:.2f}s")
    
    # 3. 압축 통계
    stats = calculate_compression_stats(rbe_model)
    print(f"\n📊 Compression Statistics:")
    print(f"Original parameters: {stats['total_params']:,}")
    print(f"Compressed size: {stats['compressed_params']:,} seeds")
    print(f"Overall compression ratio: {stats['compression_ratio']:.0f}:1")
    
    # 4. GPU로 이동하여 생성 테스트
    print("\n🚀 Testing compressed model generation...")
    rbe_model = rbe_model.cuda()
    
    test_prompts = [
        "The meaning of life is",
        "인공지능의 미래는",
        "Once upon a time",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = rbe_model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
    
    print("\n✅ Test completed!")
    
    # 5. 모델 저장 (선택적)
    save_compressed = input("\nSave compressed model? (y/n): ")
    if save_compressed.lower() == 'y':
        save_path = 'compressed_rbe_model'
        os.makedirs(save_path, exist_ok=True)
        
        # 시드 정보 저장
        seeds_dict = encode_model_to_seeds(rbe_model)
        torch.save(seeds_dict, f'{save_path}/rbe_seeds.pt')
        
        # 설정 저장
        config.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}/")
        print(f"Seeds file size: {os.path.getsize(f'{save_path}/rbe_seeds.pt') / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main() 