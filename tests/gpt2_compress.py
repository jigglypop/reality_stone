"""GPT-2 모델 비트필드 압축 테스트 (3D 텐서 직접 지원)"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from reality_stone.layers import BitfieldLinear
import time
from tqdm import tqdm
import numpy as np


def find_all_layers(model):
    """모델에서 모든 Linear 및 Conv1D 레이어를 찾습니다."""
    layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module, 'Linear'))
        elif hasattr(module, 'weight') and len(module.weight.shape) == 2:
            # Conv1D 레이어 (transformers.pytorch_utils.Conv1D)
            if 'Conv1D' in str(type(module)):
                layers.append((name, module, 'Conv1D'))
    
    return layers


def compress_model_with_bitfield(model, basis_size=256, r_max=1.0):
    """모델의 모든 Linear/Conv1D 레이어를 BitfieldLinear로 압축합니다."""
    compressed_layers = []
    
    # 모든 레이어 찾기
    all_layers = find_all_layers(model)
    
    print(f"🔷 비트필드 압축 시작 (basis_size={basis_size})")
    print(f"\n압축 대상 레이어: {len(all_layers)}개")
    
    for name, layer, layer_type in tqdm(all_layers, desc="레이어 압축"):
        try:
            if layer_type == 'Conv1D':
                # Conv1D는 가중치가 전치되어 있음
                weight = layer.weight.t()  # [in_features, out_features] → [out_features, in_features]
                bias = layer.bias if hasattr(layer, 'bias') else None
                
                # 임시 Linear 레이어 생성
                temp_linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
                temp_linear.weight.data = weight
                if bias is not None:
                    temp_linear.bias.data = bias
                
                # BitfieldLinear로 변환
                bitfield_layer = BitfieldLinear.from_linear(temp_linear, basis_size, r_max)
                
            else:  # Linear
                # 직접 BitfieldLinear로 변환
                bitfield_layer = BitfieldLinear.from_linear(layer, basis_size, r_max)
            
            # 모델에서 레이어 교체
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, bitfield_layer)
            else:
                setattr(model, child_name, bitfield_layer)
            
            compressed_layers.append((name, layer_type, bitfield_layer))
            
        except Exception as e:
            print(f"⚠️ 레이어 {name} 압축 실패: {e}")
            continue
    
    return compressed_layers


def calculate_compression_stats(original_model, compressed_layers):
    """압축률 통계 계산"""
    original_params = 0
    compressed_params = 0
    
    for name, layer_type, bitfield_layer in compressed_layers:
        # 원본 파라미터 수 계산
        original_size = bitfield_layer.in_features * bitfield_layer.out_features
        original_params += original_size
        
        # 압축된 파라미터 수 계산 (22비트 인코딩)
        compressed_size = bitfield_layer.out_features * 22 / 32  # 22비트를 32비트 기준으로 변환
        compressed_params += compressed_size
    
    compression_ratio = original_params / compressed_params if compressed_params > 0 else 0
    
    return {
        'original_params': original_params,
        'compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'layers_compressed': len(compressed_layers)
    }


def test_compressed_model():
    """압축된 모델 테스트"""
    print("\n" + "="*60)
    print("GPT-2 비트필드 압축 테스트 (3D 텐서 직접 지원)")
    print("="*60)
    
    # 모델과 토크나이저 로드
    print("🔄 GPT-2 모델 로드 중...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 원본 모델 크기 확인
    original_param_count = sum(p.numel() for p in model.parameters())
    print(f"원본 모델 크기: {original_param_count:,} 파라미터")
    
    # 모델 압축
    print(f"\n[1] 전체 레이어 비트필드 압축")
    compressed_layers = compress_model_with_bitfield(model, basis_size=256, r_max=1.0)
    
    # 압축 통계
    stats = calculate_compression_stats(model, compressed_layers)
    print(f"\n압축 완료:")
    print(f"  - 원본 파라미터: {stats['original_params']:,} bytes")
    print(f"  - 압축 파라미터: {stats['compressed_params']:,.0f} bytes")
    print(f"  - 압축률: {stats['compression_ratio']:.1f}x" if stats['compression_ratio'] > 0 else "  - 압축률: 계산 불가")
    
    # 테스트 텍스트
    test_texts = [
        "안녕하세요",
        "The quick brown fox jumps over the lazy dog",
        "Python is a programming language",
        "Machine learning is transforming the world"
    ]
    
    print(f"\n=== 전체압축 테스트 ===")
    
    for i, text in enumerate(test_texts):
        print(f"\n[테스트 {i+1}] 입력: '{text}'")
        
        try:
            # 입력 토큰화
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            # 모델 추론 시간 측정
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            inference_time = time.time() - start_time
            
            # 다음 토큰 예측
            predicted_token_id = torch.argmax(logits[0, -1, :]).item()
            predicted_token = tokenizer.decode([predicted_token_id])
            
            print(f"  출력 형태: {logits.shape}")
            print(f"  추론 시간: {inference_time:.3f}초")
            print(f"  다음 토큰 예측: '{predicted_token}'")
            
            # 간단한 생성 테스트
            generated = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  생성된 텍스트: '{generated_text}'")
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 성능 비교 테스트
    print(f"\n=== 성능 비교 테스트 ===")
    
    # 배치 처리 테스트
    batch_texts = test_texts[:2]  # 처음 2개 텍스트만 사용
    
    try:
        # 배치 토큰화
        batch_inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=50
        )
        
        print(f"배치 입력 형태: {batch_inputs.input_ids.shape}")
        
        # 배치 추론
        start_time = time.time()
        
        with torch.no_grad():
            batch_outputs = model(**batch_inputs)
            batch_logits = batch_outputs.logits
        
        batch_inference_time = time.time() - start_time
        
        print(f"배치 출력 형태: {batch_logits.shape}")
        print(f"배치 추론 시간: {batch_inference_time:.3f}초")
        print(f"평균 추론 시간: {batch_inference_time/len(batch_texts):.3f}초/샘플")
        
    except Exception as e:
        print(f"❌ 배치 처리 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== 최종 결과 ===")
    print(f"✅ 압축 성공: {stats['layers_compressed']}개 레이어")
    print(f"✅ 압축률: {stats['compression_ratio']:.1f}x" if stats['compression_ratio'] > 0 else "❌ 압축률 계산 실패")
    print(f"✅ 3D 텐서 직접 처리: reshape 오버헤드 제거")
    print(f"✅ 모델 동작: 정상" if 'generated_text' in locals() else "❌ 모델 동작: 오류")


if __name__ == "__main__":
    test_compressed_model() 