"""GPT-2 모델 스플라인 압축 테스트"""

import torch
import torch.nn as nn
from reality_stone.layers import SplineLinear
import time
from tqdm import tqdm

def find_all_linear_layers(model):
    """모델에서 모든 Linear 레이어를 찾습니다."""
    layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))
        elif hasattr(module, 'weight') and len(module.weight.shape) == 2:
            # Conv1D 레이어 (transformers.pytorch_utils.Conv1D)
            if 'Conv1D' in str(type(module)):
                layers.append((name, module))
    
    return layers


def compress_model_with_spline(model, k=8, ignore_layers=None, tokenizer=None):
    """
    모델의 Linear 레이어를 SplineLinear로 압축합니다.
    
    Args:
        model: 압축할 모델
        k: 스플라인 세그먼트 수 (제어점 = k+1)
        ignore_layers: 압축하지 않을 레이어 이름 리스트
    """
    if ignore_layers is None:
        ignore_layers = []
        
    compressed_layers = []
    all_layers = find_all_linear_layers(model)
    
    print(f"\n🔷 스플라인 압축 시작 (k={k}, 제어점={k+1})")
    print(f"압축 대상 레이어: {len(all_layers)}개\n")
    
    # 먼저 레이어 정보를 출력해보자
    print("레이어 정보:")
    for name, layer in all_layers[:5]:  # 처음 5개만
        if 'Conv1D' in str(type(layer)):
            print(f"  - {name}: Conv1D, weight shape = {layer.weight.shape}")
        else:
            print(f"  - {name}: Linear, in={layer.in_features}, out={layer.out_features}")
    print()
    
    # 진행 상황을 위한 tqdm 설정
    pbar = tqdm(all_layers, desc="레이어 압축", ncols=120)
    
    for idx, (name, layer) in enumerate(pbar):
        if any(ignore_name in name for ignore_name in ignore_layers):
            continue
            
        try:
            # Conv1D 처리
            if 'Conv1D' in str(type(layer)):
                # GPT-2의 Conv1D 가중치 shape는 (in_features, out_features)
                weight = layer.weight
                in_features = weight.shape[0]
                out_features = weight.shape[1]
                
                # from_linear에 전달할 표준 Linear 레이어 생성
                linear_layer = nn.Linear(in_features, out_features, bias=(layer.bias is not None))
                
                # nn.Linear의 가중치 shape는 (out_features, in_features)이므로 전치(transpose) 필요
                linear_layer.weight.data = weight.t().clone()
                if layer.bias is not None:
                    linear_layer.bias.data = layer.bias.clone()
            else:
                linear_layer = layer
                in_features = layer.in_features
                out_features = layer.out_features
            
            # 작은 레이어는 압축 효과가 적으므로 건너뜀
            if in_features < 64 or out_features < 64:
                pbar.set_postfix_str(f"'{name}' 건너뜀 (크기 작음)")
                continue
            
            # SplineLinear로 변환
            pbar.set_postfix_str(f"'{name}' 압축 중... ({in_features}x{out_features})")
            
            # 압축 시작 시간
            start_time = time.time()
            
            spline_layer = SplineLinear.from_linear(
                linear_layer, 
                k=k,
                learning_rate=0.1,  # 학습률 증가
                steps=10,  # 스텝 수 대폭 감소
                use_residual=False  # residual 사용하지 않음 (압축률 향상)
            )
            
            # 압축 시간 측정
            compress_time = time.time() - start_time
            
            # 모델에서 레이어 교체
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = model.get_submodule(parent_name) if parent_name else model
            setattr(parent_module, child_name, spline_layer)
            
            compressed_layers.append((name, spline_layer))
            
            # 압축률 계산
            compression_ratio = spline_layer.get_compression_ratio()
            pbar.set_postfix_str(f"✅ '{name}' (압축률: {compression_ratio:.1f}x, {compress_time:.1f}초)")
            
            # 5개마다 중간 결과 확인
            if len(compressed_layers) % 5 == 0 and len(compressed_layers) > 0:
                print(f"\n\n📊 중간 점검 ({len(compressed_layers)}개 레이어 압축됨)")
                print(f"   - 진행률: {idx + 1}/{len(all_layers)} ({(idx + 1) / len(all_layers) * 100:.1f}%)")
                
                # 간단한 생성 테스트
                if tokenizer is not None:
                    try:
                        # 모델 타입 확인
                        model_type = getattr(model.config, 'model_type', 'gpt2')
                        is_korean = 'kogpt' in model.config._name_or_path.lower() if hasattr(model.config, '_name_or_path') else False
                        
                        test_prompt = "안녕하세요, 오늘은" if is_korean else "Hello, today is"
                        inputs = tokenizer(test_prompt, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs.input_ids,
                                max_length=30,
                                do_sample=True,
                                temperature=0.8,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print(f"   - 샘플 생성: {generated}")
                        print(f"   - 모델 타입: {model_type}, 한국어: {is_korean}")
                    except Exception as e:
                        print(f"   - 생성 테스트 실패: {e}")
                print()
            
        except Exception as e:
            pbar.set_postfix_str(f"❌ '{name}' 실패")
            print(f"\n  ❌ '{name}' 압축 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return compressed_layers


def calculate_compression_stats(compressed_layers):
    """압축률 통계 계산"""
    total_original_params = 0
    total_compressed_params = 0
    
    for name, spline_layer in compressed_layers:
        # 원본 파라미터 수
        original_params = spline_layer.in_features * spline_layer.out_features
        # 압축된 파라미터 수 (제어점)
        compressed_params = (spline_layer.k + 1) * spline_layer.in_features
        
        total_original_params += original_params
        total_compressed_params += compressed_params
        
        compression_ratio = spline_layer.get_compression_ratio()
        print(f"  - {name}: {compression_ratio:.1f}x 압축")
    
    overall_compression = total_original_params / total_compressed_params if total_compressed_params > 0 else 0
    
    return {
        'original_params': total_original_params,
        'compressed_params': total_compressed_params,
        'compression_ratio': overall_compression,
        'layers_compressed': len(compressed_layers)
    }


def test_generation_quality(model, tokenizer, test_prompts):
    """압축된 모델의 생성 품질 테스트"""
    print("\n🔍 생성 품질 테스트")
    
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\n프롬프트: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors='pt')
            
            try:
                # 생성
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"생성된 텍스트: '{generated_text}'")
            except Exception as e:
                print(f"생성 실패: {e}")


def test_inference_speed(model, tokenizer, num_iterations=10):
    """추론 속도 테스트"""
    print(f"\n⚡ 추론 속도 테스트 ({num_iterations}회 반복)")
    
    test_text = "안녕하세요. 오늘은 날씨가 정말 좋네요. " * 5
    inputs = tokenizer(test_text, return_tensors='pt', max_length=128, truncation=True)
    
    model.eval()
    with torch.no_grad():
        try:
            # 워밍업
            for _ in range(3):
                _ = model(**inputs)
            
            # 속도 측정
            start_time = time.time()
            for _ in range(num_iterations):
                _ = model(**inputs)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            print(f"평균 추론 시간: {avg_time:.2f} ms")
            return avg_time
        except Exception as e:
            print(f"추론 실패: {e}")
            return float('inf')


def main():
    print("="*60)
    print("🗜️ 모델 압축 시작")
    print("="*60)
    
    # KoGPT2 모델 로드
    print("\n📥 KoGPT2 모델 로드 중...")
    model_name = 'skt/kogpt2-base-v2'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # safetensors 형식으로 로드 시도, 없으면 일반 GPT2 사용
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"⚠️ KoGPT2 로드 실패: {e}")
        print("📥 대신 GPT2 모델을 사용합니다...")
        model_name = 'gpt2'
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 원본 모델 파라미터 수
    original_params = sum(p.numel() for p in model.parameters())
    print(f"원본 모델 파라미터 수: {original_params:,}")
    
    # 원본 모델 테스트
    print("\n--- 원본 모델 테스트 ---")
    
    # 모델에 따라 다른 프롬프트 사용
    if 'kogpt2' in model_name.lower():
        test_prompts = [
            "안녕하세요, 오늘은",
            "인공지능의 미래는",
            "한국의 전통 문화는",
        ]
    else:
        test_prompts = [
            "Hello, today is",
            "The future of AI is",
            "Once upon a time",
        ]
    
    print("\n원본 모델 생성 예시:")
    test_generation_quality(model, tokenizer, test_prompts[:1])
    
    original_speed = test_inference_speed(model, tokenizer)
    
    # 모델 압축
    print("\n" + "="*60)
    print("🗜️ 모델 압축 시작")
    print("="*60)
    
    # 임베딩 레이어는 압축하지 않음
    ignore_layers = ['wte', 'wpe', 'ln_f']
    
    # 간단한 테스트를 위해 k=8만 사용
    k = 8
    
    print(f"\n\n### k={k} 테스트 ###")
    
    # 모델 복사 (원본 유지)
    import copy
    compressed_model = copy.deepcopy(model)
    
    # 압축
    compressed_layers = compress_model_with_spline(
        compressed_model, 
        k=k,
        ignore_layers=ignore_layers,
        tokenizer=tokenizer
    )
    
    if compressed_layers:
        # 압축 통계
        print(f"\n📊 압축 통계 (k={k}):")
        stats = calculate_compression_stats(compressed_layers)
        print(f"  - 압축된 레이어 수: {stats['layers_compressed']}")
        print(f"  - 원본 파라미터: {stats['original_params']:,}")
        print(f"  - 압축 파라미터: {stats['compressed_params']:,}")
        print(f"  - 전체 압축률: {stats['compression_ratio']:.1f}x")
        
        # 압축 모델 테스트
        print(f"\n압축 모델 생성 테스트 (k={k}):")
        try:
            test_generation_quality(compressed_model, tokenizer, test_prompts)
        except Exception as e:
            print(f"생성 테스트 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n⚡ 속도 테스트...")
        try:
            compressed_speed = test_inference_speed(compressed_model, tokenizer)
            if compressed_speed != float('inf'):
                speedup = original_speed / compressed_speed
                print(f"속도 향상: {speedup:.2f}x")
        except Exception as e:
            print(f"속도 테스트 중 오류: {e}")
        
        # 메모리 사용량
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        memory_reduction = (1 - compressed_params / original_params) * 100
        print(f"메모리 절감: {memory_reduction:.1f}%")
    else:
        print("\n⚠️ 압축된 레이어가 없습니다.")
    
    print("\n\n✅ 테스트 완료!")


if __name__ == "__main__":
    main() 