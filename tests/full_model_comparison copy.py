import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.layers import EquivalentHyperbolicLinear
import time
import os
import psutil
import gc

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
    """모델을 저장하고 디스크 크기를 반환합니다"""
    model.save_pretrained(save_path)
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
    conversion_count = 0
    
    for name, module in model.named_children():
        # 재귀적으로 하위 모듈 탐색
        if len(list(module.children())) > 0:
            sub_count = convert_to_equivalent_hyperbolic(module, c=c)
            conversion_count += sub_count
        
        # Conv1D와 Linear 레이어를 교체
        if isinstance(module, nn.Linear) or 'Conv1D' in str(type(module)):
            equiv_layer = EquivalentHyperbolicLinear.from_linear(module, c=c)
            setattr(model, name, equiv_layer)
            conversion_count += 1
    
    return conversion_count

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
    print("="*70)
    print("Full Model Conversion and Comparison Test")
    print("="*70)
    
    # 1. 원본 모델 로드
    print("\n📥 Loading original KoGPT-2 model...")
    model_name = "skt/kogpt2-base-v2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    
    # 메모리 사용량 측정을 위한 초기화
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. 원본 모델 분석
    print("\n📊 Original Model Analysis:")
    original_size = get_model_size(original_model)
    print(f"  Memory size: {original_size:.2f} MB")
    
    # 원본 모델 저장
    print("  Saving original model...")
    original_disk_size = save_model(original_model, tokenizer, "./original_model")
    print(f"  Disk size: {original_disk_size:.2f} MB")
    
    # 테스트 프롬프트
    test_prompts = [
        "안녕하세요, 오늘 날씨가",
        "인공지능의 발전은",
        "한국의 전통 문화는",
        "미래의 기술은",
        "자연과 환경을"
    ]
    
    # 원본 모델 벤치마크
    print("\n⏱️  Benchmarking original model...")
    original_time = benchmark_generation(original_model, tokenizer, test_prompts)
    print(f"  Average generation time: {original_time:.3f} seconds")
    
    # 3. 모델 변환
    print("\n🔄 Converting to EquivalentHyperbolicLinear...")
    start_conversion = time.time()
    
    # 변환을 위한 모델 복사 (원본 보존)
    converted_model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    
    # 변환 수행
    num_converted = convert_to_equivalent_hyperbolic(converted_model, c=1.0)
    
    conversion_time = time.time() - start_conversion
    print(f"  Converted {num_converted} layers in {conversion_time:.2f} seconds")
    
    # 4. 변환된 모델 분석
    print("\n📊 Converted Model Analysis:")
    converted_size = get_model_size(converted_model)
    print(f"  Memory size: {converted_size:.2f} MB")
    
    # 변환된 모델 저장
    print("  Saving converted model...")
    converted_disk_size = save_model(converted_model, tokenizer, "./hyperbolic_model")
    print(f"  Disk size: {converted_disk_size:.2f} MB")
    
    # 변환된 모델 벤치마크
    print("\n⏱️  Benchmarking converted model...")
    converted_time = benchmark_generation(converted_model, tokenizer, test_prompts)
    print(f"  Average generation time: {converted_time:.3f} seconds")
    
    # 5. 정확도 비교
    print("\n🎯 Accuracy Comparison:")
    accuracy_prompts = ["안녕하세요", "오늘은", "인공지능"]
    
    matches = 0
    total = len(accuracy_prompts)
    
    for prompt in accuracy_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            # 원본 모델
            orig_out = original_model.generate(
                inputs.input_ids, max_length=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            orig_text = tokenizer.decode(orig_out[0], skip_special_tokens=True)
            
            # 변환된 모델
            conv_out = converted_model.generate(
                inputs.input_ids, max_length=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            conv_text = tokenizer.decode(conv_out[0], skip_special_tokens=True)
            
            if orig_text == conv_text:
                matches += 1
                print(f"  ✅ '{prompt}' -> Identical output")
            else:
                print(f"  ❌ '{prompt}' -> Different output")
    
    accuracy = matches / total * 100
    print(f"\n  Accuracy: {accuracy:.1f}% ({matches}/{total} identical)")
    
    # 6. 최종 비교 요약
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON SUMMARY")
    print("="*70)
    
    print("\n🗄️ Storage Comparison:")
    print(f"  Original model:")
    print(f"    - Memory: {original_size:.2f} MB")
    print(f"    - Disk: {original_disk_size:.2f} MB")
    print(f"  Hyperbolic model:")
    print(f"    - Memory: {converted_size:.2f} MB")
    print(f"    - Disk: {converted_disk_size:.2f} MB")
    print(f"  Memory ratio: {converted_size/original_size:.2%}")
    print(f"  Disk ratio: {converted_disk_size/original_disk_size:.2%}")
    
    print("\n⚡ Speed Comparison:")
    print(f"  Original: {original_time:.3f}s")
    print(f"  Hyperbolic: {converted_time:.3f}s")
    print(f"  Speed ratio: {converted_time/original_time:.2f}x")
    
    print("\n🎯 Accuracy:")
    print(f"  {accuracy:.1f}% identical outputs")
    
    # 7. 결론
    print("\n" + "="*70)
    if accuracy >= 90 and converted_disk_size <= original_disk_size * 1.1:
        print("✅ SUCCESS: EquivalentHyperbolicLinear maintains accuracy")
        print("   with comparable storage requirements!")
    else:
        print("⚠️ WARNING: Some trade-offs detected")
    print("="*70)
    
    # 정리
    print("\n🧹 Cleaning up saved models...")
    import shutil
    shutil.rmtree("./original_model", ignore_errors=True)
    shutil.rmtree("./hyperbolic_model", ignore_errors=True)
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    main() 