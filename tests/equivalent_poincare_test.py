import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.layers import EquivalentHyperbolicLinear, project_to_ball
import time
import numpy as np

def convert_to_equivalent_hyperbolic(model: nn.Module, c: float = 1.0):
    """
    모델의 모든 선형 레이어를 EquivalentHyperbolicLinear로 교체합니다.
    """
    for name, module in model.named_children():
        # 재귀적으로 하위 모듈 탐색
        if len(list(module.children())) > 0:
            convert_to_equivalent_hyperbolic(module, c=c)
        
        # Conv1D와 Linear 레이어를 교체
        if isinstance(module, nn.Linear) or 'Conv1D' in str(type(module)):
            equiv_layer = EquivalentHyperbolicLinear.from_linear(module, c=c)
            setattr(model, name, equiv_layer)
            print(f"✅ Replaced '{name}' with EquivalentHyperbolicLinear(c={c})")

def test_layer_equivalence():
    """레이어 변환의 동등성을 테스트합니다."""
    print("\n🧪 Testing layer equivalence...")
    
    # 테스트용 선형 레이어
    linear = nn.Linear(768, 2304)
    x = torch.randn(10, 768)
    
    # 원본 출력
    with torch.no_grad():
        original_output = linear(x)
    
    # EquivalentHyperbolicLinear로 변환
    equiv_layer = EquivalentHyperbolicLinear.from_linear(linear, c=1.0)
    
    with torch.no_grad():
        equiv_output = equiv_layer(x)
    
    # 차이 계산
    diff = torch.abs(original_output - equiv_output).mean()
    relative_diff = diff / torch.abs(original_output).mean()
    
    print(f"Original output shape: {original_output.shape}")
    print(f"Equivalent output shape: {equiv_output.shape}")
    print(f"Mean absolute difference: {diff:.6f}")
    print(f"Relative difference: {relative_diff:.4%}")
    
    # 출력이 쌍곡 공간 내에 있는지 확인
    norms = torch.norm(equiv_output, p=2, dim=-1)
    print(f"Max output norm: {norms.max():.4f} (should be < 1.0)")
    print(f"Mean output norm: {norms.mean():.4f}")
    
    return relative_diff < 0.05  # 5% 이내의 차이 허용

def main():
    # 1. 레이어 동등성 테스트
    if test_layer_equivalence():
        print("✅ Layer equivalence test passed!")
    else:
        print("❌ Layer equivalence test failed!")
    
    # 2. 모델 로드
    print("\n📥 Loading KoGPT-2 model...")
    model_name = "skt/kogpt2-base-v2"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 3. 테스트 프롬프트로 원본 모델 평가
    test_prompts = [
        "안녕하세요, 오늘",
        "인공지능의 미래는",
        "한국의 전통 음식인",
    ]
    
    print("\n📊 Testing original model...")
    original_outputs = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                do_sample=False,  # 결정적 생성
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        original_outputs.append(generated_text)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()
    
    # 4. 모델 변환
    print("\n🔄 Converting to EquivalentHyperbolicLinear...")
    start_time = time.time()
    
    # 모든 레이어 변환
    convert_to_equivalent_hyperbolic(model, c=1.0)
    
    conversion_time = time.time() - start_time
    print(f"⏰ Conversion finished in {conversion_time:.2f} seconds.")
    
    # 5. 변환된 모델 테스트
    print("\n📊 Testing converted model...")
    converted_outputs = []
    
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=20,
                    do_sample=False,  # 결정적 생성
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            converted_outputs.append(generated_text)
            
            print(f"Prompt: {prompt}")
            print(f"Original: {original_outputs[i]}")
            print(f"Converted: {generated_text}")
            
            # 동일성 확인
            if original_outputs[i] == generated_text:
                print("✅ Output is identical!")
            else:
                # 토큰 레벨에서 유사도 계산
                orig_tokens = tokenizer.encode(original_outputs[i])
                conv_tokens = tokenizer.encode(generated_text)
                
                min_len = min(len(orig_tokens), len(conv_tokens))
                if min_len > 0:
                    matching = sum(1 for a, b in zip(orig_tokens[:min_len], conv_tokens[:min_len]) if a == b)
                    similarity = matching / min_len
                    print(f"⚠️ Token similarity: {similarity:.1%}")
                else:
                    print("⚠️ Outputs differ")
            print()
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 6. 최종 평가
    print("\n" + "="*50)
    print("📊 Final Evaluation:")
    
    # 정확한 일치 비율 계산
    exact_matches = sum(1 for o, c in zip(original_outputs, converted_outputs) if o == c)
    match_rate = exact_matches / len(test_prompts) if test_prompts else 0
    
    print(f"Exact match rate: {match_rate:.1%} ({exact_matches}/{len(test_prompts)})")
    print(f"Conversion time: {conversion_time:.2f}s")
    
    if match_rate >= 0.8:  # 80% 이상 일치
        print("\n✅ SUCCESS: EquivalentHyperbolicLinear maintains accuracy!")
    else:
        print("\n⚠️ WARNING: Some outputs differ. Fine-tuning may be needed.")
    
    # 7. 메모리 사용량 비교 (선택적)
    print("\n💾 Memory usage:")
    print("Note: EquivalentHyperbolicLinear uses same memory as original")
    print("(No compression, focus on accuracy preservation)")

if __name__ == "__main__":
    main() 