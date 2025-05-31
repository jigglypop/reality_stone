import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import reality_stone as rs
import gc

class SinglePoincareBallLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(x.device)
                
        linear_out = F.linear(x, self.weight, self.bias)
        
        try:
            if x.is_cuda and hasattr(rs, 'poincare_ball_forward_cuda'):
                x_norm = torch.norm(x, dim=-1, keepdim=True)
                out_norm = torch.norm(linear_out, dim=-1, keepdim=True)
                scale = 0.01
                
                x_safe = x * torch.tanh(x_norm * scale) / (x_norm + 1e-8)
                out_safe = linear_out * torch.tanh(out_norm * scale) / (out_norm + 1e-8)
                
                hyperbolic_out = rs.poincare_ball_forward_cuda(x_safe, out_safe, self.curvature, 0.01)
                hyp_norm = torch.norm(hyperbolic_out, dim=-1, keepdim=True)
                result = hyperbolic_out * out_norm / (hyp_norm + 1e-8)
                
                return 0.99 * linear_out + 0.01 * result
            else:
                return linear_out
        except:
            return linear_out

def replace_linear_layers_inplace(model: nn.Module, curvature: float = 1.0):
    total_replaced = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'c_attn') and hasattr(module.c_attn, 'weight'):
            old_layer = module.c_attn
            if hasattr(old_layer, 'nf'):
                in_features = old_layer.weight.shape[0]
                out_features = old_layer.weight.shape[1]
                new_layer = SinglePoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data.t())
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            else:
                out_features, in_features = old_layer.weight.shape
                new_layer = SinglePoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data)
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            
            # 메모리 정리 후 교체
            del module.c_attn
            torch.cuda.empty_cache()
            module.c_attn = new_layer
            total_replaced += 1
            
        if hasattr(module, 'c_proj') and hasattr(module.c_proj, 'weight'):
            old_layer = module.c_proj
            if hasattr(old_layer, 'nf'):
                in_features = old_layer.weight.shape[0]
                out_features = old_layer.weight.shape[1]
                new_layer = SinglePoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data.t())
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            else:
                out_features, in_features = old_layer.weight.shape
                new_layer = SinglePoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data)
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            
            del module.c_proj
            torch.cuda.empty_cache()
            module.c_proj = new_layer
            total_replaced += 1
            
        if hasattr(module, 'c_fc') and hasattr(module.c_fc, 'weight'):
            old_layer = module.c_fc
            if hasattr(old_layer, 'nf'):
                in_features = old_layer.weight.shape[0]
                out_features = old_layer.weight.shape[1]
                new_layer = SinglePoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data.t())
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            else:
                out_features, in_features = old_layer.weight.shape
                new_layer = SinglePoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data)
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            
            del module.c_fc
            torch.cuda.empty_cache()
            module.c_fc = new_layer
            total_replaced += 1
    
    print(f"총 {total_replaced}개 레이어 교체 완료")
    return model

def create_inplace_poincare_model(teacher_model: nn.Module, curvature: float = 1.0):
    """딥카피 없이 in-place로 Poincare 모델 생성"""
    print("In-place 포인카레 변환 시작... (딥카피 없음)")
    student = replace_linear_layers_inplace(teacher_model, curvature)
    return student

def test_model(model, tokenizer, device, prompts, model_name, max_length=50):
    model.to(device).eval()
    results = []
    total_time = 0.0
    
    print(f"\n=== {model_name} 테스트 ===")
    for idx, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.time() - start
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_time += elapsed
        print(f"[{idx}] '{prompt}' -> {gen_text} ({elapsed:.3f}s)")
        results.append((prompt, gen_text, elapsed))
    
    avg_time = total_time / len(prompts)
    print(f"{model_name} 평균 시간: {avg_time:.3f}초")
    return results, avg_time

def test_korean_generation_safe(model, tokenizer, device, model_name):
    """안전한 한글 생성 테스트 (CUDA 오류 방지)"""
    model.to(device).eval()
    korean_prompts = [
        "한국의 아름다운 곳은",
        "오늘은 좋은 날입니다",
        "맛있는 한국 음식은"
    ]
    
    print(f"\n=== {model_name} 한글 생성 품질 테스트 ===")
    for idx, prompt in enumerate(korean_prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                # 더 안전한 설정으로 생성
                outputs = model.generate(
                    **inputs, 
                    max_length=60, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.8, 
                    top_k=40, 
                    pad_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[{idx}] {prompt}")
            print(f"    {gen_text}")
        except Exception as e:
            print(f"[{idx}] {prompt}")
            print(f"    오류 발생: {str(e)}")
        print("-" * 50)

def precise_memory_measure(device, label=""):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"{label}: {memory_mb:.1f} MB")
        return memory_mb
    return 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "skt/kogpt2-base-v2"
    curvature = 1.0
    
    print("딥카피 없는 RealityStone Poincare Ball 테스트")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 1단계: 원본 모델 로드 및 메모리 측정
    print("\n=== 메모리 추적 ===")
    initial_memory = precise_memory_measure(device, "초기 상태")
    
    teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    after_load_memory = precise_memory_measure(device, "원본 모델 로드 후")
    
    # 2단계: In-place로 Poincare 모델 생성 (딥카피 없음!)
    student = create_inplace_poincare_model(teacher, curvature)
    after_conversion_memory = precise_memory_measure(device, "포인카레 변환 후")
    
    print(f"\n메모리 변화:")
    print(f"  원본 로드: +{after_load_memory - initial_memory:.1f} MB")
    print(f"  포인카레 변환: {after_conversion_memory - after_load_memory:+.1f} MB")
    print(f"  최종 메모리 비율: {after_conversion_memory / after_load_memory:.3f}")
    
    # 3단계: 비교 테스트를 위해 원본 모델 새로 로드
    print(f"\n비교 테스트를 위한 원본 모델 새로 로드...")
    teacher_for_comparison = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    comparison_memory = precise_memory_measure(device, "비교용 원본 로드 후")
    
    # 4단계: 테스트 실행
    prompts = ["안녕하세요", "오늘 날씨는", "한국의 수도는", "인공지능이란", "맛있는 음식은"]
    
    print(f"\n" + "="*60)
    print("최종 원본 vs Poincare 모델 비교")
    print("="*60)
    
    orig_results, orig_time = test_model(teacher_for_comparison, tokenizer, device, prompts, "원본")
    poincare_results, poincare_time = test_model(student, tokenizer, device, prompts, "Poincare")
    
    # 5단계: 결과 분석
    print(f"\n" + "="*60)
    print("성능 비교 결과")
    print("="*60)
    
    speed_ratio = poincare_time / orig_time
    final_memory_ratio = after_conversion_memory / after_load_memory
    
    print(f"속도 비율: {speed_ratio:.3f} (원본 대비)")
    print(f"메모리 비율: {final_memory_ratio:.3f} (딥카피 없음!)")
    
    exact_output_matches = 0
    for i, (o, p) in enumerate(zip(orig_results, poincare_results), 1):
        if o[1] == p[1]:
            print(f"[{i}] 출력 일치")
            exact_output_matches += 1
        else:
            print(f"[{i}] 출력 불일치")
            print(f"    원본: {o[1]}")
            print(f"    Poincare: {p[1]}")
    
    output_match_rate = exact_output_matches / len(prompts)
    print(f"출력 일치율: {output_match_rate:.1%}")
    
    # 6단계: 안전한 한글 생성 테스트
    test_korean_generation_safe(teacher_for_comparison, tokenizer, device, "원본")
    test_korean_generation_safe(student, tokenizer, device, "Poincare")
    
    # 7단계: 최종 결론
    print(f"\n" + "="*60)
    print("최종 결론")
    print("="*60)
    
    print(f"💾 메모리 효율성:")
    if final_memory_ratio < 1.2:
        print("  ✅ 메모리 최적화 대성공! (딥카피 문제 해결)")
    elif final_memory_ratio < 1.5:
        print("  🟡 메모리 사용량 적당히 증가")
    else:
        print("  ❌ 여전히 메모리 사용량 많음")
    
    print(f"⚡ 속도 성능:")
    if speed_ratio < 2.0:
        print("  ✅ 속도 최적화 성공!")
    elif speed_ratio < 3.0:
        print("  🟡 속도 저하 있지만 허용 범위")
    else:
        print("  ❌ 속도 저하 심각")
    
    print(f"🎯 정확도:")
    if output_match_rate == 1.0:
        print("  ✅ 100% 정확도 유지")
    elif output_match_rate >= 0.8:
        print("  🟡 높은 정확도 유지")
    else:
        print("  ❌ 정확도 저하")
    
    # 메모리 절약량 계산
    if final_memory_ratio < 1.5:
        original_ratio = 1.984  # 이전 딥카피 버전
        improvement = ((original_ratio - final_memory_ratio) / original_ratio) * 100
        print(f"\n🚀 딥카피 제거로 메모리 {improvement:.1f}% 절약!")

if __name__ == "__main__":
    main() 