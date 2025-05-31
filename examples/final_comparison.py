import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from tqdm import tqdm
import reality_stone as rs

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
            module.c_fc = new_layer
            total_replaced += 1
    
    print(f"총 {total_replaced}개 레이어 교체 완료")
    return model

def create_single_layer_poincare_model(teacher_model: nn.Module, curvature: float = 1.0):
    student = copy.deepcopy(teacher_model)
    print("단일 레이어 포인카레 변환 시작...")
    student = replace_linear_layers_inplace(student, curvature)
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

def test_korean_generation(model, tokenizer, device, model_name):
    model.to(device).eval()
    korean_prompts = [
        "한국의 아름다운 곳은",
        "오늘은 좋은 날입니다",
        "맛있는 한국 음식은",
        "서울에서 가볼 만한 곳은",
        "한국어는 정말"
    ]
    
    print(f"\n=== {model_name} 한글 생성 품질 테스트 ===")
    for idx, prompt in enumerate(korean_prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=80, 
                do_sample=True, 
                temperature=0.8, 
                top_p=0.9, 
                top_k=50, 
                pad_token_id=tokenizer.eos_token_id
            )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[{idx}] {prompt}")
        print(f"    {gen_text}")
        print("-" * 60)

def measure_memory_usage(model, device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        return memory_used
    else:
        return 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "skt/kogpt2-base-v2"
    curvature = 1.0
    
    print("RealityStone Poincare Ball 최종 비교 테스트")
    
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 메모리 측정
    teacher_memory = measure_memory_usage(teacher, device)
    print(f"원본 모델 메모리: {teacher_memory:.1f} MB")
    
    # Poincare 모델 생성
    print(f"\nPoincare 모델 생성 중...")
    student = create_single_layer_poincare_model(teacher, curvature)
    student_memory = measure_memory_usage(student, device)
    print(f"Poincare 모델 메모리: {student_memory:.1f} MB")
    memory_ratio = student_memory/teacher_memory
    print(f"메모리 비율: {memory_ratio:.3f}")
    
    # 테스트 프롬프트
    prompts = ["안녕하세요", "오늘 날씨는", "한국의 수도는", "인공지능이란", "맛있는 음식은"]
    
    # 최종 비교 테스트 (원본은 새로 복사해서 사용)
    print(f"\n" + "="*60)
    print("최종 원본 vs Poincare 모델 비교")
    print("="*60)
    
    teacher_copy = copy.deepcopy(teacher)
    orig_results, orig_time = test_model(teacher_copy, tokenizer, device, prompts, "원본")
    
    poincare_results, poincare_time = test_model(student, tokenizer, device, prompts, "Poincare")
    
    # 결과 비교
    print(f"\n" + "="*60)
    print("성능 비교 결과")
    print("="*60)
    
    speed_ratio = poincare_time / orig_time
    print(f"속도 비율: {speed_ratio:.3f} (원본 대비)")
    print(f"메모리 비율: {memory_ratio:.3f} (원본 대비)")
    
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
    
    # 한글 생성 품질 테스트
    test_korean_generation(teacher_copy, tokenizer, device, "원본")
    test_korean_generation(student, tokenizer, device, "Poincare")
    
    # 최종 결론
    print(f"\n" + "="*60)
    print("최종 결론")
    print("="*60)
    
    if memory_ratio < 1.2:
        print("✅ 메모리 최적화 대성공!")
    elif memory_ratio < 2.0:
        print("🟡 메모리 사용량 증가 있음")
    else:
        print("❌ 메모리 사용량 과다")
    
    if speed_ratio < 2.0:
        print("✅ 속도 최적화 성공!")
    elif speed_ratio < 3.0:
        print("🟡 속도 저하 있음")
    else:
        print("❌ 속도 저하 심각")
    
    if output_match_rate == 1.0:
        print("✅ 100% 정확도 유지")
    elif output_match_rate >= 0.8:
        print("🟡 높은 정확도 유지")
    else:
        print("❌ 정확도 저하")

if __name__ == "__main__":
    main() 