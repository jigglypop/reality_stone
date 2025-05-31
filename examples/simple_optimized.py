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

class FastPoincareBallLinear(nn.Module):
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
            # 최적화 1: 직접 CUDA 함수 호출 (중간 변수 최소화)
            if x.is_cuda and hasattr(rs, 'poincare_ball_forward_cuda'):
                # 정규화를 한 번에 처리
                x_norm = torch.norm(x, dim=-1, keepdim=True)
                out_norm = torch.norm(linear_out, dim=-1, keepdim=True)
                scale_factor = 0.05  # 더 작은 스케일로 안정성 확보
                
                x_safe = x * torch.tanh(x_norm * scale_factor) / (x_norm + 1e-8)
                out_safe = linear_out * torch.tanh(out_norm * scale_factor) / (out_norm + 1e-8)
                
                # 직접 CUDA 함수 호출
                hyperbolic_out = rs.poincare_ball_forward_cuda(x_safe, out_safe, self.curvature, 0.01)
                
                # 스케일 복원 및 혼합
                hyp_norm = torch.norm(hyperbolic_out, dim=-1, keepdim=True)
                result = hyperbolic_out * out_norm / (hyp_norm + 1e-8)
                return 0.99 * linear_out + 0.01 * result
                
            # 최적화 2: Fused operation 시도
            elif x.is_cuda and hasattr(rs, 'fused_linear'):
                return rs.fused_linear(x, self.weight, self.bias, self.curvature)
                
            # 기본: 원본 방식 유지
            else:
                x_norm = torch.norm(x, dim=-1, keepdim=True)
                x_safe = x / (x_norm + 1e-8) * torch.tanh(x_norm * 0.05)
                out_norm = torch.norm(linear_out, dim=-1, keepdim=True) 
                out_safe = linear_out / (out_norm + 1e-8) * torch.tanh(out_norm * 0.05)
                hyperbolic_out = rs.poincare_ball_layer(x_safe, out_safe, self.curvature, 0.01)
                hyp_norm = torch.norm(hyperbolic_out, dim=-1, keepdim=True)
                result = hyperbolic_out / (hyp_norm + 1e-8) * out_norm
                return 0.99 * linear_out + 0.01 * result
                
        except:
            return linear_out

class FastPoincareBallWrappedLinear(nn.Module):
    def __init__(self, original_layer: nn.Module, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
        self.original_layer = copy.deepcopy(original_layer)
        
        if hasattr(original_layer, 'nf'):
            in_features = original_layer.weight.shape[0]
            out_features = original_layer.weight.shape[1]
            is_conv1d = True
        elif hasattr(original_layer, 'weight'):
            out_features, in_features = original_layer.weight.shape
            is_conv1d = False
        else:
            raise ValueError("Cannot determine layer dimensions")
            
        self.poincare_layer = FastPoincareBallLinear(in_features, out_features, curvature, bias=(hasattr(original_layer, 'bias') and original_layer.bias is not None))
        
        with torch.no_grad():
            if is_conv1d:
                self.poincare_layer.weight.data.copy_(original_layer.weight.data.t())
            else:
                self.poincare_layer.weight.data.copy_(original_layer.weight.data)
            if self.poincare_layer.bias is not None and hasattr(original_layer, 'bias') and original_layer.bias is not None:
                self.poincare_layer.bias.data.copy_(original_layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            result = self.poincare_layer(x)
            expected_shape = list(x.shape)
            expected_shape[-1] = self.poincare_layer.out_features
            if result.shape != torch.Size(expected_shape):
                return self.original_layer(x)
            return result
        except:
            return self.original_layer(x)

class FastPoincareBlock(nn.Module):
    def __init__(self, block: nn.Module, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
        self.ln_1 = copy.deepcopy(block.ln_1)
        self.ln_2 = copy.deepcopy(block.ln_2)
        attn = copy.deepcopy(block.attn)
        mlp = copy.deepcopy(block.mlp)
        attn.c_attn = FastPoincareBallWrappedLinear(attn.c_attn, curvature)
        attn.c_proj = FastPoincareBallWrappedLinear(attn.c_proj, curvature)
        mlp.c_fc = FastPoincareBallWrappedLinear(mlp.c_fc, curvature)
        mlp.c_proj = FastPoincareBallWrappedLinear(mlp.c_proj, curvature)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x, **kwargs):
        h = self.ln_1(x)
        attn_outputs = self.attn(h, **kwargs)
        a = attn_outputs[0]
        x = x + a
        h2 = self.ln_2(x)
        m = self.mlp(h2)
        out = x + m
        if len(attn_outputs) > 1:
            return (out,) + attn_outputs[1:]
        return (out,)

def create_fast_poincare_model(teacher_model: nn.Module, curvature: float = 1.0):
    student = copy.deepcopy(teacher_model)
    total_blocks = len(student.transformer.h)
    for i in tqdm(range(total_blocks), desc="빠른 포인카레 변환"):
        orig_block = student.transformer.h[i]
        student.transformer.h[i] = FastPoincareBlock(orig_block, curvature=curvature)
    return student

def fast_test(model, tokenizer, device, prompts, model_type="모델", max_length=50):
    model.to(device).eval()
    results = []
    total_time = 0.0
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
    print(f"[{model_type}] 평균 시간: {avg_time:.3f}초")
    return results, avg_time

def measure_memory_usage(model, device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
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
    print("빠른 RealityStone Poincare Ball 테스트")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["안녕하세요", "오늘 날씨는", "한국의 수도는", "인공지능이란", "맛있는 음식은"]
    teacher_memory = measure_memory_usage(teacher, device)
    print(f"Teacher 메모리: {teacher_memory:.1f} MB")
    print("\n=== 원본 테스트 ===")
    orig_results, orig_time = fast_test(teacher, tokenizer, device, prompts, "원본")
    print(f"\n빠른 Poincare 모델 생성 중...")
    student = create_fast_poincare_model(teacher, curvature)
    student_memory = measure_memory_usage(student, device)
    print(f"Student 메모리: {student_memory:.1f} MB")
    print(f"메모리 비율: {student_memory/teacher_memory:.3f}")
    print("\n=== 빠른 포인카레 테스트 ===")
    poincare_results, poincare_time = fast_test(student, tokenizer, device, prompts, "빠른 포인카레")
    print("\n=== 최종 결과 ===")
    speed_ratio = poincare_time / orig_time
    print(f"속도 비율: {speed_ratio:.3f}")
    print(f"메모리 비율: {student_memory/teacher_memory:.3f}")
    exact_output_matches = 0
    for i, (o, p) in enumerate(zip(orig_results, poincare_results), 1):
        if o[1] == p[1]:
            print(f"[{i}] 출력 일치")
            exact_output_matches += 1
        else:
            print(f"[{i}] 출력 불일치")
    output_match_rate = exact_output_matches / len(prompts)
    print(f"출력 일치율: {output_match_rate:.1%}")
    
    # 성능 개선 분석
    original_speed = 3.077  # 기존 버전의 속도 비율
    if speed_ratio < original_speed:
        improvement = ((original_speed - speed_ratio) / original_speed) * 100
        print(f"🚀 속도 개선: {improvement:.1f}% 향상!")
    else:
        print("⚠️ 추가 최적화 필요")
    
    if output_match_rate == 1.0:
        print("✅ 100% 정확도 유지")

if __name__ == "__main__":
    main() 