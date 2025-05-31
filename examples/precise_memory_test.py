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
import gc

def precise_memory_measure(device, label=""):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"{label}: {memory_mb:.1f} MB")
        return memory_mb
    return 0.0

class MinimalPoincareBallLinear(nn.Module):
    """최소한의 메모리 사용 Poincaré Ball 레이어"""
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
        
        # 기본 선형 연산만 (hyperbolic 연산 제거로 메모리 절약)
        return F.linear(x, self.weight, self.bias)

def replace_with_minimal_layers(model: nn.Module, curvature: float = 1.0):
    """메모리 사용 최소화를 위해 기본 레이어로만 교체"""
    total_replaced = 0
    
    # 메모리 사용량 추적
    device = next(model.parameters()).device
    start_memory = precise_memory_measure(device, "교체 시작")
    
    for name, module in model.named_modules():
        if hasattr(module, 'c_attn') and hasattr(module.c_attn, 'weight'):
            old_layer = module.c_attn
            if hasattr(old_layer, 'nf'):  # GPT2Conv1D
                in_features = old_layer.weight.shape[0]
                out_features = old_layer.weight.shape[1]
                new_layer = MinimalPoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data.t())
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            else:  # nn.Linear
                out_features, in_features = old_layer.weight.shape
                new_layer = MinimalPoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
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
                new_layer = MinimalPoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data.t())
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            else:
                out_features, in_features = old_layer.weight.shape
                new_layer = MinimalPoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
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
                new_layer = MinimalPoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data.t())
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            else:
                out_features, in_features = old_layer.weight.shape
                new_layer = MinimalPoincareBallLinear(in_features, out_features, curvature, bias=(old_layer.bias is not None))
                with torch.no_grad():
                    new_layer.weight.data.copy_(old_layer.weight.data)
                    if new_layer.bias is not None and old_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)
            
            del module.c_fc
            torch.cuda.empty_cache()
            module.c_fc = new_layer
            total_replaced += 1
    
    end_memory = precise_memory_measure(device, "교체 완료")
    print(f"총 {total_replaced}개 레이어 교체 (메모리 변화: {end_memory - start_memory:+.1f} MB)")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "skt/kogpt2-base-v2"
    
    print("정밀 메모리 측정 테스트")
    
    # 1단계: 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    initial_memory = precise_memory_measure(device, "초기 상태")
    
    # 2단계: 모델 로드
    print("\n=== 모델 로드 단계별 메모리 ===")
    teacher = AutoModelForCausalLM.from_pretrained(model_name)
    after_load_memory = precise_memory_measure(device, "모델 로드 후 (CPU)")
    
    teacher = teacher.to(device)
    after_gpu_memory = precise_memory_measure(device, "GPU 이동 후")
    
    # 3단계: deepcopy 테스트
    print("\n=== deepcopy 메모리 영향 ===")
    student = copy.deepcopy(teacher)
    after_copy_memory = precise_memory_measure(device, "deepcopy 후")
    copy_overhead = after_copy_memory - after_gpu_memory
    print(f"deepcopy 오버헤드: {copy_overhead:.1f} MB ({copy_overhead/after_gpu_memory*100:.1f}%)")
    
    # 4단계: 레이어 교체
    print("\n=== 레이어 교체 메모리 영향 ===")
    student = replace_with_minimal_layers(student, 1.0)
    after_replace_memory = precise_memory_measure(device, "레이어 교체 후")
    
    # 5단계: 원본 모델 삭제 테스트
    print("\n=== 모델 삭제 메모리 영향 ===")
    del teacher
    torch.cuda.empty_cache()
    gc.collect()
    after_teacher_del_memory = precise_memory_measure(device, "teacher 삭제 후")
    
    # 6단계: 최종 메모리 비율
    print("\n=== 최종 메모리 분석 ===")
    final_ratio = after_teacher_del_memory / after_gpu_memory
    print(f"원본 모델 메모리: {after_gpu_memory:.1f} MB")
    print(f"최종 student 메모리: {after_teacher_del_memory:.1f} MB")
    print(f"실제 메모리 비율: {final_ratio:.3f}")
    
    if final_ratio < 1.2:
        print("✅ 메모리 최적화 성공!")
    elif final_ratio < 1.5:
        print("🟡 부분적 메모리 최적화")
    else:
        print("❌ 메모리 최적화 실패")
    
    # 7단계: 간단한 동작 테스트
    print("\n=== 동작 테스트 ===")
    test_input = tokenizer("안녕하세요", return_tensors="pt").to(device)
    with torch.no_grad():
        output = student.generate(**test_input, max_length=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"테스트 출력: {result}")
    
    final_memory = precise_memory_measure(device, "최종 상태")
    print(f"\n=== 메모리 변화 요약 ===")
    print(f"초기: {initial_memory:.1f} MB")
    print(f"원본 로드: {after_gpu_memory:.1f} MB (+{after_gpu_memory-initial_memory:.1f})")
    print(f"복사 후: {after_copy_memory:.1f} MB (+{copy_overhead:.1f})")
    print(f"교체 후: {after_replace_memory:.1f} MB (+{after_replace_memory-after_copy_memory:.1f})")
    print(f"삭제 후: {after_teacher_del_memory:.1f} MB ({final_ratio:.3f}배)")

if __name__ == "__main__":
    main() 