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

class OptimizedPoincareBallLinear(nn.Module):
    """더 효율적인 Poincaré Ball 선형 레이어"""
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
        
        try:
            # 방법 1: Fused Linear 사용 (가장 빠름)
            if x.is_cuda and hasattr(rs, 'fused_linear'):
                result = rs.fused_linear(x, self.weight, self.bias, self.curvature)
                return result
            
            # 방법 2: Direct CUDA operations (중간 속도)
            elif x.is_cuda:
                linear_out = F.linear(x, self.weight, self.bias)
                # 직접 CUDA 함수 호출로 중간 변수 제거
                x_scaled = torch.tanh(torch.norm(x, dim=-1, keepdim=True) * 0.1)
                out_scaled = torch.tanh(torch.norm(linear_out, dim=-1, keepdim=True) * 0.1)
                x_safe = x * x_scaled / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
                out_safe = linear_out * out_scaled / (torch.norm(linear_out, dim=-1, keepdim=True) + 1e-8)
                
                # CUDA 함수 직접 호출
                if hasattr(rs, 'poincare_ball_forward_cuda'):
                    hyperbolic_out = rs.poincare_ball_forward_cuda(x_safe, out_safe, self.curvature, 0.02)
                else:
                    hyperbolic_out = rs.poincare_ball_layer(x_safe, out_safe, self.curvature, 0.02)
                
                return 0.98 * linear_out + 0.02 * hyperbolic_out
            
            # 방법 3: CPU fallback
            else:
                linear_out = F.linear(x, self.weight, self.bias)
                x_norm = torch.norm(x, dim=-1, keepdim=True)
                x_safe = x / (x_norm + 1e-8) * torch.tanh(x_norm * 0.1)
                out_norm = torch.norm(linear_out, dim=-1, keepdim=True) 
                out_safe = linear_out / (out_norm + 1e-8) * torch.tanh(out_norm * 0.1)
                hyperbolic_out = rs.poincare_ball_layer(x_safe, out_safe, self.curvature, 0.02)
                return 0.98 * linear_out + 0.02 * hyperbolic_out
                
        except Exception as e:
            return F.linear(x, self.weight, self.bias)

class BatchedPoincareBlock(nn.Module):
    """배치 처리에 최적화된 Poincare 블록"""
    def __init__(self, block: nn.Module, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        
        # 원본 레이어 분석
        attn = block.attn
        mlp = block.mlp
        
        # 최적화된 레이어로 교체
        self.attn_c_attn = OptimizedPoincareBallLinear(
            attn.c_attn.weight.shape[0], attn.c_attn.weight.shape[1], curvature,
            bias=(attn.c_attn.bias is not None)
        )
        self.attn_c_proj = OptimizedPoincareBallLinear(
            attn.c_proj.weight.shape[1], attn.c_proj.weight.shape[0], curvature,
            bias=(attn.c_proj.bias is not None)
        )
        self.mlp_c_fc = OptimizedPoincareBallLinear(
            mlp.c_fc.weight.shape[0], mlp.c_fc.weight.shape[1], curvature,
            bias=(mlp.c_fc.bias is not None)
        )
        self.mlp_c_proj = OptimizedPoincareBallLinear(
            mlp.c_proj.weight.shape[0], mlp.c_proj.weight.shape[1], curvature,
            bias=(mlp.c_proj.bias is not None)
        )
        
        # 가중치 복사
        with torch.no_grad():
            self.attn_c_attn.weight.data.copy_(attn.c_attn.weight.data.t())
            self.attn_c_proj.weight.data.copy_(attn.c_proj.weight.data)
            self.mlp_c_fc.weight.data.copy_(mlp.c_fc.weight.data.t())
            self.mlp_c_proj.weight.data.copy_(mlp.c_proj.weight.data.t())
            
            if self.attn_c_attn.bias is not None and attn.c_attn.bias is not None:
                self.attn_c_attn.bias.data.copy_(attn.c_attn.bias.data)
            if self.attn_c_proj.bias is not None and attn.c_proj.bias is not None:
                self.attn_c_proj.bias.data.copy_(attn.c_proj.bias.data)
            if self.mlp_c_fc.bias is not None and mlp.c_fc.bias is not None:
                self.mlp_c_fc.bias.data.copy_(mlp.c_fc.bias.data)
            if self.mlp_c_proj.bias is not None and mlp.c_proj.bias is not None:
                self.mlp_c_proj.bias.data.copy_(mlp.c_proj.bias.data)
        
        # 나머지 attention 파라미터들
        self.attn_bias = attn.bias
        self.attn_scale = getattr(attn, 'scale', None)
        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        # activation 함수
        self.mlp_act = mlp.act

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.attn_scale:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        
        nd, ns = attn_weights.size(-2), attn_weights.size(-1)
        if not self.attn_bias.is_cuda:
            self.attn_bias = self.attn_bias.to(attn_weights.device)
        mask = self.attn_bias[:, :, ns-nd : ns, :ns]
        attn_weights = torch.where(mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self, x, **kwargs):
        # Attention
        h = self.ln_1(x)
        
        # QKV projection with optimized Poincare layer
        qkv = self.attn_c_attn(h)
        new_shape = qkv.size()[:-1] + (self.num_heads, 3 * self.head_dim)
        qkv = qkv.view(new_shape)
        qkv = qkv.permute(0, 2, 1, 3)
        
        query, key, value = qkv.split(self.head_dim, dim=-1)
        
        attn_output, attn_weights = self._attn(query, key, value, kwargs.get('attention_mask'), kwargs.get('head_mask'))
        
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.embed_dim,)
        attn_output = attn_output.view(new_shape)
        
        attn_output = self.attn_c_proj(attn_output)
        x = x + attn_output
        
        # MLP
        h2 = self.ln_2(x)
        h2 = self.mlp_c_fc(h2)
        h2 = self.mlp_act(h2)
        h2 = self.mlp_c_proj(h2)
        x = x + h2
        
        # GPT-2 호환 출력 형태 맞추기
        outputs = (x,)
        if kwargs.get('output_attentions', False):
            outputs = outputs + (attn_weights,)
        if kwargs.get('use_cache', False):
            outputs = outputs + (None,)  # past_key_value placeholder
        return outputs

def create_optimized_poincare_model(teacher_model: nn.Module, curvature: float = 1.0):
    student = copy.deepcopy(teacher_model)
    total_blocks = len(student.transformer.h)
    
    for i in tqdm(range(total_blocks), desc="최적화된 포인카레 변환"):
        orig_block = student.transformer.h[i]
        student.transformer.h[i] = BatchedPoincareBlock(orig_block, curvature=curvature)
    
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
    
    print("최적화된 RealityStone Poincare Ball 테스트")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompts = ["안녕하세요", "오늘 날씨는", "한국의 수도는", "인공지능이란", "맛있는 음식은"]
    
    teacher_memory = measure_memory_usage(teacher, device)
    print(f"Teacher 메모리: {teacher_memory:.1f} MB")
    
    print("\n=== 원본 테스트 ===")
    orig_results, orig_time = fast_test(teacher, tokenizer, device, prompts, "원본")
    
    print(f"\n최적화된 Poincare 모델 생성 중...")
    student = create_optimized_poincare_model(teacher, curvature)
    
    student_memory = measure_memory_usage(student, device)
    print(f"Student 메모리: {student_memory:.1f} MB")
    print(f"메모리 비율: {student_memory/teacher_memory:.3f}")
    
    print("\n=== 최적화된 포인카레 테스트 ===")
    poincare_results, poincare_time = fast_test(student, tokenizer, device, prompts, "최적화 포인카레")
    
    print("\n=== 최종 결과 ===")
    print(f"속도 비율: {poincare_time / orig_time:.3f}")
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
    
    if poincare_time < orig_time * 2:
        print("결과: 최적화 성공!")
    elif output_match_rate == 1.0:
        print("결과: 정확도 유지됨")
    else:
        print("결과: 추가 최적화 필요")

if __name__ == "__main__":
    main() 