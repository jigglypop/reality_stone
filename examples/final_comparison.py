import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from tqdm import tqdm
try:
    import reality_stone as rs
    from reality_stone.layers import RBELinear
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    # Define a placeholder class if reality_stone is not available
    class RBELinear(nn.Module):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
        def forward(self, x):
            return self.linear(x)
        @classmethod
        def from_linear(cls, linear, **kwargs):
            return cls(linear.in_features, linear.out_features)

# --- 헬가손-RBE 하이브리드 압축기 ---
class HelgasonRBECompressor:
    """
    헬가손-푸리에 변환으로 주요 특징을 잡고,
    잔차를 RBE로 정밀하게 압축하는 극한 압축기
    """
    def __init__(self, W: torch.Tensor, compression_ratio=0.1):
        self.shape = W.shape
        self.dtype = W.dtype
        self.device = W.device

        print(f"    🌀 헬가손-비트필드 압축 시작: {self.shape}, 목표압축률={compression_ratio:.1%}")

        # 1. 헬가손-푸리에 변환으로 매크로 구조 압축
        W_fft = torch.fft.fft2(W.float())
        energy = torch.abs(W_fft)**2
        sorted_indices = torch.argsort(energy.flatten(), descending=True)
        
        # 에너지 기반으로 주요 주파수 선택 (매크로 정보)
        macro_budget = int(W.numel() * compression_ratio * 0.5) # 예산의 50%
        important_indices = sorted_indices[:macro_budget]
        
        freq_mask = torch.zeros_like(energy.flatten(), dtype=torch.bool)
        freq_mask[important_indices] = True
        self.freq_mask = freq_mask.reshape(W_fft.shape)
        
        self.important_freqs = torch.where(self.freq_mask, W_fft, torch.zeros_like(W_fft))

        # 2. 잔차 계산 (마이크로 정보)
        macro_reconstructed = torch.fft.ifft2(self.important_freqs).real.to(self.dtype)
        residual = W - macro_reconstructed
        print(f"       - 1단계(헬가손): 잔차에너지 = {torch.norm(residual) / torch.norm(W):.2%}")

        # 3. 잔차를 Bitfield로 2차 압축
        if RS_AVAILABLE:
            # BitfieldLinear.from_linear는 nn.Linear 객체를 인자로 받음
            residual_linear_layer = nn.Linear(self.shape[1], self.shape[0], bias=False)
            residual_linear_layer.weight.data = residual
            residual_linear_layer.to(self.device)
            self.residual_bitfield = BitfieldLinear.from_linear(residual_linear_layer, r_max=0.5)
        else: # Fallback
            self.residual_bitfield = residual

    def reconstruct(self) -> torch.Tensor:
        """압축된 가중치 복원 (디버깅용, 현재 decompress 미지원)"""
        macro_reconstructed = torch.fft.ifft2(self.important_freqs).real.to(self.dtype)
        
        if RS_AVAILABLE:
            # Bitfield 복원은 현재 불가. 
            # decompress() 메소드를 BitfieldLinear에 추가해야 완전한 복원이 가능합니다.
            print("⚠️ Bitfield decompress는 미구현 상태입니다. 매크로 부분만 복원합니다.")
            residual_reconstructed = torch.zeros_like(macro_reconstructed)
        else:
            residual_reconstructed = self.residual_bitfield

        return macro_reconstructed + residual_reconstructed

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """압축된 연산 적용"""
        # 1. Macro (FFT) 부분 적용
        macro_reconstructed = torch.fft.ifft2(self.important_freqs).real.to(x.dtype)
        macro_output = F.linear(x, macro_reconstructed)

        # 2. Micro (Bitfield) 부분 적용
        if RS_AVAILABLE:
            residual_output = self.residual_bitfield(x)
        else:
            residual_output = F.linear(x, self.residual_bitfield)

        return macro_output + residual_output

class HybridCompressedLinear(nn.Module):
    """헬가손-비트필드 압축을 적용한 최종 선형 레이어"""
    def __init__(self, linear_layer: nn.Linear, compression_ratio=0.1, is_attn=False):
        super().__init__()
        
        self.is_attn = is_attn
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        if self.is_attn:
            # c_attn 가중치는 [out*3, in] 모양
            weights = linear_layer.weight.data
            w_q, w_k, w_v = torch.chunk(weights, 3, dim=0)
            self.compressor_q = HelgasonBitfieldCompressor(w_q, compression_ratio)
            self.compressor_k = HelgasonBitfieldCompressor(w_k, compression_ratio)
            self.compressor_v = HelgasonBitfieldCompressor(w_v, compression_ratio)
        else:
            self.compressor = HelgasonBitfieldCompressor(linear_layer.weight.data, compression_ratio)
        
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.is_attn:
            # Q, K, V 각각에 대해 압축된 연산 수행 후 결합
            q = self.compressor_q.apply(x)
            k = self.compressor_k.apply(x)
            v = self.compressor_v.apply(x)
            # transformers의 c_attn 출력과 동일한 차원으로 합침
            output = torch.cat([q, k, v], dim=-1)
        else:
            output = self.compressor.apply(x)
            
        if self.bias is not None:
            output += self.bias
        return output

def apply_hybrid_compression_to_model(model, compression_ratio=0.1):
    """모델의 모든 선형 레이어를 하이브리드 압축 레이어로 교체"""
    print(f"\n🌀 모델 전체 하이브리드 압축 시작 (목표: {compression_ratio:.1%})")
    
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or type(module).__name__ == 'Conv1D':
            layers_to_replace.append(name)
            
    for name in tqdm(layers_to_replace, desc="압축 진행"):
        module = model.get_submodule(name)
        if '.' not in name:
            continue

        parent_name, child_name = name.rsplit('.', 1)
        parent_module = model.get_submodule(parent_name)

        is_attn = 'c_attn' in name
        
        # 가중치 모양을 [out_features, in_features]로 통일
        if type(module).__name__ == 'Conv1D':
            # Conv1D의 가중치는 [in, out] -> t() -> [out, in]
            out_features, in_features = module.weight.shape
            linear_equiv = nn.Linear(in_features, out_features, bias=(module.bias is not None))
            linear_equiv.weight.data = module.weight.data.t()
            if module.bias is not None:
                linear_equiv.bias.data = module.bias.data
        else: # nn.Linear
            linear_equiv = module
            
        new_layer = HybridCompressedLinear(linear_equiv, compression_ratio, is_attn=is_attn)
        setattr(parent_module, child_name, new_layer)
        
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "skt/kogpt2-base-v2"
    
    if not RS_AVAILABLE:
        print("="*60)
        print("⚠️ 경고: RealityStone 라이브러리를 찾을 수 없습니다.")
        print("압축 기능 없이 폴백 모드로 실행합니다.")
        print("="*60)

    print("RealityStone 하이브리드 압축 최종 비교 테스트")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # use_safetensors=True를 추가하여 보안 및 로딩 문제 해결
    teacher = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompts = ["안녕하세요", "오늘 날씨는", "한국의 수도는", "인공지능이란", "맛있는 음식은"]
    
    # 원본 모델 테스트
    teacher_copy = copy.deepcopy(teacher)
    orig_results, orig_time = test_model(teacher_copy, tokenizer, device, prompts, "원본")
    del teacher_copy

    # 하이브리드 압축 모델 생성 및 테스트
    student = copy.deepcopy(teacher)
    student = apply_hybrid_compression_to_model(student, compression_ratio=0.1) # 10% 압축률 목표
    
    # TODO: 지식 증류 파인튜닝 추가
    
    comp_results, comp_time = test_model(student, tokenizer, device, prompts, "하이브리드 압축")

    # 결과 비교
    print("\n" + "="*60 + "\n성능 비교 결과\n" + "="*60)
    
    speed_ratio = comp_time / orig_time if orig_time > 0 else 0
    print(f"속도 비율: {speed_ratio:.3f} (원본 대비)")
    
    # 메모리 사용량 측정 (단순 파라미터 수 비교)
    orig_params = sum(p.numel() for p in teacher.parameters())
    comp_params = sum(p.numel() for p in student.parameters())
    mem_ratio = comp_params / orig_params if orig_params > 0 else 0
    print(f"파라미터 비율: {mem_ratio:.3f} ({1/mem_ratio if mem_ratio > 0 else 0:.1f}x 압축)")

    exact_output_matches = 0
    for i, (o, p) in enumerate(zip(orig_results, comp_results), 1):
        if o[1] == p[1]:
            print(f"[{i}] 출력 일치")
            exact_output_matches += 1
        else:
            print(f"[{i}] 출력 불일치\n    원본: {o[1]}\n    압축: {p[1]}")
    
    output_match_rate = exact_output_matches / len(prompts)
    print(f"출력 일치율: {output_match_rate:.1%}")

def test_model(model, tokenizer, device, prompts, model_name, max_length=50):
    model.to(device).eval()
    results = []
    total_time = 0.0
    
    print(f"\n=== {model_name} 테스트 ===")
    for idx, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.time() - start
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_time += elapsed
        print(f"[{idx}] '{prompt}' -> {gen_text} ({elapsed:.3f}s)")
        results.append((prompt, gen_text, elapsed))
    
    avg_time = total_time / len(prompts) if prompts else 0
    print(f"{model_name} 평균 시간: {avg_time:.3f}초")
    return results, avg_time

if __name__ == "__main__":
    main() 