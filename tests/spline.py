import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import Conv1D
import copy

# --- 스플라인 압축 유틸리티 함수 (spline_compression.py에서 가져옴) ---

def catmull_rom_interpolation(control_points, t_values):
    """벡터화된 Catmull-Rom 스플라인 보간"""
    k = control_points.shape[0] - 1
    m = t_values.shape[0]
    
    t_scaled = t_values * k
    j = torch.floor(t_scaled).long()
    t_local = t_scaled - j.float()
    
    j = torch.clamp(j, 1, k - 2)
    
    p0 = control_points[j - 1]
    p1 = control_points[j]
    p2 = control_points[j + 1]
    p3 = control_points[j + 2]
    
    t_local = t_local.unsqueeze(1)
    
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t_local +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t_local.pow(2) +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t_local.pow(3)
    )

# --- 스플라인 압축 레이어 ---

class SplineCompressedLinear(nn.Module):
    """스플라인으로 압축된 선형 변환 레이어"""
    def __init__(self, in_features: int, out_features: int, 
                 k: int = 15,  # 제어점 개수 - 1
                 bias: bool = True,
                 initial_weight: Optional[torch.Tensor] = None,
                 initial_bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        if initial_weight is not None:
            self.control_points = nn.Parameter(self._fit_control_points(initial_weight))
        else:
            self.control_points = nn.Parameter(torch.randn(k + 1, in_features) * 0.02)
        
        if bias:
            self.bias = nn.Parameter(initial_bias if initial_bias is not None else torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('cached_weight', torch.empty(0), persistent=False)

    def _fit_control_points(self, target_weight: torch.Tensor) -> torch.Tensor:
        """가중치에 최적화된 제어점을 찾습니다."""
        temp_control_points = nn.Parameter(
            torch.randn(self.k + 1, self.in_features, device=target_weight.device) * 0.02
        )
        optimizer = torch.optim.AdamW([temp_control_points], lr=1e-2, weight_decay=1e-4)
        
        # 간단한 피팅 루프 (실제로는 더 많은 반복이 필요할 수 있음)
        for _ in range(100): 
            optimizer.zero_grad()
            reconstructed = self._interpolate(temp_control_points)
            loss = F.mse_loss(reconstructed, target_weight)
            loss.backward()
            optimizer.step()
        return temp_control_points.detach()

    def _interpolate(self, control_points: torch.Tensor) -> torch.Tensor:
        t_values = torch.linspace(0, 1, self.out_features, device=control_points.device)
        return catmull_rom_interpolation(control_points, t_values)

    def get_weight(self) -> torch.Tensor:
        """현재 제어점에서 가중치를 복원합니다."""
        if not self.training and self.cached_weight.numel() > 0:
            return self.cached_weight
        weight = self._interpolate(self.control_points)
        if not self.training:
            self.cached_weight = weight.detach()
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)

    def train(self, mode: bool = True):
        self.cached_weight = torch.empty(0, device=self.control_points.device)
        return super().train(mode)

def apply_spline_compression_to_model(model: nn.Module, k: int) -> nn.Module:
    """모델의 모든 선형 및 Conv1D 레이어를 스플라인 압축 레이어로 교체합니다."""
    for name, module in tqdm(list(model.named_modules()), desc="Spline 압축 진행"):
        if isinstance(module, (nn.Linear, Conv1D)):
            if '.' not in name: continue
            
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            
            if isinstance(module, Conv1D):
                in_features, out_features = module.weight.shape
                weight = module.weight.data.t() # [out, in]
            else:
                out_features, in_features = module.weight.shape
                weight = module.weight.data

            new_layer = SplineCompressedLinear(
                in_features, out_features, k=k, bias=(module.bias is not None),
                initial_weight=weight,
                initial_bias=module.bias.data if module.bias is not None else None
            )
            setattr(parent_module, child_name, new_layer)
    return model

# --- 테스트 실행 코드 ---

def test_spline_compression():
    """GPT-2 모델에 스플라인 압축을 적용하고 성능을 테스트합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "skt/kogpt2-base-v2"
    k = 15 # 제어점 개수 - 1

    print(f"모델 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n--- 원본 모델 테스트 ---")
    prompts = ["안녕하세요", "오늘 날씨는", "한국의 수도는"]
    test_model_generation(original_model, tokenizer, device, "원본", prompts)

    print(f"\n--- k={k}로 스플라인 압축 적용 ---")
    compressed_model = copy.deepcopy(original_model)
    compressed_model = apply_spline_compression_to_model(compressed_model, k=k)
    
    print("\n--- 압축 모델 테스트 ---")
    test_model_generation(compressed_model, tokenizer, device, "압축 후", prompts)

    # 파라미터 수 비교
    original_params = sum(p.numel() for p in original_model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    ratio = compressed_params / original_params
    
    print("\n--- 최종 결과 ---")
    print(f"원본 파라미터: {original_params:,}")
    print(f"압축 파라미터: {compressed_params:,}")
    print(f"압축률: {ratio:.3f} ({1/ratio:.1f}x)")

def test_model_generation(model, tokenizer, device, model_type, prompts):
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=50, do_sample=True, top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[{model_type}] '{prompt}' -> {generated_text}")

if __name__ == "__main__":
    test_spline_compression() 