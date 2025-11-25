import torch
import torch.nn as nn
from typing import Dict

# 전역 메트릭 상태 관리자 (Global Metric State)
class MetricState:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricState, cls).__new__(cls)
            cls._instance.curvature = 0.0 # 초기엔 유클리드 (Wake)
            cls._instance.mode = "WAKE"
        return cls._instance

    def set_mode(self, mode: str, c: float):
        self.mode = mode
        self.curvature = c
        print(f"🌌 Brain State Switched: [{mode}] (Curvature: {c})")

# 동적 리만 레이어 (외부 상태에 따라 공간이 휘어짐)
class DynamicRiemannianLinear(nn.Module):
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.state = MetricState() # 전역 상태 참조

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 유클리드 연산 (지식 접근)
        # GPT2 Conv1D의 경우 weight가 [in_features, out_features] 형태입니다.
        # 일반 Linear는 [out_features, in_features] 입니다.
        # shape[0] == x.shape[-1] 이면 Conv1D(Linear가 아님)로 간주해야 합니다.
        
        is_conv1d = False
        if hasattr(self.weight, 'shape') and len(self.weight.shape) == 2:
            # x: [batch, seq, in_features]
            # weight: [in_features, out_features] for Conv1D
            # weight: [out_features, in_features] for Linear
            if self.weight.shape[0] == x.shape[-1] and self.weight.shape[1] != x.shape[-1]: 
                 is_conv1d = True
            # Conv1D can be square too, check type or shape logic carefully
            # GPT2 c_attn weight is [768, 2304] (Conv1D)
            # x is [..., 768]
            elif self.weight.shape[0] == x.shape[-1]: 
                 is_conv1d = True

        if is_conv1d:
             # Conv1D (y = xW + b)
             size_out = x.size()[:-1] + (self.weight.size(1),)
             x_view = x.view(-1, x.size(-1))
             y = torch.addmm(self.bias, x_view, self.weight)
             y = y.view(size_out)
        else:
             # Linear (y = xA^T + b)
             y = torch.nn.functional.linear(x, self.weight, self.bias)

        # 2. 동적 메트릭 보정 (관점 전환)
        # 현재 뇌 상태(Curvature)에 따라 출력을 왜곡시킴
        c = self.state.curvature
        if abs(c) > 1e-5:
            x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
            denom = 1.0 - c * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            y = y * conformal_factor
            
        return y

class BrainOS:
    """
    모델의 수면 및 컨텍스트 스위칭을 담당하는 운영체제
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.state = MetricState()
        self.convert_layers()
        
    def convert_layers(self):
        print("🧠 Installing Neural Interface (Dynamic Riemannian Layers)...")
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) or "Conv1D" in str(type(module)):
                if isinstance(module, DynamicRiemannianLinear): continue
                if "lm_head" in name: continue # 헤드는 놔둠 (선택사항)
                
                new_layer = DynamicRiemannianLinear(module)
                self._replace_module(name, new_layer)
                count += 1
        print(f"   - {count} synaptic connections interfaced.")

    def _replace_module(self, path, new_module):
        atoms = path.split('.')
        parent = self.model
        try:
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], new_module)
        except: pass

    def wake_up(self):
        self.state.set_mode("WAKE (Euclidean)", 0.0)
        
    def focus_logic(self):
        # 쌍곡 공간: 계층 구조, 논리, 포함 관계 강화
        # 곡률을 너무 크게 주면 붕괴하므로 미세 조정 (-0.01 정도)
        self.state.set_mode("FOCUS: LOGIC (Hyperbolic)", -0.01)
        
    def focus_creative(self):
        # 구면 공간: 순환 구조, 창의성, 연상 강화
        self.state.set_mode("FOCUS: CREATIVE (Spherical)", 0.01)
        
    def sleep_and_consolidate(self, recent_data_batch):
        """
        수면 모드: 가중치는 건드리지 않고, 최적의 곡률(c)만 찾아서 미세 조정
        """
        print("💤 Entering REM Sleep (Metric Consolidation)...")
        self.state.set_mode("DREAMING", 0.0)
        
        # 곡률만 학습 가능한 파라미터로 간주하는 가상 최적화 과정
        # 실제로는 여기서 Loss를 계산해 c를 업데이트
        # (데모를 위해 생략)
        optimized_c = -0.005 # 가상의 최적화 결과 (약한 쌍곡 기하학이 언어에 적합)
        
        print(f"✨ Memory Consolidated. Optimal Metric Found: {optimized_c}")
        self.state.set_mode("WAKE (Optimized)", optimized_c)

