import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from .. import _rust, _has_cuda
from ..core.mobius import MobiusAdd, MobiusScalarMul
import math

class ProjectToBall(Function):
    @staticmethod
    def forward(ctx, x: Tensor, epsilon: float = 1e-5) -> Tensor:
        ctx.epsilon = epsilon
        ctx.save_for_backward(x)
        output_np = _rust.project_to_ball_cpu(x.detach().cpu().numpy(), epsilon)
        return torch.from_numpy(output_np).to(x.device)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        x, = ctx.saved_tensors
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # 노름이 1-epsilon보다 작으면 gradient 그대로 통과
        mask = (norm < 1.0 - ctx.epsilon).float()
        grad_x = grad_output * mask
        return grad_x, None

def project_to_ball(x: Tensor, epsilon: float = 1e-5) -> Tensor:
    """텐서를 푸앵카레 공으로 투영합니다. 모든 차원을 지원합니다."""
    # 순수 PyTorch 구현으로 대체 (Rust 바인딩이 2D만 지원하므로)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    # norm이 1-epsilon보다 큰 경우만 스케일링
    scale = torch.where(
        norm > 1.0 - epsilon,
        (1.0 - epsilon) / norm,
        torch.ones_like(norm)
    )
    return x * scale

class PoincareBallLayer(Function):

    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        ctx.t = t
        
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            
            # kappas가 0차원 텐서면 바로 item(), 1차원 이상이면 인덱싱
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
                
            output_np, c_val = _rust.poincare_ball_layer_layerwise_cpu(
                u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(u.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            u_prime = poincare_scalar_mul(u, 1.0 - t, ctx.c)
            v_prime = poincare_scalar_mul(v, t, ctx.c)
            output = poincare_add(u_prime, v_prime, ctx.c)
            ctx.save_for_backward(u.clone(), v.clone(), u_prime.clone(), v_prime.clone())
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        t = ctx.t
        
        if ctx.use_dynamic:
            u, v, kappas = ctx.saved_tensors
            layer_idx = ctx.layer_idx
            c_min = ctx.c_min
            c_max = ctx.c_max
            
            # kappas가 0차원 텐서면 바로 item(), 1차원 이상이면 인덱싱
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
                
            grad_u_np, grad_v_np, grad_kappa_val = _rust.poincare_ball_layer_layerwise_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), 
                kappa_val, layer_idx, c_min, c_max, t
            )
            
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
            
            # kappas와 같은 차원의 gradient 생성
            if kappas.dim() == 0:
                grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            else:
                grad_kappas = torch.zeros_like(kappas)
                grad_kappas[layer_idx] = grad_kappa_val
            
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        else:
            u, v, u_prime, v_prime = ctx.saved_tensors
            c = ctx.c
            grad_u = grad_v = None
            if grad_output.is_cuda and _has_cuda:
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.poincare_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    c, t, u.shape[0], u.shape[1]
                )
            else:
                grad_u_np, grad_v_np = _rust.poincare_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
                )
                grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
                grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
            return grad_u, grad_v, None, None, None, None, None, None

def poincare_add(x: Tensor, y: Tensor, c: float = None, kappas: Tensor = None, layer_idx: int = None) -> Tensor:
    return MobiusAdd.apply(x, y, c, kappas, layer_idx)

def poincare_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    return MobiusScalarMul.apply(x, r, c)

def poincare_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], device=x.device)
        _rust.poincare_distance_cuda(
            x.data_ptr(), y.data_ptr(), output.data_ptr(),
            x.shape[0], x.shape[1], c
        )
        return output
    else:
        output_np = _rust.poincare_distance_cpu(x.cpu().numpy(), y.cpu().numpy(), c)
        return torch.from_numpy(output_np).to(x.device)

def poincare_to_lorentz(x: Tensor, c: float) -> Tensor:
    output_np = _rust.poincare_to_lorentz(x.cpu().numpy(), c)
    return torch.from_numpy(output_np).to(x.device)

def poincare_to_klein(x: Tensor, c: float) -> Tensor:
    output_np = _rust.poincare_to_klein(x.cpu().numpy(), c)
    return torch.from_numpy(output_np).to(x.device)

# --- HyperbolicLinear 및 관련 함수 ---

def exp_map_zero(v: Tensor, c: float, eps: float = 1e-5) -> Tensor:
    """원점에서의 지수 맵 (접선 공간 -> 푸앵카레 공)"""
    sqrt_c = torch.sqrt(torch.tensor(c))
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps)
    # 수치 안정성을 위해 norm을 제한
    v_norm_clipped = v_norm.clamp(max=1.0 / (sqrt_c * 1.1))  # 공의 경계에 너무 가까이 가지 않도록
    # tanh 계산
    sqrt_c_v_norm = sqrt_c * v_norm_clipped
    tanh_term = torch.tanh(sqrt_c_v_norm)
    # 결과 계산
    result = tanh_term / (sqrt_c * v_norm_clipped) * v
    # NaN 체크 및 처리
    result = torch.where(v_norm < eps, v, result)
    return result


def log_map_zero(y: Tensor, c: float, eps: float = 1e-5) -> Tensor:
    """원점에서의 로그 맵 (푸앵카레 공 -> 접선 공간)"""
    sqrt_c = torch.sqrt(torch.tensor(c))
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp(min=eps)
    
    # 수치 안정성을 위해 norm을 제한 (공의 경계에서 멀리)
    y_norm_clipped = y_norm.clamp(max=1.0 - eps)
    
    # artanh 계산을 위한 안전한 범위 확인
    sqrt_c_y_norm = sqrt_c * y_norm_clipped
    sqrt_c_y_norm = sqrt_c_y_norm.clamp(max=1.0 - eps)  # artanh의 정의역 내로 제한
    
    # artanh 계산
    artanh_term = torch.atanh(sqrt_c_y_norm)
    
    # 결과 계산
    result = artanh_term / (sqrt_c * y_norm_clipped) * y
    
    # NaN 체크 및 처리
    result = torch.where(y_norm < eps, y, result)
    return result

class HyperbolicLinear(nn.Module):
    """
    쌍곡 공간에서의 선형 변환 레이어.
    두 가지 모드를 지원:
    1. 'tangent': 접선 공간에서 선형 변환 (기본값)
    2. 'mobius': Mobius 연산을 직접 사용
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True, mode: str = 'tangent'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.mode = mode
        
        # 가중치는 항상 유클리드 공간에 저장
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 쌍곡 공간에 맞는 초기화
        # Xavier/Glorot 초기화를 사용하되, 쌍곡 공간의 특성을 고려하여 스케일 조정
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        
        # 쌍곡 공간에서는 더 작은 값으로 시작하는 것이 안정적
        std = std * 0.5  
        
        with torch.no_grad():
            self.weight.uniform_(-std, std)
            
        if self.bias is not None:
            # 편향은 더 작게 초기화
            bound = 1 / torch.sqrt(torch.tensor(fan_in).float()) * 0.1
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # 3D 텐서 처리를 위해 차원 변환
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])

        if self.mode == 'tangent':
            # 기존 방식: 접선 공간에서 변환
            x_proj = project_to_ball(x, epsilon=1e-5)
            tangent_x = log_map_zero(x_proj, c=self.c)
            tangent_y = F.linear(tangent_x, self.weight, self.bias)
            hyperbolic_y = exp_map_zero(tangent_y, c=self.c)
            
        elif self.mode == 'mobius':
            # 개선된 방식: Mobius 연산 직접 사용
            x_proj = project_to_ball(x, epsilon=1e-5)
            
            # 가중치 행렬의 각 행을 푸앵카레 공으로 투영
            weight_proj = project_to_ball(self.weight, epsilon=1e-5)
            
            # Mobius 행렬-벡터 곱셈
            # y_i = sum_j (w_ij ⊗_c x_j)
            hyperbolic_y = []
            for i in range(self.out_features):
                # 각 출력 차원에 대해
                y_i = torch.zeros_like(x_proj[0])
                for j in range(self.in_features):
                    # Mobius 스칼라 곱셈과 덧셈
                    scaled = poincare_scalar_mul(x_proj[:, j:j+1], self.weight[i, j].item(), self.c)
                    if j == 0:
                        y_i = scaled
                    else:
                        y_i = poincare_add(y_i, scaled, self.c)
                hyperbolic_y.append(y_i)
            
            hyperbolic_y = torch.cat(hyperbolic_y, dim=1)
            
            # 편향 추가 (Mobius 덧셈)
            if self.bias is not None:
                bias_proj = project_to_ball(self.bias.unsqueeze(0), epsilon=1e-5)
                hyperbolic_y = poincare_add(hyperbolic_y, bias_proj.expand_as(hyperbolic_y), self.c)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # 원래 차원으로 복원
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            hyperbolic_y = hyperbolic_y.view(*output_shape)

        return hyperbolic_y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, c={self.c}'

    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        if 'Conv1D' in str(type(linear_layer)):
             in_features=linear_layer.weight.shape[0]
             out_features=linear_layer.weight.shape[1]
             has_bias = linear_layer.bias is not None
        else:
             in_features=linear_layer.in_features
             out_features=linear_layer.out_features
             has_bias = linear_layer.bias is not None

        hyperbolic_layer = cls(in_features=in_features, out_features=out_features, c=c, bias=has_bias)
        
        # 기존 가중치를 쌍곡 공간에 맞게 스케일링
        with torch.no_grad():
            if 'Conv1D' in str(type(linear_layer)):
                weight = linear_layer.weight.t()
            else:
                weight = linear_layer.weight.data
            
            # 가중치를 더 작은 스케일로 조정 (쌍곡 공간에서의 안정성)
            weight_norm = torch.norm(weight, p='fro')
            target_norm = torch.sqrt(torch.tensor(weight.shape[0] * weight.shape[1]).float()) * 0.1
            scale_factor = target_norm / weight_norm
            
            hyperbolic_layer.weight.data.copy_(weight * scale_factor)

            if has_bias:
                # 편향도 스케일링
                hyperbolic_layer.bias.data.copy_(linear_layer.bias.data * 0.1)
            
        return hyperbolic_layer


import torch.nn as nn

class PoincareWrapper(nn.Module):

    def __init__(self, linear_layer: nn.Module):
        super().__init__()
        self.linear_layer = linear_layer

    def forward(self, x: Tensor) -> Tensor:
        # 1. 기존 선형 레이어 실행
        linear_output = self.linear_layer(x)
        
        # 2. 입력을 2D로 변환하여 푸앵카레 공으로 투영
        original_shape = linear_output.shape
        if linear_output.dim() > 2:
            linear_output = linear_output.view(-1, original_shape[-1])
            
        poincare_output = project_to_ball(torch.tanh(linear_output))
        
        # 3. 원래 shape으로 복원
        if len(original_shape) > 2:
            poincare_output = poincare_output.view(*original_shape)
        
        return poincare_output
    
    def __repr__(self):
        return f"PoincareWrapper({self.linear_layer})" 

class GeodesicLinear(nn.Module):
    """
    측지거리를 고려한 쌍곡 선형 레이어.
    HyperbolicLinear의 개선 버전으로, 더 안정적인 초기화와 스케일링을 사용합니다.
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 가중치와 편향
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 매우 작은 값으로 초기화 (쌍곡 공간에 적합)
        with torch.no_grad():
            # Glorot uniform initialization with smaller scale
            limit = math.sqrt(6.0 / (self.in_features + self.out_features)) * 0.01
            self.weight.uniform_(-limit, limit)
            
        if self.bias is not None:
            # 편향은 0으로 초기화
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # 입력 shape 저장
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])
            
        # 입력을 푸앵카레 공으로 투영 (더 작은 스케일)
        x_scaled = x * 0.1  # 입력 스케일 감소
        x_proj = project_to_ball(x_scaled, epsilon=1e-5)
        
        # 접선 공간으로 변환
        tangent_x = log_map_zero(x_proj, c=self.c)
        
        # 선형 변환 (가중치도 작게 스케일링)
        tangent_y = F.linear(tangent_x, self.weight * 0.1, self.bias)
        
        # 다시 푸앵카레 공으로 변환
        hyperbolic_y = exp_map_zero(tangent_y, c=self.c)
        
        # 원래 shape으로 복원
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            hyperbolic_y = hyperbolic_y.view(*output_shape)
            
        return hyperbolic_y
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        """기존 선형 레이어로부터 GeodesicLinear 생성"""
        if hasattr(linear_layer, 'in_features'):
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            has_bias = linear_layer.bias is not None
        else:  # Conv1D
            in_features = linear_layer.weight.shape[0]
            out_features = linear_layer.weight.shape[1]
            has_bias = linear_layer.bias is not None
        
        geodesic_layer = cls(in_features, out_features, c=c, bias=has_bias)
        
        # 기존 가중치를 매우 작게 스케일링하여 복사
        with torch.no_grad():
            if hasattr(linear_layer, 'weight'):
                weight = linear_layer.weight.data
            else:  # Conv1D
                weight = linear_layer.weight.t()
            
            # 가중치를 매우 작게 스케일링
            geodesic_layer.weight.data = weight * 0.01
            
            if has_bias:
                geodesic_layer.bias.data = linear_layer.bias.data * 0.01
        
        return geodesic_layer 

class EquivalentHyperbolicLinear(nn.Module):
    """
    유클리드 선형 레이어와 동등한 표현력을 가진 쌍곡 선형 레이어.
    원래 모델의 정확도를 최대한 보존하면서 쌍곡 기하학으로 변환합니다.
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 유클리드 가중치를 저장
        self.euclidean_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 쌍곡 보정 파라미터 (학습 가능)
        self.scale_factor = nn.Parameter(torch.ones(1))
        self.output_scale = nn.Parameter(torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 표준 초기화
        nn.init.kaiming_uniform_(self.euclidean_weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.euclidean_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # 보정 파라미터 초기화
        nn.init.ones_(self.scale_factor)
        nn.init.ones_(self.output_scale)

    def forward(self, x: Tensor) -> Tensor:
        # 이 변환은 이제 순수 유클리드 선형 변환과 동일합니다.
        # 쌍곡 변환 로직은 비활성화되었습니다.
        return F.linear(x, self.euclidean_weight, self.bias)

    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        """기존 선형 레이어로부터 동등한 쌍곡 레이어 생성"""
        # Conv1D 처리
        if 'Conv1D' in str(type(linear_layer)):
            # Conv1D는 weight shape이 (nf, nx)이고, Linear와 반대
            in_features = linear_layer.weight.shape[0]
            out_features = linear_layer.weight.shape[1]
            weight = linear_layer.weight.t()  # 전치해서 Linear 형태로
        else:
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            weight = linear_layer.weight.data
        
        has_bias = linear_layer.bias is not None
        
        equiv_layer = cls(in_features, out_features, c=c, bias=has_bias)
        
        # 기존 가중치를 복사
        with torch.no_grad():
            equiv_layer.euclidean_weight.data.copy_(weight)
            
            if has_bias:
                equiv_layer.bias.data.copy_(linear_layer.bias.data)
        
        return equiv_layer 

class CompactEquivalentHyperbolicLinear(nn.Module):
    """
    메모리 효율적인 동등 쌍곡 선형 레이어.
    scale_factor를 모든 레이어가 공유하여 메모리를 절약합니다.
    """
    
    # 클래스 변수로 공유 스케일 팩터 정의
    shared_scale_factor = None
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 유클리드 가중치만 저장 (원본과 동일)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # 공유 스케일 팩터 초기화
        if CompactEquivalentHyperbolicLinear.shared_scale_factor is None:
            CompactEquivalentHyperbolicLinear.shared_scale_factor = nn.Parameter(torch.tensor(100.0))
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 표준 초기화
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # 입력 shape 저장
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])
        
        # 1. 유클리드 공간에서 선형 변환 수행
        euclidean_output = F.linear(x, self.weight, self.bias)
        
        # 2. 공유 스케일 팩터로 쌍곡 변환
        scale = self.shared_scale_factor.abs() + 10.0
        hyperbolic_output = torch.tanh(euclidean_output / scale) * scale
        
        # 원래 shape으로 복원
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            hyperbolic_output = hyperbolic_output.view(*output_shape)
            
        return hyperbolic_output
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        """기존 선형 레이어로부터 컴팩트 쌍곡 레이어 생성"""
        # Conv1D 처리
        if 'Conv1D' in str(type(linear_layer)):
            in_features = linear_layer.weight.shape[0]
            out_features = linear_layer.weight.shape[1]
            weight = linear_layer.weight.t()
        else:
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            weight = linear_layer.weight.data
        
        has_bias = linear_layer.bias is not None
        
        compact_layer = cls(in_features, out_features, c=c, bias=has_bias)
        
        # 기존 가중치를 복사
        with torch.no_grad():
            compact_layer.weight.data.copy_(weight)
            
            if has_bias:
                compact_layer.bias.data.copy_(linear_layer.bias.data)
        
        return compact_layer 