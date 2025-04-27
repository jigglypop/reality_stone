from torch.autograd import Function
import torch
from . import _C

# 확장 모듈에서 함수를 가져옵니다
try:
    from ._C import (
        add_tensors, 
        poincare_exp_map, 
        poincare_log_map, 
        poincare_distance,
        butterfly_factor,
        hyper_butterfly_forward,
        is_cuda_available
    )
    HAS_CPP_EXTENSION = True
    HAS_CUDA_EXTENSION = False # CUDA 구현 없음
    print("C++ 확장이 성공적으로 로드되었습니다. (CPU 구현)")
except ImportError:
    HAS_CPP_EXTENSION = False
    HAS_CUDA_EXTENSION = False
    print("C++ 확장을 로드할 수 없습니다. 순수 Python 구현을 사용합니다.")

# 순수 Python 폴백 구현
def py_add_tensors(a, b):
    return a + b

def py_poincare_exp_map(x, v, c=1.0):
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / (1.0 - c * x_norm_squared)
    
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=1e-8)
    
    second_term = torch.tanh(torch.sqrt(torch.tensor(c, device=x.device)) * lambda_x * v_norm / 2.0) / (torch.sqrt(torch.tensor(c, device=x.device)) * v_norm) * v
    
    numerator = (1.0 - c * x_norm_squared) * second_term
    denominator = 1.0 - 2.0 * c * torch.sum(x * second_term, dim=-1, keepdim=True) + c * c * x_norm_squared * torch.sum(second_term * second_term, dim=-1, keepdim=True)
    
    return x + numerator / denominator

def py_poincare_log_map(x, y, c=1.0):
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / (1.0 - c * x_norm_squared)
    
    diff = y - x
    diff_norm_squared = torch.sum(diff * diff, dim=-1, keepdim=True)
    y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
    
    transport_vector = (-x * y_norm_squared + y * (1.0 + c * x_norm_squared) - 2 * c * torch.sum(x * y, dim=-1, keepdim=True) * x) / (1.0 - c * x_norm_squared)
    transport_norm = torch.norm(transport_vector, p=2, dim=-1, keepdim=True)
    
    numerator = 2 * torch.sqrt(torch.tensor(c, device=x.device)) * torch.atanh(torch.sqrt(torch.tensor(c, device=x.device)) * transport_norm)
    denominator = torch.sqrt(torch.tensor(c, device=x.device)) * lambda_x * transport_norm
    
    return numerator / denominator * transport_vector

def py_poincare_distance(x, y, c=1.0):
    norm_x = torch.sum(x * x, dim=-1, keepdim=True)
    norm_y = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
    
    numerator = 2 * torch.sqrt(torch.tensor(c, device=x.device)) * torch.norm(x - y, p=2, dim=-1, keepdim=True)
    denominator = torch.sqrt((1 - c * norm_x) * (1 - c * norm_y)) + torch.sqrt(torch.tensor(c, device=x.device)) * xy_inner
    
    return 2 * torch.atanh(numerator / denominator) / torch.sqrt(torch.tensor(c, device=x.device))

def py_butterfly_factor(input_tensor, params, layer):
    n = input_tensor.size(0)
    block_size = 1 << layer
    num_blocks = n // block_size
    
    result = input_tensor.clone()
    
    param_idx = 0
    total_params = params.size(0)
    
    for b in range(num_blocks):
        for i in range(0, block_size, 2):
            if b * block_size + i + 1 >= n:
                break
            if param_idx + 1 >= total_params:
                break
                
            idx = b * block_size + i
            a = params[param_idx].item()
            b_val = params[param_idx + 1].item()
            param_idx += 2
            
            temp1 = a * input_tensor[idx] + b_val * input_tensor[idx + 1]
            temp2 = -b_val * input_tensor[idx] + a * input_tensor[idx + 1]
            
            result[idx] = temp1
            result[idx + 1] = temp2
    
    return result

def py_hyper_butterfly_forward(x, params, c, L):
    zeros = torch.zeros_like(x)
    u = py_poincare_log_map(zeros, x, c)
    
    param_idx = 0
    for l in range(L):
        if param_idx >= params.size(0):
            break
        
        layer_params = params[param_idx:].clone()
        u = py_butterfly_factor(u, layer_params, l)
    
    return py_poincare_exp_map(zeros, u, c)

# 실제 사용 함수
def exp_map(x, v, c=1.0):
    # GPU 텐서는 먼저 CPU로 이동
    if x.is_cuda:
        print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
        x_cpu = x.cpu()
        v_cpu = v.cpu()
        if HAS_CPP_EXTENSION:
            result = poincare_exp_map(x_cpu, v_cpu, c)
        else:
            result = py_poincare_exp_map(x_cpu, v_cpu, c)
        return result.to(x.device)
    else:
        if HAS_CPP_EXTENSION:
            return poincare_exp_map(x, v, c)
        else:
            return py_poincare_exp_map(x, v, c)

def log_map(x, y, c=1.0):
    # GPU 텐서는 먼저 CPU로 이동
    if x.is_cuda:
        print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        if HAS_CPP_EXTENSION:
            result = poincare_log_map(x_cpu, y_cpu, c)
        else:
            result = py_poincare_log_map(x_cpu, y_cpu, c)
        return result.to(x.device)
    else:
        if HAS_CPP_EXTENSION:
            return poincare_log_map(x, y, c)
        else:
            return py_poincare_log_map(x, y, c)

def distance(x, y, c=1.0):
    # GPU 텐서는 먼저 CPU로 이동
    if x.is_cuda:
        print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        if HAS_CPP_EXTENSION:
            result = poincare_distance(x_cpu, y_cpu, c)
        else:
            result = py_poincare_distance(x_cpu, y_cpu, c)
        return result.to(x.device)
    else:
        if HAS_CPP_EXTENSION:
            return poincare_distance(x, y, c)
        else:
            return py_poincare_distance(x, y, c)

def butterfly_transform(x, params, layer):
    # GPU 텐서는 먼저 CPU로 이동
    if x.is_cuda:
        print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
        x_cpu = x.cpu()
        params_cpu = params.cpu()
        if HAS_CPP_EXTENSION:
            result = butterfly_factor(x_cpu, params_cpu, layer)
        else:
            result = py_butterfly_factor(x_cpu, params_cpu, layer)
        return result.to(x.device)
    else:
        if HAS_CPP_EXTENSION:
            return butterfly_factor(x, params, layer)
        else:
            return py_butterfly_factor(x, params, layer)

def hyper_butterfly(x, params, c, L):
    # GPU 텐서는 먼저 CPU로 이동
    if x.is_cuda:
        print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
        x_cpu = x.cpu()
        params_cpu = params.cpu()
        if HAS_CPP_EXTENSION:
            result = hyper_butterfly_forward(x_cpu, params_cpu, c, L)
        else:
            result = py_hyper_butterfly_forward(x_cpu, params_cpu, c, L)
        return result.to(x.device)
    else:
        if HAS_CPP_EXTENSION:
            return hyper_butterfly_forward(x, params, c, L)
        else:
            return py_hyper_butterfly_forward(x, params, c, L)

# CUDA 최적화된 하이퍼볼릭 버터플라이 층
class HyperButterflyLayer(torch.nn.Module):
    def __init__(self, dim, num_layers, curvature=1.0):
        super(HyperButterflyLayer, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.curvature = curvature
        
        # 차원이 2의 거듭제곱인지 확인하고 조정
        power = int(torch.ceil(torch.log2(torch.tensor(float(dim)))))
        self.adjusted_dim = 2 ** power
        
        # 각 버터플라이 레이어의 파라미터 초기화 (a, b)
        # 회전 행렬 조건: a^2 + b^2 = 1
        params = []
        for l in range(num_layers):
            block_size = 1 << l
            num_blocks = self.adjusted_dim // block_size
            for b in range(num_blocks // 2):  # 각 블록은 2x2 회전
                # cos(θ), sin(θ) 형태로 초기화
                theta = torch.rand(1) * 0.1  # 작은 초기 회전
                a = torch.nn.Parameter(torch.cos(theta))
                b = torch.nn.Parameter(torch.sin(theta))
                params.extend([a, b])
        
        self.params = torch.nn.ParameterList(params)
    
    def forward(self, x):
        # 입력 크기가 조정된 차원보다 작으면 패딩
        if x.size(-1) < self.adjusted_dim:
            padding = torch.zeros(*x.shape[:-1], self.adjusted_dim - x.size(-1), device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x
        
        # 파라미터 텐서 구성
        params_tensor = torch.cat([p for p in self.params])
        
        # GPU 감지 및 처리
        device = x.device
        
        # 하이퍼볼릭 버터플라이 변환 적용
        output = hyper_butterfly(x_padded, params_tensor, self.curvature, self.num_layers)
        
        # 원래 차원으로 복원
        if x.size(-1) < self.adjusted_dim:
            output = output[..., :x.size(-1)]
        
        # 결과를 원래 디바이스로 전송
        if device.type == 'cuda':
            output = output.to(device)
        
        return output