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
        hyper_butterfly_forward
    )
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    print("C++ 확장을 로드할 수 없습니다. 순수 Python 구현을 사용합니다.")

# 순수 Python 폴백 구현
def py_add_tensors(a, b):
    return a + b

def py_poincare_exp_map(x, v, c=1.0):
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / (1.0 - c * x_norm_squared)
    
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=1e-8)
    
    second_term = torch.tanh(torch.sqrt(torch.tensor(c)) * lambda_x * v_norm / 2.0) / (torch.sqrt(torch.tensor(c)) * v_norm) * v
    
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
    
    numerator = 2 * torch.sqrt(torch.tensor(c)) * torch.atanh(torch.sqrt(torch.tensor(c)) * transport_norm)
    denominator = torch.sqrt(torch.tensor(c)) * lambda_x * transport_norm
    
    return numerator / denominator * transport_vector

def py_poincare_distance(x, y, c=1.0):
    norm_x = torch.sum(x * x, dim=-1, keepdim=True)
    norm_y = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
    
    numerator = 2 * torch.sqrt(torch.tensor(c)) * torch.norm(x - y, p=2, dim=-1, keepdim=True)
    denominator = torch.sqrt((1 - c * norm_x) * (1 - c * norm_y)) + torch.sqrt(c) * xy_inner
    
    return 2 * torch.atanh(numerator / denominator) / torch.sqrt(torch.tensor(c))

def py_butterfly_factor(input_tensor, params, layer):
    n = input_tensor.size(0)
    block_size = 1 << layer
    num_blocks = n // block_size
    
    result = input_tensor.clone()
    
    for b in range(num_blocks):
        for i in range(0, block_size, 2):
            idx = b * block_size + i
            a = params[b * 2].item()
            b_val = params[b * 2 + 1].item()
            
            temp1 = a * input_tensor[idx] + b_val * input_tensor[idx + 1]
            temp2 = -b_val * input_tensor[idx] + a * input_tensor[idx + 1]
            
            result[idx] = temp1
            result[idx + 1] = temp2
    
    return result

def py_hyper_butterfly_forward(x, params, c, L):
    u = py_poincare_log_map(torch.zeros_like(x), x, c)
    
    for l in range(L):
        u = py_butterfly_factor(u, params[l*2:(l+1)*2], l)
    
    return py_poincare_exp_map(torch.zeros_like(u), u, c)

# 실제 사용 함수
def exp_map(x, v, c=1.0):
    if HAS_CPP_EXTENSION:
        return poincare_exp_map(x, v, c)
    else:
        return py_poincare_exp_map(x, v, c)

def log_map(x, y, c=1.0):
    if HAS_CPP_EXTENSION:
        return poincare_log_map(x, y, c)
    else:
        return py_poincare_log_map(x, y, c)

def distance(x, y, c=1.0):
    if HAS_CPP_EXTENSION:
        return poincare_distance(x, y, c)
    else:
        return py_poincare_distance(x, y, c)

def butterfly_transform(x, params, layer):
    if HAS_CPP_EXTENSION:
        return butterfly_factor(x, params, layer)
    else:
        return py_butterfly_factor(x, params, layer)

def hyper_butterfly(x, params, c, L):
    if HAS_CPP_EXTENSION:
        return hyper_butterfly_forward(x, params, c, L)
    else:
        return py_hyper_butterfly_forward(x, params, c, L)

class HyperButterflyLayer(torch.nn.Module):
    def __init__(self, dim, num_layers, curvature=1.0):
        super(HyperButterflyLayer, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.curvature = curvature
        
        # 각 버터플라이 레이어의 파라미터 초기화 (a, b)
        # 각 a^2 - b^2 >= delta > 0를 유지하기 위해 특별한 초기화 사용
        params = []
        for l in range(num_layers):
            block_size = 1 << l
            num_blocks = dim // block_size
            for b in range(num_blocks):
                a = torch.nn.Parameter(torch.ones(1) * 0.9)  # a 초기값
                b = torch.nn.Parameter(torch.zeros(1))       # b 초기값
                params.extend([a, b])
        
        self.params = torch.nn.ParameterList(params)
    
    def forward(self, x):
        params_tensor = torch.cat([p for p in self.params])
        return hyper_butterfly(x, params_tensor, self.curvature, self.num_layers)