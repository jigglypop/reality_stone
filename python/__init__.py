import os
import sys
from sympy import Function
import torch
import ctypes

# Windows DLL 로딩 문제 해결
if sys.platform == 'win32':
    # PyTorch lib 디렉토리
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    
    # Python 3.8+ DLL 디렉토리 추가
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(torch_lib_path):
            # CUDA DLL 사전 로딩
            cuda_dlls = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
            for dll_name in cuda_dlls:
                dll_path = os.path.join(torch_lib_path, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception as e:
                        print(f"Warning: Failed to load {dll_name}: {e}")
    else:
        # Python 3.7 이하
        os.environ['PATH'] = torch_lib_path + ';' + os.environ.get('PATH', '')

# 디버그 정보
print(f"Attempting to import _C from: {os.path.dirname(__file__)}")

try:
    from ._C import (
        log_map_cpu,
        exp_map_cpu,
        poincare_forward_cpu,
        geodesic_cpu,
        geodesic_forward_cpu,
        geodesic_backward_cpu,
    )
    print("CPU functions imported successfully")
except ImportError as e:
    print(f"Failed to import CPU functions: {e}")
    raise

# CUDA 함수들은 별도로 import
_has_cuda = False
if torch.cuda.is_available():
    print("CUDA is available")
    try:
        from ._C import (
            log_map_cuda,
            exp_map_cuda,
            log_map_forward_cuda,
            exp_map_forward_cuda,
            poincare_forward_cuda,
            poincare_backward_cuda,
            geodesic_cuda,
            geodesic_forward_cuda,
            geodesic_backward_cuda,
        )
        _has_cuda = True
        print("CUDA functions imported successfully")
    except ImportError as e:
        print(f"Failed to import CUDA functions: {e}")
        _has_cuda = False

# 나머지 import들...
from .maps import log_map, exp_map, geodesic
from .layers import HyperButterflyFunction, GeodesicButterflyLayer

def reality_stone(x: torch.Tensor, params: torch.Tensor, c: float, L: int):
    return HyperButterflyFunction.apply(x, params, c, L)

def geodesic_butterfly(x: torch.Tensor,params: torch.Tensor,c: float,L: int,t: float) -> torch.Tensor:
    layer = GeodesicButterflyLayer(x.size(1), c, L, t)
    return layer(x) 
# autograd를 사용하는 버전으로 변경
class GeodesicFunction(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # forward만 수행하고 requires_grad 처리
        with torch.enable_grad():
            if u.is_cuda and _has_cuda:
                result = geodesic_forward_cuda(u, v, c, t)
            else:
                result = geodesic_forward_cpu(u, v, c, t)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        
        # autograd를 사용하여 backprop 계산
        with torch.enable_grad():
            u_new = u.detach().requires_grad_(True)
            v_new = v.detach().requires_grad_(True)
            
            if u_new.is_cuda and _has_cuda:
                output = geodesic_forward_cuda(u_new, v_new, c, t)
            else:
                output = geodesic_forward_cpu(u_new, v_new, c, t)
            
            grad_u, grad_v = torch.autograd.grad(
                output, 
                [u_new, v_new], 
                grad_outputs=grad_output,
                retain_graph=True
            )
        
        return grad_u, grad_v, None, None

def geodesic_layer(u, v, c, t):
    return GeodesicFunction.apply(u, v, c, t)
# class GeodesicFunction(Function):
#     @staticmethod
#     def forward(ctx, u, v, c, t):
#         if u.is_cuda and _has_cuda:
#             result = geodesic_forward_cuda(u, v, c, t)
#         else:
#             result = geodesic_forward_cpu(u, v, c, t)
#         ctx.save_for_backward(u, v)
#         ctx.c = c
#         ctx.t = t
#         return result
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         u, v = ctx.saved_tensors
#         c, t = ctx.c, ctx.t
#         print(_has_cuda)
#         grad_u, grad_v = geodesic_backward_cuda(grad_output, u, v, c, t)
#         # else:
#         #     # CPU 버전이 문제이므로 대안 사용
#         #     grad_u = torch.zeros_like(u)
#         #     grad_v = torch.zeros_like(v)
#         #     
#         #     # 단순한 그래디언트 근사
#         #     grad_u = grad_output * (1.0 - t)
#         #     grad_v = grad_output * t
#         return grad_u, grad_v, None, None
# 
# def geodesic_layer(u, v, c, t):
#     return GeodesicFunction.apply(u, v, c, t)