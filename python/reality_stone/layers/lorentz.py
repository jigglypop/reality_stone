import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda
from .poincare import poincare_to_lorentz

class LorentzDistance(Function):
    """
    로렌츠 거리(Lorentz Distance)를 계산하는 Autograd Function입니다.
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float) -> Tensor:
        ctx.c = c
        ctx.save_for_backward(u, v)
        
        if u.is_cuda and _has_cuda:
            # CUDA 구현 사용
            output = torch.empty(u.shape[0], dtype=u.dtype, device=u.device)
            _rust.lorentz_distance_cuda(
                output.data_ptr(), u.data_ptr(), v.data_ptr(),
                c, u.shape[0], u.shape[1]
            )
            return output
        else:
            # CPU 구현 사용
            result_np = _rust.lorentz_distance(u.detach().cpu().numpy(), v.detach().cpu().numpy(), c)
            return torch.from_numpy(result_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        u, v = ctx.saved_tensors
        c = ctx.c
        
        # 로렌츠 거리의 해석적 역전파 구현
        # d(dist) = acosh(z) / sqrt(c) where z = c * <u,v>_L
        
        # 역전파를 위한 재계산
        # <u, v>_L = u0v0 - u.v
        inner = u[..., 0] * v[..., 0] - (u[..., 1:] * v[..., 1:]).sum(dim=-1)
        z = (c * inner).clamp(min=1.0 + 1e-7)
        
        # dist = acosh(z) / sqrt(c)
        # d(dist)/dz = 1 / (sqrt(c) * sqrt(z^2 - 1))
        sqrt_c = c**0.5
        d_dist_dz = 1.0 / (sqrt_c * torch.sqrt(z*z - 1.0))
        
        # d(z)/du = c * d(<u,v>)/du
        # 내적의 미분 (민코프스키 계량 고려)
        
        grad_z_u = torch.empty_like(v)
        grad_z_u[..., 0] = c * v[..., 0]
        grad_z_u[..., 1:] = -c * v[..., 1:]
        
        grad_z_v = torch.empty_like(u)
        grad_z_v[..., 0] = c * u[..., 0]
        grad_z_v[..., 1:] = -c * u[..., 1:]
        
        # 연쇄 법칙
        scale = (grad_output * d_dist_dz).unsqueeze(-1)
        grad_u = scale * grad_z_u
        grad_v = scale * grad_z_v
        
        return grad_u, grad_v, None

def lorentz_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    """
    로렌츠 거리를 계산합니다. Reality Stone의 최적화된 커널을 사용합니다.
    """
    return LorentzDistance.apply(x, y, c)


class LorentzLayer(Function):
    """
    로렌츠 레이어 (지오데식 보간) Function
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float, t: float) -> Tensor:
        ctx.c = c
        ctx.t = t
        ctx.save_for_backward(u, v)
        if u.is_cuda and _has_cuda:
            output = torch.empty_like(u)
            _rust.lorentz_layer_forward_cuda(
                output.data_ptr(), u.data_ptr(), v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
            return output
        else:
            output_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), c, t)
            return torch.from_numpy(output_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        grad_u = grad_v = None
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.lorentz_ball_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.lorentz_ball_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        return grad_u, grad_v, None, None

def lorentz_add(u: Tensor, v: Tensor, c: float) -> Tensor:
    """
    로렌츠 덧셈 (자이로벡터 합)
    """
    result_np = _rust.lorentz_add(u.cpu().numpy(), v.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(u.device)

def lorentz_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    """
    로렌츠 스칼라 곱
    """
    result_np = _rust.lorentz_scalar(x.cpu().numpy(), r, c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_inner(u: Tensor, v: Tensor) -> Tensor:
    """
    로렌츠 민코프스키 내적
    """
    result_np = _rust.lorentz_inner(u.cpu().numpy(), v.cpu().numpy())
    return torch.from_numpy(result_np).to(u.device)

def lorentz_to_poincare(x: Tensor, c: float) -> Tensor:
    """
    로렌츠 -> 푸앵카레 변환
    """
    result_np = _rust.lorentz_to_poincare(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_to_klein(x: Tensor, c: float) -> Tensor:
    """
    로렌츠 -> 클라인 변환
    """
    result_np = _rust.lorentz_to_klein(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device) 

def euclidean_to_lorentz(x: Tensor, c: float = 1.0, epsilon: float = 1e-6) -> Tensor:
    """
    유클리드 벡터 x를 로렌츠 쌍곡면으로 들어올립니다(Lift).
    R^n의 x를 R^{n+1}의 부분집합인 H^n의 (sqrt(1/c + ||x||^2), x)로 매핑합니다.
    """
    # sq = ||x||^2
    sq = (x * x).sum(dim=-1, keepdim=True)
    # x0 = sqrt(1/c + ||x||^2)
    time_comp = torch.sqrt(torch.clamp(1.0 / c + sq, min=epsilon))
    return torch.cat([time_comp, x], dim=-1) 


class LorentzBallLayer(Function):
    """
    로렌츠 공 레이어 (Lorentz Ball Layer)
    
    동적 곡률(Dynamic Curvature)을 지원하며, 로렌츠 모델 위에서 두 점을 보간합니다.
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = 0.1, c_max: float = 5.0) -> Tensor:
        ctx.t = t
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
            # 네이티브 바인딩이 있다면 우선 사용
            if hasattr(_rust, 'lorentz_layer_layerwise_cpu'):
                out_np, c_val = _rust.lorentz_layer_layerwise_cpu(
                    u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
                )
                ctx.c_val = c_val
                return torch.from_numpy(out_np).to(u.device)
            else:
                # Python fallback
                sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
                c_val = c_min + (c_max - c_min) * sig.item()
                ctx.c_val = c_val
                out_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), c_val, t)
                return torch.from_numpy(out_np).to(u.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            ctx.save_for_backward(u.clone(), v.clone())
            out_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), ctx.c, t)
            return torch.from_numpy(out_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        t = ctx.t
        if ctx.use_dynamic:
            u, v, kappas = ctx.saved_tensors
            layer_idx = ctx.layer_idx
            c_min = ctx.c_min
            c_max = ctx.c_max
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
            c_val = getattr(ctx, 'c_val', None)
            if c_val is None:
                sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
                c_val = (c_min + (c_max - c_min) * sig.item())
                ctx.c_val = c_val
            
            # 정적 역전파를 통한 u, v 그라디언트 계산
            if grad_output.is_cuda and _has_cuda:
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.lorentz_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    float(c_val), t, u.shape[0], u.shape[1]
                )
            else:
                gu_np, gv_np = _rust.lorentz_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), float(c_val), t
                )
                grad_u = torch.from_numpy(gu_np).to(grad_output.device)
                grad_v = torch.from_numpy(gv_np).to(grad_output.device)

            # kappa에 대한 정확한 그라디언트 (연쇄 법칙)
            def minkowski_inner(p: Tensor, q: Tensor) -> Tensor:
                return p[..., :1]*q[..., :1] - (p[..., 1:]*q[..., 1:]).sum(dim=-1, keepdim=True)

            eps = 1e-7
            inner = minkowski_inner(u, v)  # (B,1)
            z = torch.clamp_min(-float(c_val) * inner, 1.0 + eps)
            alpha = torch.acosh(z)
            sinh_a = torch.sinh(alpha).clamp_min(eps)
            cosh_a = torch.cosh(alpha)

            t1 = (1.0 - t) * alpha
            t2 = t * alpha
            w1 = torch.where(alpha.abs() < 1e-6, torch.full_like(alpha, 1.0 - t), torch.sinh(t1) / sinh_a)
            w2 = torch.where(alpha.abs() < 1e-6, torch.full_like(alpha, t), torch.sinh(t2) / sinh_a)

            num1 = (1.0 - t) * torch.cosh(t1) * sinh_a - torch.sinh(t1) * cosh_a
            num2 = t * torch.cosh(t2) * sinh_a - torch.sinh(t2) * cosh_a
            denom = (sinh_a * sinh_a).clamp_min(eps)
            dw1_da = torch.where(alpha.abs() < 1e-6, torch.zeros_like(alpha), num1 / denom)
            dw2_da = torch.where(alpha.abs() < 1e-6, torch.zeros_like(alpha), num2 / denom)

            dalpha_dz = 1.0 / (torch.sqrt(torch.clamp_min(z+1.0, 1.0+eps)) * torch.sqrt(torch.clamp_min(z-1.0, eps)))
            dz_dc = -inner
            dalpha_dc = dalpha_dz * dz_dc

            dw1_dc = dw1_da * dalpha_dc
            dw2_dc = dw2_da * dalpha_dc

            # dy/dc = dw1_dc * u + dw2_dc * v
            dy_dc = dw1_dc * u + dw2_dc * v
            grad_c_total = (grad_output * dy_dc).sum()
            sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val, dtype=torch.float32, device=grad_output.device)))
            dc_dkappa = (c_max - c_min) * sig * (1.0 - sig)
            gk_val = (grad_c_total * dc_dkappa).item()

            if kappas.dim() == 0:
                grad_kappas = torch.tensor(gk_val, device=kappas.device)
            else:
                grad_kappas = torch.zeros_like(kappas)
                grad_kappas[layer_idx] = gk_val
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        else:
            u, v = ctx.saved_tensors
            c = ctx.c
            if grad_output.is_cuda and _has_cuda:
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.lorentz_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    c, t, u.shape[0], u.shape[1]
                )
                return grad_u, grad_v, None, None, None, None, None, None
            else:
                gu_np, gv_np = _rust.lorentz_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
                )
                grad_u = torch.from_numpy(gu_np).to(grad_output.device)
                grad_v = torch.from_numpy(gv_np).to(grad_output.device)
                return grad_u, grad_v, None, None, None, None, None, None


def lorentz_ball(u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = 0.1, c_max: float = 5.0) -> Tensor:
    """
    로렌츠 공 레이어 편의 함수
    """
    return LorentzBallLayer.apply(u, v, c, t, kappas, layer_idx, c_min, c_max)

class LorentzFromPoincare(Function):
    """
    푸앵카레 -> 로렌츠 변환 (Autograd 지원, 동적 곡률 지원)
    """
    @staticmethod
    def forward(ctx, x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        if kappas is not None:
            ctx.use_dynamic = True
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(x, kappas)
            
            output_np, c_val = _rust.from_poincare_dynamic_cpu(
                x.cpu().numpy(), kappas.item(), c_min, c_max
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(x.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            # 정적 경로
            output = poincare_to_lorentz(x, ctx.c)
            ctx.save_for_backward(x)
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, kappas = ctx.saved_tensors
            grad_x_np, grad_kappa_val = _rust.from_poincare_dynamic_backward_cpu(
                grad_output.cpu().numpy(), x.cpu().numpy(), kappas.item(), ctx.c_min, ctx.c_max
            )
            grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
            grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            return grad_x, None, grad_kappas, None, None
        else:
            x, = ctx.saved_tensors
            grad_x = torch.zeros_like(x)
            return grad_x, None, None, None, None

def from_poincare(x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
    """
    푸앵카레 -> 로렌츠 변환 편의 함수
    """
    return LorentzFromPoincare.apply(x, c, kappas, c_min, c_max)
