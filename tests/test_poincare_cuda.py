import pytest
import torch
import reality_stone as rs
from reality_stone.layers.klein import project_to_klein


@pytest.mark.cuda
def test_has_cuda_flag_matches_torch():
    """
    reality_stone._has_cuda 플래그가
    PyTorch CUDA 가능 여부 및 CUDA 심볼 로딩 상태와 일관되는지 확인.
    """
    if not torch.cuda.is_available():
        # CUDA 없는 환경에서는 _has_cuda 가 False 여야 한다.
        assert rs._has_cuda is False
        return

    # CUDA 환경에서는 최소한 rust 확장이 있고, _has_cuda 가 True 여야 한다.
    assert rs._has_rust_ext, "Rust extension must be available when CUDA is used"
    assert rs._has_cuda, "Reality Stone CUDA bindings not detected"


@pytest.mark.cuda
def test_poincare_ball_layer_cpu_cuda_consistency():
    """
    Poincaré ball layer 의 CPU / CUDA 결과가 충분히 근접하는지 검증.

    - 입력: 랜덤 B x D 텐서 (Poincaré ball 내부로 투영)
    - 경로:
        rs.poincare_ball_layer (autograd) → CPU / CUDA 분기
    """
    if not (torch.cuda.is_available() and rs._has_cuda):
        pytest.skip("CUDA or Reality Stone CUDA bindings are not available")

    torch.manual_seed(42)
    B, d = 8, 16
    c = 1.0
    t = 0.5

    # 간단한 Poincaré ball 투영
    def project_to_ball(x, epsilon=1e-5):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        max_norm = 1.0 - epsilon
        scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * scale

    h_cpu = project_to_ball(torch.randn(B, d))
    u_cpu = project_to_ball(torch.randn(B, d))

    # CPU 경로
    z_cpu = rs.poincare_ball_layer(h_cpu, u_cpu, c=c, t=t)

    # CUDA 경로
    h_cuda = h_cpu.cuda()
    u_cuda = u_cpu.cuda()
    z_cuda = rs.poincare_ball_layer(h_cuda, u_cuda, c=c, t=t)

    max_diff = torch.max(torch.abs(z_cpu - z_cuda.cpu())).item()
    # Allow a small numerical tolerance between CPU and CUDA paths
    assert max_diff < 5e-5, f"Poincaré CPU/CUDA mismatch: max_diff={max_diff:.3e}"


@pytest.mark.cuda
def test_lorentz_layer_cpu_cuda_consistency_forward_backward():
    """
    Lorentz layer 의 CPU / CUDA 순전파와 역전파 결과가 충분히 근접하는지 검증.
    """
    if not (torch.cuda.is_available() and rs._has_cuda):
        pytest.skip("CUDA or Reality Stone CUDA bindings are not available")

    torch.manual_seed(42)
    B, dim = 4, 5  # Minkowski: time + (dim-1) space
    c = 1.0
    t = 0.3

    def sample_lorentz(batch: int, d: int, device: torch.device) -> torch.Tensor:
        # Generate points on the Lorentz hyperboloid: x0^2 - ||x||^2 = 1
        spatial = torch.randn(batch, d - 1, device=device) * 0.1
        sq = (spatial * spatial).sum(dim=1, keepdim=True)
        time = torch.sqrt(1.0 + sq)
        return torch.cat([time, spatial], dim=1)

    u_cpu = sample_lorentz(B, dim, device=torch.device("cpu")).requires_grad_(True)
    v_cpu = sample_lorentz(B, dim, device=torch.device("cpu")).requires_grad_(True)

    y_cpu = rs.lorentz_layer(u_cpu, v_cpu, c=c, t=t)

    u_cuda = u_cpu.detach().clone().cuda().requires_grad_(True)
    v_cuda = v_cpu.detach().clone().cuda().requires_grad_(True)
    y_cuda = rs.lorentz_layer(u_cuda, v_cuda, c=c, t=t)

    # Forward consistency
    max_diff_fwd = torch.max(torch.abs(y_cpu - y_cuda.cpu())).item()
    assert max_diff_fwd < 1e-4, f"Lorentz layer forward CPU/CUDA mismatch: max_diff={max_diff_fwd:.3e}"

    # Backward consistency
    grad = torch.randn_like(y_cpu)
    y_cpu.backward(grad)
    y_cuda.backward(grad.cuda())

    max_diff_gu = torch.max(torch.abs(u_cpu.grad - u_cuda.grad.cpu())).item()
    max_diff_gv = torch.max(torch.abs(v_cpu.grad - v_cuda.grad.cpu())).item()
    max_grad_diff = max(max_diff_gu, max_diff_gv)
    assert max_grad_diff < 1e-3, f"Lorentz layer backward CPU/CUDA mismatch: max_diff={max_grad_diff:.3e}"


@pytest.mark.cuda
def test_klein_layer_cpu_cuda_consistency_forward_backward():
    """
    Klein layer 의 CPU / CUDA 순전파와 역전파 결과가 충분히 근접하는지 검증.
    """
    if not (torch.cuda.is_available() and rs._has_cuda):
        pytest.skip("CUDA or Reality Stone CUDA bindings are not available")

    torch.manual_seed(42)
    B, d = 4, 4
    c = 1.0
    t = 0.3

    # Project random vectors safely into the Klein disk for curvature c.
    u_base = torch.randn(B, d)
    v_base = torch.randn(B, d)
    u_cpu = project_to_klein(u_base, c).requires_grad_(True)
    v_cpu = project_to_klein(v_base, c).requires_grad_(True)

    y_cpu = rs.klein_layer(u_cpu, v_cpu, c=c, t=t)

    u_cuda = u_cpu.detach().clone().cuda().requires_grad_(True)
    v_cuda = v_cpu.detach().clone().cuda().requires_grad_(True)
    y_cuda = rs.klein_layer(u_cuda, v_cuda, c=c, t=t)

    # Forward consistency
    max_diff_fwd = torch.max(torch.abs(y_cpu - y_cuda.cpu())).item()
    assert max_diff_fwd < 1e-4, f"Klein layer forward CPU/CUDA mismatch: max_diff={max_diff_fwd:.3e}"

    # Backward consistency
    grad = torch.randn_like(y_cpu)
    y_cpu.backward(grad)
    y_cuda.backward(grad.cuda())

    max_diff_gu = torch.max(torch.abs(u_cpu.grad - u_cuda.grad.cpu())).item()
    max_diff_gv = torch.max(torch.abs(v_cpu.grad - v_cuda.grad.cpu())).item()
    max_grad_diff = max(max_diff_gu, max_diff_gv)
    assert max_grad_diff < 1e-3, f"Klein layer backward CPU/CUDA mismatch: max_diff={max_grad_diff:.3e}"



