import pytest
import torch
import reality_stone as rs


@pytest.mark.cuda
def test_rust_extension_loaded_when_cuda_available():
    """
    CUDA 환경에서는 Rust 확장이 로드되어 있어야 한다.

    이 테스트는 빌드/배포 과정에서 `_rust` 모듈이 누락되는 경우를 빠르게 잡기 위한 것이다.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")
    assert rs._has_rust_ext, "Rust extension must be available when CUDA is used"
    assert hasattr(rs, "_rust"), "reality_stone must expose `_rust` module"


@pytest.mark.cuda
def test_required_cuda_symbols_exist_on_rust_module():
    """
    reality_stone.__init__ 에서 사용하는 CUDA 바인딩 심볼들이
    실제 Rust 모듈에도 모두 존재하는지 검증한다.

    - 누락된 심볼이 있으면 `_has_cuda` 가 False 가 되어 CUDA 경로 전체가 비활성화된다.
    - 이 테스트로 '바인딩은 구현됐는데 __init__ 이 다른 이름을 보고 있다' 같은 실수를 방지한다.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    assert rs._has_rust_ext, "Rust extension must be available when CUDA is used"

    required_cuda_symbols = [
        # Möbius
        "mobius_add_cuda",
        "mobius_scalar_cuda",
        # Poincaré
        "poincare_ball_layer_cuda",
        "poincare_ball_layer_backward_cuda",
        "poincare_distance_cuda",
        # Lorentz
        "lorentz_layer_forward_cuda",
        "lorentz_ball_layer_backward_cuda",
        "lorentz_distance_cuda",
        # Klein
        "klein_layer_forward_cuda",
        "klein_ball_layer_backward_cuda",
        "klein_distance_cuda",
    ]

    missing = [name for name in required_cuda_symbols if not hasattr(rs._rust, name)]
    assert (
        not missing
    ), f"Missing CUDA bindings on rs._rust: {missing}. Check Rust bindings and __init__._has_cuda list."


@pytest.mark.cuda
def test_has_cuda_flag_consistent_with_symbols_and_torch():
    """
    `_has_cuda` 플래그가 PyTorch / 심볼 상태와 일관되는지 확인.

    - torch.cuda.is_available() == False 이면 _has_cuda 도 False
    - torch.cuda.is_available() == True 이면, 필수 심볼이 모두 있을 때만 _has_cuda 가 True
    """
    if not torch.cuda.is_available():
        assert rs._has_cuda is False
        return

    required_cuda_symbols = [
        "mobius_add_cuda",
        "mobius_scalar_cuda",
        "poincare_ball_layer_cuda",
        "poincare_ball_layer_backward_cuda",
        "poincare_distance_cuda",
        "lorentz_layer_forward_cuda",
        "lorentz_ball_layer_backward_cuda",
        "lorentz_distance_cuda",
        "klein_layer_forward_cuda",
        "klein_ball_layer_backward_cuda",
        "klein_distance_cuda",
    ]
    all_symbols_present = all(hasattr(rs._rust, name) for name in required_cuda_symbols)

    # CUDA 환경에서는 심볼이 모두 있으면 True, 하나라도 없으면 False 여야 한다.
    assert rs._has_cuda == all_symbols_present


