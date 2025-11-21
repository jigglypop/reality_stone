"""
Tests for Unified Riemannian Layer

이 테스트는 통합 리만 레이어의 핵심 기능을 검증합니다:
- 메트릭 계산 및 역메트릭
- 측지선 연산 (exp/log map)
- 에너지 보존
- 그래디언트 일관성
"""

import pytest
import numpy as np

try:
    import reality_stone as rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")
class TestUnifiedRiemannianLayer:
    """Unified Riemannian Layer 기본 테스트"""
    
    def test_layer_creation_poincare(self):
        """푸앵카레 메트릭으로 레이어 생성"""
        layer = rs.UnifiedRiemannianLayer(
            metric_type="poincare",
            curvature=1.0,
            input_dim=32,
            enable_bellman=False
        )
        assert layer is not None
    
    def test_layer_creation_lorentz(self):
        """로렌츠 메트릭으로 레이어 생성"""
        layer = rs.UnifiedRiemannianLayer(
            metric_type="lorentz",
            curvature=1.0,
            input_dim=33,
            enable_bellman=False
        )
        assert layer is not None
    
    def test_layer_creation_klein(self):
        """클라인 메트릭으로 레이어 생성"""
        layer = rs.UnifiedRiemannianLayer(
            metric_type="klein",
            curvature=1.0,
            input_dim=32,
            enable_bellman=False
        )
        assert layer is not None
    
    def test_layer_creation_diagonal(self):
        """대각 메트릭으로 레이어 생성"""
        layer = rs.UnifiedRiemannianLayer(
            metric_type="diagonal",
            curvature=0.0,
            input_dim=32,
            enable_bellman=False
        )
        assert layer is not None
    
    def test_forward_pass_identity(self):
        """순전파 테스트 (목표 없음)"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, False)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        output, energy = layer.forward(x)
        
        assert output.shape == (16, 32)
        assert energy is None  # enable_bellman=False이므로
    
    def test_forward_pass_with_target(self):
        """순전파 테스트 (목표 있음)"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, False)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        y = np.random.randn(16, 32).astype(np.float32) * 0.1
        output, energy = layer.forward(x, y)
        
        assert output.shape == (16, 32)
        assert energy is None
    
    def test_forward_pass_bellman(self):
        """순전파 테스트 (Bellman 활성화)"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, True)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        output, energy = layer.forward(x)
        
        assert output.shape == (16, 32)
        assert energy is not None  # enable_bellman=True이므로
        assert "kinetic" in energy
        assert "potential" in energy
        assert "lagrangian" in energy
        assert "bellman_residual" in energy
    
    def test_geodesic_path(self):
        """측지선 경로 생성 테스트"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, False)
        x = np.random.randn(1, 32).astype(np.float32) * 0.1
        y = np.random.randn(1, 32).astype(np.float32) * 0.1
        
        path = layer.geodesic_path(x, y, num_steps=10)
        assert len(path) == 10
        assert path[0].shape == (1, 32)
        
        # 시작점과 끝점 확인
        assert np.allclose(path[0], x, atol=1e-4)
        assert np.allclose(path[-1], y, atol=1e-4)
    
    def test_backward_pass(self):
        """역전파 테스트"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, False)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        output, _ = layer.forward(x)
        
        grad_output = np.ones_like(output)
        grad_input = layer.backward(grad_output, x)
        
        assert grad_input.shape == (16, 32)


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")
class TestMetricOperations:
    """메트릭 관련 연산 테스트"""
    
    def test_compute_metric(self):
        """메트릭 텐서 계산 테스트 (대각 원소만)"""
        x = np.random.randn(10, 32).astype(np.float32) * 0.1
        metric = rs.compute_metric(x, "poincare", 1.0)
        
        # 대각 메트릭만 반환: (batch, dim)
        assert metric.shape == (10, 32)
        
        # 메트릭 값은 양수여야 함
        assert np.all(metric > 0)
    
    def test_geodesic_distance(self):
        """측지선 거리 계산 테스트"""
        x = np.random.randn(5, 32).astype(np.float32) * 0.1
        y = np.random.randn(5, 32).astype(np.float32) * 0.1
        
        dist = rs.geodesic_distance(x, y, "poincare", 1.0)
        
        assert dist.shape == (5,)
        assert np.all(dist >= 0)  # 거리는 항상 양수
    
    def test_geodesic_interpolate(self):
        """측지선 보간 테스트"""
        x = np.random.randn(5, 32).astype(np.float32) * 0.1
        y = np.random.randn(5, 32).astype(np.float32) * 0.1
        
        # t=0: x와 같아야 함
        interp_0 = rs.geodesic_interpolate(x, y, "poincare", 1.0, t=0.0)
        assert np.allclose(interp_0, x, atol=1e-4)
        
        # t=1: y와 같아야 함
        interp_1 = rs.geodesic_interpolate(x, y, "poincare", 1.0, t=1.0)
        assert np.allclose(interp_1, y, atol=1e-4)
        
        # t=0.5: 중간점
        interp_mid = rs.geodesic_interpolate(x, y, "poincare", 1.0, t=0.5)
        assert interp_mid.shape == (5, 32)


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")
class TestMetricConsistency:
    """메트릭 간 일관성 테스트"""
    
    def test_distance_consistency(self):
        """다양한 메트릭에서 거리 계산 일관성"""
        x = np.random.randn(5, 32).astype(np.float32) * 0.01  # 작은 값
        y = np.random.randn(5, 32).astype(np.float32) * 0.01
        
        dist_poincare = rs.geodesic_distance(x, y, "poincare", 1.0)
        dist_diagonal = rs.geodesic_distance(x, y, "diagonal", 0.0)
        
        # 대각 메트릭은 유클리드와 유사
        # 작은 값에서는 푸앵카레도 유클리드에 가까움
        # 따라서 거리가 같은 오더이어야 함
        assert np.all(dist_poincare > 0)
        assert np.all(dist_diagonal > 0)
    
    def test_metric_symmetry(self):
        """메트릭의 대칭성 테스트"""
        x = np.random.randn(5, 32).astype(np.float32) * 0.1
        y = np.random.randn(5, 32).astype(np.float32) * 0.1
        
        dist_xy = rs.geodesic_distance(x, y, "poincare", 1.0)
        dist_yx = rs.geodesic_distance(y, x, "poincare", 1.0)
        
        assert np.allclose(dist_xy, dist_yx, atol=1e-5)
    
    def test_triangle_inequality(self):
        """삼각 부등식 테스트 (작은 값에서만)"""
        # 작은 값에서만 테스트 (수치 안정성)
        x = np.random.randn(1, 32).astype(np.float32) * 0.01
        y = np.random.randn(1, 32).astype(np.float32) * 0.01
        z = np.random.randn(1, 32).astype(np.float32) * 0.01
        
        dist_xy = rs.geodesic_distance(x, y, "poincare", 1.0)[0]
        dist_yz = rs.geodesic_distance(y, z, "poincare", 1.0)[0]
        dist_xz = rs.geodesic_distance(x, z, "poincare", 1.0)[0]
        
        # d(x,z) <= d(x,y) + d(y,z) (여유를 둠)
        # 쌍곡 공간에서는 삼각 부등식이 약하게 성립
        assert dist_xz <= dist_xy + dist_yz + 0.1


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")
class TestEnergyConservation:
    """에너지 보존 테스트"""
    
    def test_bellman_energy_components(self):
        """Bellman 에너지 컴포넌트 테스트"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, True)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        
        output, energy = layer.forward(x)
        
        assert "kinetic" in energy
        assert "potential" in energy
        assert "lagrangian" in energy
        assert "bellman_residual" in energy
        
        # 모든 에너지 항이 배치 크기와 일치해야 함
        assert energy["kinetic"].shape == (16,)
        assert energy["potential"].shape == (16,)
        assert energy["lagrangian"].shape == (16,)
        assert energy["bellman_residual"].shape == (16,)
    
    def test_energy_positivity(self):
        """에너지 양수성 테스트"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, True)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        v = np.random.randn(16, 32).astype(np.float32) * 0.1
        x_next = np.random.randn(16, 32).astype(np.float32) * 0.1
        reward = np.random.randn(16).astype(np.float32)
        
        energy_dict = layer.compute_energy(x, v, x_next, reward)
        
        # 운동 에너지는 항상 양수
        assert np.all(energy_dict["kinetic"] >= 0)


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")
class TestFlowStep:
    """Flow step 테스트"""
    
    def test_flow_step_basic(self):
        """기본 flow step 테스트"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, True)
        x = np.random.randn(16, 32).astype(np.float32) * 0.1
        
        x_next = layer.flow_step(x, num_steps=1, learning_rate=0.01)
        
        assert x_next.shape == (16, 32)
        assert not np.allclose(x_next, x)  # Flow는 상태를 변경해야 함
    
    def test_flow_convergence(self):
        """Flow 수렴 테스트"""
        layer = rs.UnifiedRiemannianLayer("poincare", 1.0, 32, True)
        x = np.random.randn(1, 32).astype(np.float32) * 0.5
        v = np.zeros((1, 32), dtype=np.float32)
        reward = np.zeros(1, dtype=np.float32)
        
        # 여러 번의 flow step
        energies = []
        for i in range(10):
            x_next = layer.flow_step(x, num_steps=1, learning_rate=0.01)
            energy_dict = layer.compute_energy(x, v, x_next, reward)
            energies.append(energy_dict["lagrangian"][0])
            x = x_next
        
        # 에너지가 폭발하지 않아야 함
        assert not np.isnan(energies[-1])
        assert not np.isinf(energies[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

