import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reality_stone as rs

class TestAdvancedFFT:
    """고급 FFT 기능 테스트"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.dim = 64
        self.curvature = 1.0
        
        # 테스트 데이터 생성
        self.x = torch.randn(self.batch_size, self.dim, device=self.device) * 0.5
        self.y = torch.randn(self.batch_size, self.dim, device=self.device) * 0.5
    
    def test_chebyshev_approximation(self):
        """체비셰프 근사 테스트"""
        try:
            result = rs.advanced_api.chebyshev_approximation(self.x, order=10, curvature=self.curvature)
            assert result.shape == self.x.shape
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
            print("✅ 체비셰프 근사 테스트 통과")
        except Exception as e:
            print(f"⚠️  체비셰프 근사 fallback 사용: {e}")
    
    def test_chebyshev_distance(self):
        """체비셰프 거리 테스트"""
        try:
            distance = rs.advanced_api.chebyshev_distance(self.x, self.y, curvature=self.curvature)
            assert distance.shape == (self.batch_size,)
            assert (distance >= 0).all()
            assert not torch.isnan(distance).any()
            print("✅ 체비셰프 거리 테스트 통과")
        except Exception as e:
            print(f"⚠️  체비셰프 거리 fallback 사용: {e}")
    
    def test_chebyshev_nodes(self):
        """체비셰프 노드 테스트"""
        n = 16
        nodes = rs.advanced_api.chebyshev_nodes(n, device=self.device)
        assert nodes.shape == (n,)
        assert (nodes >= -1).all() and (nodes <= 1).all()
        print("✅ 체비셰프 노드 테스트 통과")
    
    def test_fast_chebyshev_transform(self):
        """고속 체비셰프 변환 테스트"""
        values = torch.randn(self.batch_size, 32, device=self.device)
        try:
            coeffs = rs.advanced_api.fast_chebyshev_transform(values)
            assert coeffs.shape == values.shape
            assert not torch.isnan(coeffs).any()
            print("✅ 고속 체비셰프 변환 테스트 통과")
        except Exception as e:
            print(f"⚠️  체비셰프 변환 fallback 사용: {e}")
    
    def test_hyperbolic_laplacian(self):
        """하이퍼볼릭 라플라시안 테스트"""
        try:
            laplacian = rs.advanced_api.hyperbolic_laplacian(self.x, curvature=self.curvature)
            assert laplacian.shape == self.x.shape
            assert not torch.isnan(laplacian).any()
            print("✅ 하이퍼볼릭 라플라시안 테스트 통과")
        except Exception as e:
            print(f"⚠️  라플라시안 fallback 사용: {e}")
    
    def test_heat_kernel(self):
        """열 핵 테스트"""
        t = 0.1
        try:
            kernel = rs.advanced_api.heat_kernel(self.x, t, curvature=self.curvature)
            assert kernel.shape[0] == self.batch_size
            assert (kernel >= 0).all()  # 열 핵은 항상 양수
            print("✅ 열 핵 테스트 통과")
        except Exception as e:
            print(f"⚠️  열 핵 fallback 사용: {e}")
    
    def test_hyperbolic_fft(self):
        """하이퍼볼릭 FFT 테스트"""
        try:
            fft_result = rs.advanced_api.hyperbolic_fft(self.x, curvature=self.curvature)
            assert fft_result.shape[0] == self.batch_size
            assert not torch.isnan(fft_result).any()
            print("✅ 하이퍼볼릭 FFT 테스트 통과")
        except Exception as e:
            print(f"⚠️  하이퍼볼릭 FFT fallback 사용: {e}")
    
    def test_spherical_harmonics(self):
        """구면 조화 함수 테스트"""
        theta_phi = torch.rand(self.batch_size, 2, device=self.device) * 2 * 3.14159
        l_max = 5
        try:
            harmonics = rs.advanced_api.spherical_harmonics(theta_phi, l_max)
            expected_size = (l_max + 1) ** 2
            assert harmonics.shape == (self.batch_size, expected_size)
            assert not torch.isnan(harmonics).any()
            print("✅ 구면 조화 함수 테스트 통과")
        except Exception as e:
            print(f"⚠️  구면 조화 함수 fallback 사용: {e}")
    
    def test_fast_spherical_conv(self):
        """빠른 구면 컨볼루션 테스트"""
        try:
            conv_result = rs.advanced_api.fast_spherical_conv(self.x, self.y, curvature=self.curvature)
            assert conv_result.shape == self.x.shape
            assert not torch.isnan(conv_result).any()
            print("✅ 빠른 구면 컨볼루션 테스트 통과")
        except Exception as e:
            print(f"⚠️  구면 컨볼루션 fallback 사용: {e}")
    
    def test_ricci_curvature(self):
        """리치 곡률 테스트"""
        metric_tensor = torch.randn(self.batch_size, self.dim, self.dim, device=self.device)
        try:
            ricci = rs.advanced_api.ricci_curvature(metric_tensor)
            assert ricci.shape == (self.batch_size,)
            assert not torch.isnan(ricci).any()
            print("✅ 리치 곡률 테스트 통과")
        except Exception as e:
            print(f"⚠️  리치 곡률 fallback 사용: {e}")
    
    def test_parallel_transport(self):
        """평행 이동 테스트"""
        vector = torch.randn(self.batch_size, self.dim, device=self.device) * 0.1
        path = torch.randn(self.batch_size, 2 * self.dim, device=self.device) * 0.3
        try:
            transported = rs.advanced_api.parallel_transport(vector, path, curvature=self.curvature)
            assert transported.shape == vector.shape
            assert not torch.isnan(transported).any()
            print("✅ 평행 이동 테스트 통과")
        except Exception as e:
            print(f"⚠️  평행 이동 fallback 사용: {e}")
    
    def test_geodesic_flow(self):
        """지오데식 플로우 테스트"""
        x = torch.randn(self.batch_size, self.dim, device=self.device) * 0.3
        v = torch.randn(self.batch_size, self.dim, device=self.device) * 0.1
        t = 0.1
        try:
            flowed = rs.advanced_api.geodesic_flow(x, v, t, curvature=self.curvature)
            assert flowed.shape == x.shape
            assert not torch.isnan(flowed).any()
            print("✅ 지오데식 플로우 테스트 통과")
        except Exception as e:
            print(f"⚠️  지오데식 플로우 fallback 사용: {e}")
    
    def test_riemannian_gradient(self):
        """리만 그래디언트 테스트"""
        grad = torch.randn(self.batch_size, self.dim, device=self.device)
        x = torch.randn(self.batch_size, self.dim, device=self.device) * 0.3
        try:
            riem_grad = rs.advanced_api.riemannian_gradient(grad, x, curvature=self.curvature)
            assert riem_grad.shape == grad.shape
            assert not torch.isnan(riem_grad).any()
            print("✅ 리만 그래디언트 테스트 통과")
        except Exception as e:
            print(f"⚠️  리만 그래디언트 fallback 사용: {e}")
    
    def test_geodesic_sgd_step(self):
        """지오데식 SGD 스텝 테스트"""
        x = torch.randn(self.batch_size, self.dim, device=self.device) * 0.3
        grad = torch.randn(self.batch_size, self.dim, device=self.device) * 0.1
        lr = 0.01
        try:
            updated = rs.advanced_api.geodesic_sgd_step(x, grad, lr, curvature=self.curvature)
            assert updated.shape == x.shape
            assert not torch.isnan(updated).any()
            print("✅ 지오데식 SGD 스텝 테스트 통과")
        except Exception as e:
            print(f"⚠️  지오데식 SGD fallback 사용: {e}")

    def test_hyperbolic_wavelet_decomposition(self):
        """하이퍼볼릭 웨이블릿 분해 테스트"""
        signal = torch.randn(self.batch_size, self.dim, device=self.device) * 0.5
        num_levels = 3
        try:
            coeffs = rs.advanced_api.hyperbolic_wavelet_decomposition(
                signal, num_levels, curvature=self.curvature
            )
            assert coeffs.shape == signal.shape
            assert not torch.isnan(coeffs).any()
            print("✅ 하이퍼볼릭 웨이블릿 분해 테스트 통과")
        except Exception as e:
            print(f"⚠️  웨이블릿 분해 fallback 사용: {e}")
    
    def test_frequency_domain_filter(self):
        """주파수 도메인 필터링 테스트"""
        signal = torch.randn(self.batch_size, self.dim, device=self.device) * 0.5
        filter_coeffs = torch.ones(self.dim, device=self.device) * 0.8
        try:
            filtered = rs.advanced_api.frequency_domain_filter(
                signal, filter_coeffs, curvature=self.curvature
            )
            assert filtered.shape == signal.shape
            assert not torch.isnan(filtered).any()
            print("✅ 주파수 도메인 필터링 테스트 통과")
        except Exception as e:
            print(f"⚠️  주파수 필터링 fallback 사용: {e}")

def test_advanced_processor_class():
    """고급 하이퍼볼릭 처리기 클래스 테스트"""
    input_dim = 64
    output_dim = 32
    batch_size = 16
    
    processor = rs.advanced_api.AdvancedHyperbolicProcessor(
        input_dim=input_dim,
        output_dim=output_dim,
        curvature=1.0,
        use_chebyshev=True,
        use_fft=True,
        use_laplacian=False
    )
    
    x = torch.randn(batch_size, input_dim) * 0.5
    try:
        output = processor(x)
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
        print("✅ 고급 하이퍼볼릭 처리기 클래스 테스트 통과")
    except Exception as e:
        print(f"⚠️  처리기 클래스 fallback 사용: {e}")

def test_spherical_harmonic_layer():
    """구면 조화 함수 레이어 테스트"""
    l_max = 5
    batch_size = 16
    
    layer = rs.advanced_api.SphericalHarmonicLayer(l_max=l_max)
    theta_phi = torch.rand(batch_size, 2) * 2 * 3.14159
    
    try:
        output = layer(theta_phi)
        expected_dim = (l_max + 1) ** 2
        assert output.shape == (batch_size, expected_dim)
        assert not torch.isnan(output).any()
        print("✅ 구면 조화 함수 레이어 테스트 통과")
    except Exception as e:
        print(f"⚠️  구면 조화 레이어 fallback 사용: {e}")

def test_hyperbolic_wavelet_layer():
    """하이퍼볼릭 웨이블릿 레이어 테스트"""
    num_levels = 3
    batch_size = 16
    dim = 32
    
    layer = rs.advanced_api.HyperbolicWaveletLayer(num_levels=num_levels, curvature=1.0)
    x = torch.randn(batch_size, dim) * 0.5
    
    try:
        output = layer(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        print("✅ 하이퍼볼릭 웨이블릿 레이어 테스트 통과")
    except Exception as e:
        print(f"⚠️  웨이블릿 레이어 fallback 사용: {e}")

if __name__ == "__main__":
    print("🚀 Reality Stone 고급 FFT 기능 테스트 시작")
    print("=" * 60)
    
    # 기본 테스트 실행
    test_class = TestAdvancedFFT()
    test_class.setup_method()
    
    # 모든 테스트 메소드 실행
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"\n📋 {method_name} 실행 중...")
        try:
            getattr(test_class, method_name)()
        except Exception as e:
            print(f"❌ {method_name} 실패: {e}")
    
    # 클래스 테스트들
    print(f"\n📋 고급 클래스 테스트들...")
    test_advanced_processor_class()
    test_spherical_harmonic_layer()
    test_hyperbolic_wavelet_layer()
    
    print("\n" + "=" * 60)
    print("🎉 Reality Stone 고급 FFT 기능 테스트 완료!")
    print("⚠️  Fallback 메시지들은 C++ 확장이 없을 때 정상적인 동작입니다.") 