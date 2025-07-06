import torch
import unittest
import reality_stone
from reality_stone import AdvancedConfig, create_mnist_model

class TestPoincareDistance(unittest.TestCase):
    """Poincaré Distance 연산 테스트"""

    def setUp(self):
        self.dtype = torch.float64 # Use float64 for better precision in distance checks
        self.c = 1.0
        self.batch_size = 8
        self.dim = 16

    @unittest.skipUnless(getattr(reality_stone, '_has_rust_ext', False), "Rust extension not available")
    def test_distance_cpu(self):
        """poincare_distance (CPU) 테스트"""
        u = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        v = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5

        dist = reality_stone.poincare_distance(u, v, self.c)
        self.assertEqual(dist.shape, (self.batch_size,))
        self.assertTrue(torch.all(dist >= 0))

    @unittest.skipUnless(getattr(reality_stone, '_has_cuda', False), "CUDA not available")
    def test_distance_cuda(self):
        """poincare_distance (CUDA) 테스트"""
        u = torch.randn(self.batch_size, self.dim, dtype=self.dtype, device='cuda') * 0.5
        v = torch.randn(self.batch_size, self.dim, dtype=self.dtype, device='cuda') * 0.5

        dist = reality_stone.poincare_distance(u, v, self.c)
        self.assertEqual(dist.shape, (self.batch_size,))
        self.assertTrue(torch.all(dist.cpu() >= 0))
        self.assertEqual(dist.device.type, 'cuda')

    @unittest.skipUnless(getattr(reality_stone, '_has_cuda', False), "CUDA not available")
    def test_distance_cpu_cuda_equivalence(self):
        """poincare_distance의 CPU와 CUDA 결과 비교"""
        # Use float32 for direct comparison with CUDA kernel
        dtype = torch.float32
        u_cpu = torch.randn(self.batch_size, self.dim, dtype=dtype) * 0.5
        v_cpu = torch.randn(self.batch_size, self.dim, dtype=dtype) * 0.5

        # CPU
        dist_cpu = reality_stone.poincare_distance(u_cpu, v_cpu, self.c)

        # CUDA
        u_cuda = u_cpu.to('cuda')
        v_cuda = v_cpu.to('cuda')
        dist_cuda = reality_stone.poincare_distance(u_cuda, v_cuda, self.c)

        self.assertTrue(torch.allclose(dist_cpu, dist_cuda.cpu(), atol=1e-5))

    @unittest.skipUnless(getattr(reality_stone, '_has_rust_ext', False), "Rust extension not available")
    def test_distance_identity(self):
        """Distance between a point and itself should be zero"""
        u = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        dist = reality_stone.poincare_distance(u, u, self.c)
        self.assertTrue(torch.allclose(dist, torch.zeros(self.batch_size, dtype=self.dtype), atol=1e-6))

    def test_mobius_operations(self):
        """Möbius 연산 테스트"""
        u = torch.randn(self.batch_size, self.dim, dtype=torch.float32) * 0.3
        v = torch.randn(self.batch_size, self.dim, dtype=torch.float32) * 0.3
        
        # Möbius addition
        result = reality_stone.mobius_add(u, v, self.c)
        self.assertEqual(result.shape, (self.batch_size, self.dim))
        # 결과가 Poincaré ball 내부에 있어야 함
        norms = torch.norm(result, dim=1)
        self.assertTrue(torch.all(norms < 1.0 / self.c**0.5))
        
        # Möbius scalar multiplication  
        r = 0.5
        result = reality_stone.mobius_scalar(u, r, self.c)
        self.assertEqual(result.shape, (self.batch_size, self.dim))
        norms = torch.norm(result, dim=1)
        self.assertTrue(torch.all(norms < 1.0 / self.c**0.5))

    def test_poincare_ball_layer(self):
        """Poincaré ball layer 테스트"""
        u = torch.randn(self.batch_size, self.dim, dtype=torch.float32) * 0.3
        v = torch.randn(self.batch_size, self.dim, dtype=torch.float32) * 0.3
        t = 0.5
        
        result = reality_stone.poincare_ball_layer(u, v, self.c, t)
        self.assertEqual(result.shape, (self.batch_size, self.dim))
        norms = torch.norm(result, dim=1)
        self.assertTrue(torch.all(norms < 1.0 / self.c**0.5))


class TestDynamicCurvature(unittest.TestCase):
    """동적 곡률 기능 테스트"""
    
    def test_dynamic_curvature_default(self):
        """동적 곡률이 기본으로 활성화되어 있는지 확인"""
        config = AdvancedConfig()
        self.assertTrue(config.enable_dynamic_curvature)
        self.assertTrue(config.enable_fused_ops)
    
    def test_dynamic_curvature_model(self):
        """동적 곡률을 사용하는 모델 생성 테스트"""
        # 기본 모델 (동적 곡률 활성화)
        model = create_mnist_model()
        
        # 입력 데이터
        x = torch.randn(32, 784)
        
        # Forward pass
        output = model(x)
        self.assertEqual(output.shape, (32, 10))
        self.assertFalse(torch.isnan(output).any())
        
    def test_dynamic_curvature_disabled(self):
        """동적 곡률을 비활성화한 모델 테스트"""
        config = AdvancedConfig(enable_dynamic_curvature=False)
        model = create_mnist_model(config)
        
        x = torch.randn(32, 784)
        output = model(x)
        self.assertEqual(output.shape, (32, 10))
        self.assertFalse(torch.isnan(output).any())
        
    def test_gradient_flow(self):
        """그래디언트가 제대로 흐르는지 확인"""
        model = create_mnist_model()
        x = torch.randn(32, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


class TestNumericalStability(unittest.TestCase):
    """수치 안정성 테스트"""
    
    def test_boundary_cases(self):
        """경계 케이스 테스트"""
        c = 1.0
        
        # 매우 작은 벡터
        u = torch.randn(4, 8) * 1e-8
        v = torch.randn(4, 8) * 1e-8
        result = reality_stone.mobius_add(u, v, c)
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())
        
        # 경계 근처 벡터
        u = torch.randn(4, 8)
        u = u / torch.norm(u, dim=1, keepdim=True) * 0.99 / c**0.5
        v = torch.randn(4, 8) * 0.01
        result = reality_stone.mobius_add(u, v, c)
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())
        

if __name__ == "__main__":
    unittest.main(verbosity=2) 