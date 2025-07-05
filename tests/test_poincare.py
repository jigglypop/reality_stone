import torch
import unittest
import reality_stone

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

if __name__ == "__main__":
    unittest.main(verbosity=2) 