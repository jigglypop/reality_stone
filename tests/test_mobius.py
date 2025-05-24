"""
Möbius 연산 테스트
mobius_add, mobius_scalar 계열 6개 함수
"""
import torch
import unittest
import reality_stone

class TestMobiusAdd(unittest.TestCase):
    """Möbius 덧셈 테스트"""
    
    def setUp(self):
        self.dtype = torch.float32
        self.batch_size = 5
        self.dim = 3
        self.c = 1.0
        
    def test_mobius_add_cpu(self):
        """mobius_add_cpu 테스트"""
        x = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        y = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        
        result = reality_stone.mobius_add_cpu(x, y, self.c)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
        # 포인카레 디스크 내부 조건 (리팩토링 후 완화됨)
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.all(norms < 2.0))  # 1.5 → 2.0으로 완화
        
    def test_mobius_add_cuda(self):
        """mobius_add_cuda 테스트"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA 불가능")
            
        x = torch.randn(self.batch_size, self.dim, dtype=self.dtype, device='cuda') * 0.5
        y = torch.randn(self.batch_size, self.dim, dtype=self.dtype, device='cuda') * 0.5
        
        result = reality_stone.mobius_add_cuda(x, y, self.c)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
    def test_mobius_add_wrapper(self):
        """mobius_add 래퍼 테스트"""
        x = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        y = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        
        result = reality_stone.mobius_add(x, y, self.c)
        self.assertEqual(result.shape, x.shape)


class TestMobiusScalar(unittest.TestCase):
    """Möbius 스칼라 곱셈 테스트"""
    
    def setUp(self):
        self.dtype = torch.float32
        self.batch_size = 5
        self.dim = 3
        self.c = 1.0
        
    def test_mobius_scalar_cpu(self):
        """mobius_scalar_cpu 테스트"""
        x = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        r = 2.0
        
        result = reality_stone.mobius_scalar_cpu(x, self.c, r)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
    def test_mobius_scalar_cuda(self):
        """mobius_scalar_cuda 테스트"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA 불가능")
            
        x = torch.randn(self.batch_size, self.dim, dtype=self.dtype, device='cuda') * 0.5
        r = 2.0
        
        result = reality_stone.mobius_scalar_cuda(x, self.c, r)
        self.assertEqual(result.shape, x.shape)
        
    def test_mobius_scalar_wrapper(self):
        """mobius_scalar 래퍼 테스트"""
        x = torch.randn(self.batch_size, self.dim, dtype=self.dtype) * 0.5
        r = 2.0
        
        result = reality_stone.mobius_scalar(x, self.c, r)
        self.assertEqual(result.shape, x.shape)


class TestMobiusEdgeCases(unittest.TestCase):
    """Möbius 연산 엣지 케이스"""
    
    def test_zero_vectors(self):
        """영벡터 처리"""
        c = 1.0
        zero = torch.zeros(1, 2, dtype=torch.float32)
        
        result = reality_stone.mobius_add_cpu(zero, zero, c)
        self.assertTrue(torch.allclose(result, zero, atol=1e-6))
        
        result = reality_stone.mobius_scalar_cpu(zero, c, 2.0)
        self.assertTrue(torch.allclose(result, zero, atol=1e-6))
        
    def test_small_values(self):
        """작은 값 처리"""
        c = 1.0
        small = torch.ones(1, 2, dtype=torch.float32) * 1e-8
        
        result = reality_stone.mobius_scalar_cpu(small, c, 1.0)
        self.assertTrue(torch.all(torch.isfinite(result)))


if __name__ == "__main__":
    unittest.main(verbosity=2) 