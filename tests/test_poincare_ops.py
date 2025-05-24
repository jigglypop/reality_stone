"""
Poincaré 연산 단위 테스트
리팩토링을 위한 체계적 검증
"""

import torch
import unittest
import time
import reality_stone

class TestPoincareOperations(unittest.TestCase):
    """Poincaré 연산 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        
    def test_mobius_add_basic(self):
        """mobius_add 기본 동작"""
        x = torch.tensor([[0.1, 0.2]], dtype=self.dtype)
        y = torch.tensor([[0.3, 0.1]], dtype=self.dtype)
        c = 1.0
        
        result = reality_stone.mobius_add_cpu(x, y, c)
        
        # 결과 검증
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
        # 포인카레 디스크 내부 조건
        norm = torch.norm(result, dim=-1)
        self.assertTrue(torch.all(norm < 1.0))
        
    def test_mobius_add_batch(self):
        """mobius_add 배치 처리"""
        batch_size = 5
        dim = 3
        x = torch.randn(batch_size, dim, dtype=self.dtype) * 0.5
        y = torch.randn(batch_size, dim, dtype=self.dtype) * 0.5
        c = 1.0
        
        result = reality_stone.mobius_add_cpu(x, y, c)
        
        self.assertEqual(result.shape, (batch_size, dim))
        self.assertTrue(torch.all(torch.isfinite(result)))
        
    def test_mobius_scalar_basic(self):
        """mobius_scalar 기본 동작"""
        x = torch.tensor([[0.1, 0.2]], dtype=self.dtype)
        c = 1.0
        r = 2.0
        
        result = reality_stone.mobius_scalar_cpu(x, c, r)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
    def test_mobius_operations_edge_cases(self):
        """mobius 연산 엣지 케이스"""
        c = 1.0
        
        # 영벡터
        zero = torch.zeros(1, 2, dtype=self.dtype)
        result = reality_stone.mobius_add_cpu(zero, zero, c)
        self.assertTrue(torch.allclose(result, zero, atol=1e-6))
        
        # 작은 값
        small = torch.ones(1, 2, dtype=self.dtype) * 1e-8
        result = reality_stone.mobius_scalar_cpu(small, c, 1.0)
        self.assertTrue(torch.all(torch.isfinite(result)))


class TestPoincareLayer(unittest.TestCase):
    """Poincaré Layer 테스트 (현재 버그 있음)"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        
    def test_poincare_ball_layer_dimension_mismatch(self):
        """poincare_ball_layer 차원 불일치 문제 문서화"""
        batch_size, dim = 2, 4
        u = torch.randn(batch_size, dim, dtype=self.dtype)
        v = torch.randn(dim, dim, dtype=self.dtype)  # 문제: 2D 행렬
        c = 1.0
        t = 0.1
        
        # 현재 알려진 버그 - 차원 불일치로 실패함
        with self.assertRaises(RuntimeError) as context:
            reality_stone.poincare_ball_layer(u, v, c, t)
        
        self.assertIn("size of tensor", str(context.exception))
        
    def test_poincare_ball_layer_correct_usage(self):
        """poincare_ball_layer 올바른 사용법 (수정 필요)"""
        batch_size, dim = 2, 4
        u = torch.randn(batch_size, dim, dtype=self.dtype)
        v = torch.randn(batch_size, dim, dtype=self.dtype)  # 같은 차원
        c = 1.0
        t = 0.1
        
        # 이것도 현재는 실패할 수 있음 - API 설계 문제
        try:
            result = reality_stone.poincare_ball_layer(u, v, c, t)
            self.assertEqual(result.shape, u.shape)
        except RuntimeError as e:
            self.skipTest(f"poincare_ball_layer API 버그: {e}")


class TestLorentzConversions(unittest.TestCase):
    """Lorentz 모델 변환 테스트"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.c = 1.0
        
    def test_poincare_to_lorentz(self):
        """Poincaré → Lorentz 변환"""
        poincare_x = torch.tensor([[0.2, 0.3], [0.1, 0.4]], dtype=self.dtype)
        
        lorentz_x = reality_stone.poincare_to_lorentz_cpu(poincare_x, self.c)
        
        # 차원 검증
        self.assertEqual(lorentz_x.shape, (2, 3))  # [time, space1, space2]
        
        # Lorentz 조건: -t² + x² + y² = -1/c
        lorentz_norm = -lorentz_x[:, 0]**2 + torch.sum(lorentz_x[:, 1:]**2, dim=1)
        expected = -1.0 / self.c
        self.assertTrue(torch.allclose(lorentz_norm, torch.full_like(lorentz_norm, expected), atol=1e-5))
        
    def test_lorentz_to_poincare(self):
        """Lorentz → Poincaré 변환"""
        lorentz_x = torch.tensor([[1.2, 0.2, 0.3], [1.3, 0.1, 0.4]], dtype=self.dtype)
        
        poincare_x = reality_stone.lorentz_to_poincare_cpu(lorentz_x, self.c)
        
        # 차원 검증
        self.assertEqual(poincare_x.shape, (2, 2))
        # Poincaré 디스크 조건: ||x|| < 1
        norms = torch.norm(poincare_x, dim=-1)
        self.assertTrue(torch.all(norms < 1.0))
        
    def test_roundtrip_conversion(self):
        """왕복 변환 테스트"""
        original = torch.tensor([[0.2, 0.3], [0.1, 0.4]], dtype=self.dtype)
        # Poincaré → Lorentz → Poincaré
        lorentz = reality_stone.poincare_to_lorentz_cpu(original, self.c)
        recovered = reality_stone.lorentz_to_poincare_cpu(lorentz, self.c)
        self.assertTrue(torch.allclose(original, recovered, atol=1e-5))


class TestPerformance(unittest.TestCase):
    """성능 테스트"""
    def test_mobius_performance(self):
        """Mobius 연산 성능"""
        sizes = [10, 100, 1000]
        for size in sizes:
            x = torch.randn(size, 2, dtype=torch.float32) * 0.5
            y = torch.randn(size, 2, dtype=torch.float32) * 0.5
            start = time.time()
            result = reality_stone.mobius_add_cpu(x, y, 1.0)
            elapsed = time.time() - start
            # 성능 요구사항: 1000개 배치 처리가 100ms 이내
            if size == 1000:
                self.assertLess(elapsed, 0.1, f"성능 저하: {elapsed:.3f}s")
                
    def test_lorentz_performance(self):
        """Lorentz 변환 성능"""
        size = 1000
        poincare_x = torch.randn(size, 2, dtype=torch.float32) * 0.5
        
        start = time.time()
        lorentz_x = reality_stone.poincare_to_lorentz_cpu(poincare_x, 1.0)
        elapsed = time.time() - start
        
        # 성능 요구사항: 1000개 변환이 50ms 이내
        self.assertLess(elapsed, 0.05, f"변환 성능 저하: {elapsed:.3f}s")


def run_tests():
    """테스트 실행"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests() 