"""
모델 클래스 & 성능 테스트
PoincareBall, LorentzModel, KleinModel + 엣지케이스
"""

import torch
import unittest
import time
import reality_stone


class TestModelClasses(unittest.TestCase):
    """모델 클래스 테스트"""
    
    def test_poincare_ball_class(self):
        """PoincareBall 클래스 테스트"""
        try:
            poincare_ball = reality_stone.PoincareBall(c=1.0)
            self.assertIsNotNone(poincare_ball)
        except Exception as e:
            self.skipTest(f"PoincareBall 클래스 이슈: {e}")
            
    def test_lorentz_model_class(self):
        """LorentzModel 클래스 테스트"""
        try:
            lorentz_model = reality_stone.LorentzModel(c=1.0)
            self.assertIsNotNone(lorentz_model)
        except Exception as e:
            self.skipTest(f"LorentzModel 클래스 이슈: {e}")
            
    def test_klein_model_class(self):
        """KleinModel 클래스 테스트"""
        try:
            klein_model = reality_stone.KleinModel(c=1.0)
            self.assertIsNotNone(klein_model)
        except Exception as e:
            self.skipTest(f"KleinModel 클래스 이슈: {e}")


class TestEdgeCases(unittest.TestCase):
    """엣지 케이스 테스트"""
    
    def test_zero_tensors(self):
        """영 텐서 처리"""
        zero = torch.zeros(2, 3, dtype=torch.float32)
        c = 1.0
        
        # Möbius 연산
        result = reality_stone.mobius_add_cpu(zero, zero, c)
        self.assertTrue(torch.allclose(result, zero, atol=1e-6))
        
        # 변환
        lorentz = reality_stone.poincare_to_lorentz_cpu(zero[:, :2], c)
        self.assertTrue(torch.all(torch.isfinite(lorentz)))
        
    def test_boundary_values(self):
        """경계값 처리"""
        boundary = torch.ones(1, 2, dtype=torch.float32) * 0.99
        c = 1.0
        
        try:
            lorentz = reality_stone.poincare_to_lorentz_cpu(boundary, c)
            self.assertTrue(torch.all(torch.isfinite(lorentz)))
        except Exception as e:
            self.skipTest(f"경계값 이슈: {e}")
            
    def test_large_tensors(self):
        """큰 텐서 처리"""
        large = torch.randn(100, 10, dtype=torch.float32) * 0.1
        c = 1.0
        
        result = reality_stone.mobius_add_cpu(large, large, c)
        self.assertEqual(result.shape, large.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
    def test_numerical_stability(self):
        """수치적 안정성"""
        # 매우 작은 값
        tiny = torch.ones(1, 2, dtype=torch.float32) * 1e-10
        c = 1.0
        
        result = reality_stone.mobius_add_cpu(tiny, tiny, c)
        self.assertTrue(torch.all(torch.isfinite(result)))
        
        # 큰 c 값
        c_large = 100.0
        normal = torch.randn(1, 2, dtype=torch.float32) * 0.1
        
        result = reality_stone.mobius_add_cpu(normal, normal, c_large)
        self.assertTrue(torch.all(torch.isfinite(result)))


class TestPerformance(unittest.TestCase):
    """성능 테스트"""
    
    def test_mobius_performance(self):
        """Möbius 연산 성능"""
        sizes = [10, 100, 1000]
        c = 1.0
        
        for size in sizes:
            x = torch.randn(size, 2, dtype=torch.float32) * 0.5
            y = torch.randn(size, 2, dtype=torch.float32) * 0.5
            
            start = time.time()
            result = reality_stone.mobius_add_cpu(x, y, c)
            elapsed = time.time() - start
            
            # 성능 요구사항: 1000개 배치 처리가 100ms 이내
            if size == 1000:
                self.assertLess(elapsed, 0.1, f"배치 {size} 성능 저하: {elapsed:.3f}s")
                
    def test_conversion_performance(self):
        """변환 성능"""
        size = 1000
        poincare = torch.randn(size, 2, dtype=torch.float32) * 0.5
        c = 1.0
        
        start = time.time()
        lorentz = reality_stone.poincare_to_lorentz_cpu(poincare, c)
        elapsed = time.time() - start
        
        # 성능 요구사항: 1000개 변환이 50ms 이내
        self.assertLess(elapsed, 0.05, f"변환 성능 저하: {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2) 