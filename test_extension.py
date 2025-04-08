import torch
import riemannian_manifold

# C++ 확장 테스트
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = riemannian_manifold.add(a, b)
print("C++ 확장 결과:", c)

# Python 버전 테스트로 확인
python_result = a + b
print("Python 결과:", python_result)
print("결과 일치:", torch.allclose(c, python_result))