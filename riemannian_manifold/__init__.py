from torch.autograd import Function
import torch
from . import _C

# 확장 모듈에서 함수를 가져옵니다
try:
    from ._C import add_tensors, poincare_exp_map
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    print("C++ 확장을 로드할 수 없습니다. 순수 Python 구현을 사용합니다.")

# 순수 Python 폴백 구현
def py_add_tensors(a, b):
    return a + b

# 실제 사용 함수
def add_tensors_impl(a, b):
    if HAS_CPP_EXTENSION:
        return _C.add_tensors(a, b)
    else:
        return py_add_tensors(a, b)

# 편의 함수
def add(a, b):
    return add_tensors_impl(a, b)