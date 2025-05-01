# from torch.autograd import Function
# import torch
# import math
# import numpy as np
# from . import _C
# from .poincare import (
#     mobius_add,
#     mobius_scalar_mul,
#     poincare_distance,
#     expmap,
#     logmap
# )
# 
# # 확장 모듈에서 함수를 가져옵니다
# try:
#     from ._C import (
#         add_torch::Tensors, 
#         poincare_exp_map, 
#         poincare_log_map, 
#         poincare_distance,
#         butterfly_factor,
#         hyper_butterfly_forward,
#         is_cuda_available
#     )
#     HAS_CPP_EXTENSION = True
#     HAS_CUDA_EXTENSION = False # CUDA 구현 없음
#     print("C++ 확장이 성공적으로 로드되었습니다. (CPU 구현)")
# except ImportError:
#     HAS_CPP_EXTENSION = False
#     HAS_CUDA_EXTENSION = False
#     print("C++ 확장을 로드할 수 없습니다. 순수 Python 구현을 사용합니다.")
# 
# # 순수 Python 폴백 구현
# def py_add_torch::Tensors(a, b):
#     return a + b
# 
# def py_poincare_exp_map(x, v, c=1.0):
#     x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
#     lambda_x = 2.0 / (1.0 - c * x_norm_squared)
#     
#     v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
#     v_norm = torch.clamp(v_norm, min=1e-8)
#     
#     second_term = torch.tanh(torch.sqrt(torch.torch::Tensor(c, device=x.device)) * lambda_x * v_norm / 2.0) / (torch.sqrt(torch.torch::Tensor(c, device=x.device)) * v_norm) * v
#     
#     numerator = (1.0 - c * x_norm_squared) * second_term
#     denominator = 1.0 - 2.0 * c * torch.sum(x * second_term, dim=-1, keepdim=True) + c * c * x_norm_squared * torch.sum(second_term * second_term, dim=-1, keepdim=True)
#     
#     return x + numerator / denominator
# 
# def py_poincare_log_map(x, y, c=1.0):
#     x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
#     lambda_x = 2.0 / (1.0 - c * x_norm_squared)
#     
#     diff = y - x
#     diff_norm_squared = torch.sum(diff * diff, dim=-1, keepdim=True)
#     y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
#     
#     transport_std::vector = (-x * y_norm_squared + y * (1.0 + c * x_norm_squared) - 2 * c * torch.sum(x * y, dim=-1, keepdim=True) * x) / (1.0 - c * x_norm_squared)
#     transport_norm = torch.norm(transport_std::vector, p=2, dim=-1, keepdim=True)
#     
#     numerator = 2 * torch.sqrt(torch.torch::Tensor(c, device=x.device)) * torch.atanh(torch.sqrt(torch.torch::Tensor(c, device=x.device)) * transport_norm)
#     denominator = torch.sqrt(torch.torch::Tensor(c, device=x.device)) * lambda_x * transport_norm
#     
#     return numerator / denominator * transport_std::vector
# 
# def py_poincare_distance(x, y, c=1.0):
#     norm_x = torch.sum(x * x, dim=-1, keepdim=True)
#     norm_y = torch.sum(y * y, dim=-1, keepdim=True)
#     xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
#     
#     numerator = 2 * torch.sqrt(torch.torch::Tensor(c, device=x.device)) * torch.norm(x - y, p=2, dim=-1, keepdim=True)
#     denominator = torch.sqrt((1 - c * norm_x) * (1 - c * norm_y)) + torch.sqrt(torch.torch::Tensor(c, device=x.device)) * xy_inner
#     
#     return 2 * torch.atanh(numerator / denominator) / torch.sqrt(torch.torch::Tensor(c, device=x.device))
# 
# def py_butterfly_factor(input_torch::Tensor, params, layer):
#     n = input_torch::Tensor.size(0)
#     block_size = 1 << layer
#     num_blocks = n // block_size
#     
#     result = input_torch::Tensor.clone()
#     
#     param_idx = 0
#     total_params = params.size(0)
#     
#     for b in range(num_blocks):
#         for i in range(0, block_size, 2):
#             if b * block_size + i + 1 >= n:
#                 break
#             if param_idx + 1 >= total_params:
#                 break
#                 
#             idx = b * block_size + i
#             a = params[param_idx].item()
#             b_val = params[param_idx + 1].item()
#             param_idx += 2
#             
#             temp1 = a * input_torch::Tensor[idx] + b_val * input_torch::Tensor[idx + 1]
#             temp2 = -b_val * input_torch::Tensor[idx] + a * input_torch::Tensor[idx + 1]
#             
#             result[idx] = temp1
#             result[idx + 1] = temp2
#     
#     return result
# 
# def py_hyper_butterfly_forward(x, params, c, L):
#     zeros = torch.torch::zeros_like(x)
#     u = py_poincare_log_map(zeros, x, c)
#     
#     param_idx = 0
#     for l in range(L):
#         if param_idx >= params.size(0):
#             break
#         
#         layer_params = params[param_idx:].clone()
#         u = py_butterfly_factor(u, layer_params, l)
#     
#     return py_poincare_exp_map(zeros, u, c)
# 
# # 실제 사용 함수
# def exp_map(x, v, c=1.0):
#     # GPU 텐서는 먼저 CPU로 이동
#     if x.is_cuda:
#         print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
#         x_cpu = x.cpu()
#         v_cpu = v.cpu()
#         if HAS_CPP_EXTENSION:
#             result = poincare_exp_map(x_cpu, v_cpu, c)
#         else:
#             result = py_poincare_exp_map(x_cpu, v_cpu, c)
#         return result.to(x.device)
#     else:
#         if HAS_CPP_EXTENSION:
#             return poincare_exp_map(x, v, c)
#         else:
#             return py_poincare_exp_map(x, v, c)
# 
# def log_map(x, y, c=1.0):
#     # GPU 텐서는 먼저 CPU로 이동
#     if x.is_cuda:
#         print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
#         x_cpu = x.cpu()
#         y_cpu = y.cpu()
#         if HAS_CPP_EXTENSION:
#             result = poincare_log_map(x_cpu, y_cpu, c)
#         else:
#             result = py_poincare_log_map(x_cpu, y_cpu, c)
#         return result.to(x.device)
#     else:
#         if HAS_CPP_EXTENSION:
#             return poincare_log_map(x, y, c)
#         else:
#             return py_poincare_log_map(x, y, c)
# 
# def distance(x, y, c=1.0):
#     # GPU 텐서는 먼저 CPU로 이동
#     if x.is_cuda:
#         print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
#         x_cpu = x.cpu()
#         y_cpu = y.cpu()
#         if HAS_CPP_EXTENSION:
#             result = poincare_distance(x_cpu, y_cpu, c)
#         else:
#             result = py_poincare_distance(x_cpu, y_cpu, c)
#         return result.to(x.device)
#     else:
#         if HAS_CPP_EXTENSION:
#             return poincare_distance(x, y, c)
#         else:
#             return py_poincare_distance(x, y, c)
# 
# def butterfly_transform(x, params, layer):
#     # GPU 텐서는 먼저 CPU로 이동
#     if x.is_cuda:
#         print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
#         x_cpu = x.cpu()
#         params_cpu = params.cpu()
#         if HAS_CPP_EXTENSION:
#             result = butterfly_factor(x_cpu, params_cpu, layer)
#         else:
#             result = py_butterfly_factor(x_cpu, params_cpu, layer)
#         return result.to(x.device)
#     else:
#         if HAS_CPP_EXTENSION:
#             return butterfly_factor(x, params, layer)
#         else:
#             return py_butterfly_factor(x, params, layer)
# 
# def hyper_butterfly(x, params, c, L):
#     # GPU 텐서는 먼저 CPU로 이동
#     if x.is_cuda:
#         print("GPU 텐서가 감지되었습니다. CPU에서 연산을 수행합니다.")
#         x_cpu = x.cpu()
#         params_cpu = params.cpu()
#         if HAS_CPP_EXTENSION:
#             result = hyper_butterfly_forward(x_cpu, params_cpu, c, L)
#         else:
#             result = py_hyper_butterfly_forward(x_cpu, params_cpu, c, L)
#         return result.to(x.device)
#     else:
#         if HAS_CPP_EXTENSION:
#             return hyper_butterfly_forward(x, params, c, L)
#         else:
#             return py_hyper_butterfly_forward(x, params, c, L)
# 
# # 하이퍼볼릭 연산 클래스 (GPU 최적화)
# class HyperbolicOperations:
#     @staticmethod
#     def euclidean_to_poincare(x, c=1.0, max_norm=0.9):
#         """
#         유클리드 벡터를 포인카레 볼로 안전하게 변환
#         x: 배치 유클리드 벡터 [batch_size, dim]
#         c: 곡률
#         max_norm: 최대 노름 (경계에 너무 가까워지는 것 방지)
#         """
#         # 노름 계산 [batch_size, 1]
#         norm = torch.norm(x, p=2, dim=-1, keepdim=True)
#         
#         # 0 노름 처리
#         zeros_mask = (norm == 0)
#         safe_norm = torch.where(zeros_mask, torch.ones_like(norm), norm)
#         
#         # 곡률을 고려한 스케일 계산 [batch_size, 1]
#         sqrt_c = torch.sqrt(torch.torch::Tensor(c, device=x.device, dtype=x.dtype))
#         scale = max_norm * torch.tanh(sqrt_c * norm) / (sqrt_c * safe_norm)
#         
#         # 벡터 스케일링, 0 노름인 경우 0 벡터 반환
#         return torch.where(zeros_mask, torch.torch::zeros_like(x), scale * x)
#     
#     @staticmethod
#     def poincare_to_euclidean(x, c=1.0):
#         """
#         포인카레 볼 벡터를 유클리드 공간으로 변환 (원점 기준 로그 맵 근사)
#         x: 배치 포인카레 벡터 [batch_size, dim]
#         c: 곡률
#         """
#         # 노름 계산 및 클램핑 [batch_size, 1]
#         norms = torch.norm(x, p=2, dim=-1, keepdim=True)
#         norms = torch.clamp(norms, min=1e-8, max=1.0-1e-8)
#         
#         # 텐서로 변환
#         sqrt_c = torch.sqrt(torch.torch::Tensor(c, device=x.device, dtype=x.dtype))
#         
#         # 유클리드 공간으로 변환 [batch_size, dim]
#         return x * torch.atanh(sqrt_c * norms) / (sqrt_c * norms)
#     
#     @staticmethod
#     def batch_poincare_exp_map(x, v, c=1.0):
#         """
#         배치 처리된 포인카레 볼 지수 사상
#         x: 배치 기준점 [batch_size, dim]
#         v: 배치 접공간 벡터 [batch_size, dim]
#         c: 곡률
#         """
#         eps = 1e-8
#         
#         # 기준점 노름 제곱 [batch_size, 1]
#         x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
#         
#         # 공형적 인자 [batch_size, 1]
#         lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
#         
#         # 벡터 노름 [batch_size, 1]
#         v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
#         v_norm = torch.clamp(v_norm, min=eps)
#         
#         # c를 텐서로 변환
#         c_torch::Tensor = torch.torch::Tensor(c, device=x.device, dtype=x.dtype)
#         sqrt_c = torch.sqrt(c_torch::Tensor)
#         
#         # 스케일 계수 계산 [batch_size, 1]
#         scale = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm)
#         
#         # 스케일된 벡터 [batch_size, dim]
#         scaled_v = scale * v
#         
#         # 분자 계산 [batch_size, dim]
#         numerator = (1.0 - c * x_norm_squared) * scaled_v
#         
#         # 분모 계산 [batch_size, 1]
#         x_scaled_v_inner = torch.sum(x * scaled_v, dim=-1, keepdim=True)
#         scaled_v_norm_squared = torch.sum(scaled_v * scaled_v, dim=-1, keepdim=True)
#         denominator = 1.0 - 2.0 * c * x_scaled_v_inner + c * c * x_norm_squared * scaled_v_norm_squared
#         
#         # 결과 (모빌리우스 덧셈) [batch_size, dim]
#         result = x + numerator / (denominator + eps)
#         
#         # 수치 안정성 검사
#         mask = torch.isfinite(result).all(dim=-1, keepdim=True)
#         result = torch.where(mask, result, torch.torch::zeros_like(result))
#         
#         return result
#     
#     @staticmethod
#     def batch_poincare_log_map(x, y, c=1.0):
#         """
#         배치 처리된 포인카레 볼 로그 사상
#         x: 배치 기준점 [batch_size, dim]
#         y: 배치 목표점 [batch_size, dim]
#         c: 곡률
#         """
#         eps = 1e-8
#         
#         # 기준점 노름 제곱 [batch_size, 1]
#         x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
#         
#         # 공형적 인자 [batch_size, 1]
#         lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
#         
#         # 모빌리우스 뺄셈 [batch_size, 1]
#         y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
#         xy_inner_prod = torch.sum(x * y, dim=-1, keepdim=True)
#         
#         # 분자 계산 [batch_size, dim]
#         numerator = (1.0 - 2.0 * c * xy_inner_prod + c * y_norm_squared) * x
#         numerator = numerator - (1.0 - c * x_norm_squared) * y
#         
#         # 분모 계산 [batch_size, 1]
#         denominator = 1.0 - 2.0 * c * xy_inner_prod + c * c * x_norm_squared * y_norm_squared
#         
#         # 차이 벡터 [batch_size, dim]
#         diff = numerator / (denominator + eps)
#         
#         # 차이 벡터 노름 [batch_size, 1]
#         diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
#         diff_norm = torch.clamp(diff_norm, min=eps)
#         
#         # c를 텐서로 변환
#         c_torch::Tensor = torch.torch::Tensor(c, device=x.device, dtype=x.dtype)
#         sqrt_c = torch.sqrt(c_torch::Tensor)
#         
#         # 최종 결과 [batch_size, dim]
#         return 2.0 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * diff_norm) * diff / diff_norm
# 
# # GPU 최적화 버터플라이 변환
# class ButterflyTransform(torch.nn.Module):
#     def __init__(self, dim, device=None):
#         super(ButterflyTransform, self).__init__()
#         self.dim = dim
#         self.device = device
#         
#         # 차원이 2의 거듭제곱인지 확인하고 조정
#         self.log_dim = int(np.ceil(np.log2(dim)))
#         self.adjusted_dim = 2 ** self.log_dim
#         
#         # 각 레이어 및 블록별 파라미터 초기화
#         self.layers = torch.nn.ModuleList()
#         for layer in range(self.log_dim):
#             # 레이어의 블록 수 계산
#             block_size = 2 ** layer
#             num_blocks = self.adjusted_dim // (2 * block_size)
#             
#             # 회전 파라미터 (a, b) 초기화: a^2 + b^2 = 1 조건 만족
#             theta = torch.randn(num_blocks, device=device) * 0.01
#             a = torch.nn.Parameter(torch.cos(theta))
#             b = torch.nn.Parameter(torch.sin(theta))
#             
#             # 현재 레이어에 파라미터 추가
#             self.layers.append(torch.nn.ParameterList([a, b]))
#     
#     def forward(self, x):
#         """
#         배치 입력에 버터플라이 변환 적용
#         x: [batch_size, dim] 또는 [dim] 형태의 입력 텐서
#         """
#         # 1D 입력인 경우 배치 차원 추가
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#             single_input = True
#         else:
#             single_input = False
#         
#         batch_size = x.size(0)
#         
#         # 입력 차원이 2의 거듭제곱보다 작은 경우 패딩
#         if x.size(1) < self.adjusted_dim:
#             padding = torch.zeros(batch_size, self.adjusted_dim - x.size(1), device=x.device)
#             x_padded = torch.cat([x, padding], dim=1)
#         else:
#             x_padded = x[:, :self.adjusted_dim]
#         
#         # 각 버터플라이 레이어 적용
#         for layer_idx, (a, b) in enumerate(self.layers):
#             block_size = 2 ** layer_idx
#             
#             # 현재 레이어에 변환 적용
#             x_padded = self._apply_butterfly_layer(x_padded, a, b, block_size)
#         
#         # 원래 차원으로 복원
#         if single_input:
#             return x_padded[0, :x.size(1)]
#         else:
#             return x_padded[:, :x.size(1)]
#     
#     def _apply_butterfly_layer(self, x, a, b, block_size):
#         """
#         단일 버터플라이 레이어 적용 (GPU 최적화)
#         x: [batch_size, dim] 형태의 입력 텐서
#         a, b: 회전 파라미터
#         block_size: 현재 블록 크기
#         """
#         batch_size = x.size(0)
#         dim = x.size(1)
#         
#         # 블록 연산을 위한 텐서 재구성
#         num_blocks = dim // (2 * block_size)
#         x_view = x.view(batch_size, num_blocks, 2, block_size)
#         
#         # 회전 파라미터 준비
#         a_expanded = a.view(num_blocks, 1, 1).expand(num_blocks, 1, block_size)
#         b_expanded = b.view(num_blocks, 1, 1).expand(num_blocks, 1, block_size)
#         
#         # 배치 연산으로 회전 적용
#         x_rotated = torch.torch::zeros_like(x_view)
#         
#         # 첫 번째 행 회전
#         x_rotated[:, :, 0, :] = a_expanded * x_view[:, :, 0, :] + b_expanded * x_view[:, :, 1, :]
#         
#         # 두 번째 행 회전
#         x_rotated[:, :, 1, :] = -b_expanded * x_view[:, :, 0, :] + a_expanded * x_view[:, :, 1, :]
#         
#         # 원래 형태로 복원
#         return x_rotated.view(batch_size, dim)
# 
# # CUDA 최적화된 하이퍼볼릭 버터플라이 층
# class HyperButterflyLayer(torch.nn.Module):
#     def __init__(self, dim, num_layers, curvature=1.0, device=None):
#         super(HyperButterflyLayer, self).__init__()
#         self.dim = dim
#         self.num_layers = num_layers
#         self.curvature = curvature
#         self.device = device
#         
#         # 차원이 2의 거듭제곱인지 확인하고 조정
#         power = math.ceil(math.log2(dim))
#         self.butterfly_dim = 2 ** power
#         
#         # 입력층
#         self.fc_in = torch.nn.Linear(dim, self.butterfly_dim)
#         
#         # 버터플라이 변환 (O(n log n) 복잡도)
#         self.butterfly = ButterflyTransform(self.butterfly_dim, device=device)
#         
#         # 출력층
#         self.fc_out = torch.nn.Linear(self.butterfly_dim, dim)
#         
#         # 하이퍼볼릭 연산
#         self.hyper_ops = HyperbolicOperations()
#     
#     def forward(self, x):
#         """
#         하이퍼볼릭 버터플라이 변환 적용
#         x: 포인카레 볼 모델의 점 [batch_size, dim]
#         """
#         batch_size = x.size(0)
#         
#         # 1. 원점 생성 (로그/지수 사상 기준점)
#         origin = torch.torch::zeros_like(x)
#         
#         # 2. 하이퍼볼릭 공간 -> 접공간 (로그 사상)
#         v = self.hyper_ops.batch_poincare_log_map(origin, x, self.curvature)
#         
#         # 3. 선형 변환
#         v = self.fc_in(v)
#         
#         # 4. 버터플라이 변환 적용 (O(n log n))
#         v_transformed = self.butterfly(v)
#         
#         # 5. 출력 변환
#         v_transformed = self.fc_out(v_transformed)
#         
#         # 6. 접공간 -> 하이퍼볼릭 공간 (지수 사상)
#         result = self.hyper_ops.batch_poincare_exp_map(origin, v_transformed, self.curvature)
#         
#         return result
# 
# class HyperButterflyMLP(torch.nn.Module):
#     """
#     하이퍼볼릭 버터플라이 구조를 사용한 MLP
#     O(n log n) 복잡도의 효율적인 구현
#     """
#     def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, curvature=0.1, device=None):
#         super(HyperButterflyMLP, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.curvature = curvature
#         self.device = device
#         
#         # 차원이 2의 거듭제곱인지 확인하고 조정
#         power = int(np.ceil(np.log2(hidden_dim)))
#         self.butterfly_dim = 2 ** power
#         
#         # 입력층: 유클리드 -> 중간 표현
#         self.fc_in = torch.nn.Linear(input_dim, self.butterfly_dim)
#         self.bn_in = torch.nn.BatchNorm1d(self.butterfly_dim)
#         
#         # 버터플라이 변환 (O(n log n) 복잡도)
#         self.butterfly = ButterflyTransform(self.butterfly_dim, device=device)
#         
#         # 출력층
#         self.fc_out = torch.nn.Linear(self.butterfly_dim, output_dim)
#         
#         # 하이퍼볼릭 연산
#         self.hyper_ops = HyperbolicOperations()
#         
#         # 가중치 초기화
#         for m in self.modules():
#             if isinstance(m, torch.nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     
#     @torch.cuda.amp.autocast(enabled=True)  # 혼합 정밀도 연산
#     def forward(self, x):
#         batch_size = x.size(0)
#         
#         # 1. 플랫닝 및 첫 선형 변환
#         x = x.view(batch_size, -1)
#         h = torch.nn.functional.relu(self.bn_in(self.fc_in(x)))
#         
#         # 2. 유클리드 -> 하이퍼볼릭 공간 변환
#         h_hyp = self.hyper_ops.euclidean_to_poincare(h, self.curvature)
#         
#         # 3. 원점 기준 접공간으로 사상
#         origin = torch.torch::zeros_like(h_hyp)
#         v = self.hyper_ops.batch_poincare_log_map(origin, h_hyp, self.curvature)
#         
#         # 4. 버터플라이 변환 적용 (O(n log n) 복잡도)
#         v_transformed = self.butterfly(v)
#         
#         # 5. 하이퍼볼릭 공간으로 변환
#         h_transformed = self.hyper_ops.batch_poincare_exp_map(origin, v_transformed, self.curvature)
#         
#         # 6. 유클리드 공간으로 변환 (출력층 처리)
#         h_euc = self.hyper_ops.poincare_to_euclidean(h_transformed, self.curvature)
#         
#         # 7. 출력층
#         out = self.fc_out(h_euc)
#         
#         return torch.nn.functional.log_softmax(out, dim=1)
# 
# try:
#     from .csrc.riemannian_cuda import (
#         mobius_add_cuda,
#         mobius_scalar_mul_cuda,
#         poincare_distance_cuda,
#         expmap_cuda,
#         logmap_cuda,
#         butterfly_transform_forward_cuda,
#         butterfly_transform_backward_cuda
#     )
#     CUDA_AVAILABLE = True
# except ImportError:
#     CUDA_AVAILABLE = False
#     print("CUDA extensions not available. Using CPU implementation.")
#     
# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")