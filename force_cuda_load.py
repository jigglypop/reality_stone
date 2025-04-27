"""
CUDA 확장 강제 로드 스크립트
- CUDA 확장이 로드될 때까지 계속 시도
- 다양한 방법으로 로딩 시도
- 상세한 디버깅 정보 출력
"""

import os
import sys
import time
import glob
import platform
import subprocess
import importlib.util
import importlib.machinery
import ctypes
from ctypes import cdll
import time
import traceback

# 환경 변수 설정 (CUDA 버전 오류 무시)
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TORCH_USE_RTLD_GLOBAL'] = 'YES'
os.environ['TORCH_NVCC_FLAGS'] = '--expt-relaxed-constexpr'
os.environ['TORCH_ALLOW_CUDA_VERSION_MISMATCH'] = '1'  # 버전 불일치 허용
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'  # C++ 스택 트레이스 표시
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 첫 번째 GPU만 사용
os.environ['PYTHONIOENCODING'] = 'utf-8'  # 한글 출력 지원

print(f"시스템 정보: {platform.system()} {platform.release()} {platform.machine()}")
print(f"Python 버전: {sys.version}")
print(f"환경 변수:")
for env in ['CUDA_MODULE_LOADING', 'PYTORCH_CUDA_ALLOC_CONF', 'TORCH_USE_RTLD_GLOBAL', 
           'TORCH_NVCC_FLAGS', 'TORCH_ALLOW_CUDA_VERSION_MISMATCH', 'CUDA_VISIBLE_DEVICES']:
    print(f"  {env}: {os.environ.get(env, '설정되지 않음')}")

print("\n시스템 경로:")
for p in sys.path:
    print(f"  {p}")

# PyTorch 로드
print("\nPyTorch 로드 중...")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA 버전: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU 개수: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  현재 메모리 사용량: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
            print(f"  최대 메모리 용량: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f} MB")
    
    # 간단한 CUDA 텐서 생성
    x = torch.randn(10, 10)
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"CUDA 텐서 생성 성공: {x.device}")
        
except Exception as e:
    print(f"PyTorch 로드 오류: {e}")
    traceback.print_exc()
    sys.exit(1)

# 소스 파일 확인
print("\n소스 파일 확인 중...")
src_files_patterns = [
    "riemannian_manifold/csrc/*.cu",
    "riemannian_manifold/csrc/*.cpp",
    "riemannian_manifold/csrc/setup.py"
]

for pattern in src_files_patterns:
    files = glob.glob(pattern)
    if files:
        print(f"패턴 '{pattern}'에서 파일 발견:")
        for file in files:
            file_size = os.path.getsize(file) / 1024
            print(f"  {file} ({file_size:.2f} KB)")

# 빌드 상태 확인
print("\n확장 모듈 빌드 상태 확인 중...")
build_files_patterns = [
    "riemannian_manifold/build/**/*.pyd",
    "riemannian_manifold/build/**/*.so",
    "riemannian_manifold/**/_C.*.pyd",
    "**/*.pyd",
    "**/*.so"
]

for pattern in build_files_patterns:
    files = glob.glob(pattern, recursive=True)
    if files:
        print(f"패턴 '{pattern}'에서 파일 발견:")
        for file in files:
            file_size = os.path.getsize(file) / 1024
            file_time = os.path.getmtime(file)
            print(f"  {file} ({file_size:.2f} KB, {time.ctime(file_time)})")

# CUDA 확장을 직접 로드 시도
def try_load_cuda_extension(file_path, verbose=True):
    """CUDA 확장 모듈 로드 시도 함수"""
    if verbose:
        print(f"\n파일 {file_path} 로드 시도 중...")
    
    try:
        # 1. ctypes 방식으로 로드
        if platform.system() == 'Windows':
            lib = cdll.LoadLibrary(file_path)
        else:
            lib = ctypes.CDLL(file_path)
        
        if verbose:
            print(f"ctypes로 로드 성공: {lib}")
        
        # 2. importlib 방식으로 로드
        spec = importlib.util.spec_from_file_location("riemannian_cuda", file_path)
        if spec is None:
            if verbose:
                print(f"importlib spec 생성 실패")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 3. 함수 존재 여부 확인
        has_functions = False
        for func_name in ["create_butterfly_matrix", "unified_butterfly_transform", "butterfly_factor"]:
            if hasattr(module, func_name):
                has_functions = True
                if verbose:
                    print(f"함수 {func_name} 발견")
        
        if has_functions:
            if verbose:
                print(f"모듈 로드 성공: {module}")
            return module
        else:
            if verbose:
                print(f"필요한 함수가 없음")
            return None
    except Exception as e:
        if verbose:
            print(f"로드 실패: {e}")
            traceback.print_exc()
        return None

# 전체 확장 파일에서 CUDA 모듈 로드 시도
print("\nCUDA 확장 모듈 로드 시도 중...")
cuda_modules = []
cuda_module_files = []

# 1. 직접 디렉토리 탐색
extension_files = []
for pattern in ["**/*.pyd", "**/*.so"]:
    extension_files.extend(glob.glob(pattern, recursive=True))

print(f"발견된 확장 파일: {len(extension_files)}개")
for file_path in extension_files:
    if "riemannian" in file_path.lower() or "cuda" in file_path.lower():
        result = try_load_cuda_extension(file_path)
        if result:
            cuda_modules.append(result)
            cuda_module_files.append(file_path)
            print(f"*** 성공적으로 로드됨: {file_path}")

# 2. 임포트 방식으로 시도
import_candidates = [
    "riemannian_manifold.csrc.riemannian_cuda",
    "riemannian_cuda",
    "riemannian_manifold._C"
]

for candidate in import_candidates:
    try:
        print(f"임포트 시도: {candidate}")
        module = importlib.import_module(candidate)
        print(f"임포트 성공: {module}")
        
        # 함수 확인
        for func_name in ["create_butterfly_matrix", "unified_butterfly_transform", "butterfly_factor"]:
            if hasattr(module, func_name):
                print(f"함수 {func_name} 발견")
        
        cuda_modules.append(module)
    except ImportError as e:
        print(f"임포트 실패: {e}")

# 버터플라이 변환 클래스 (버터플라이_컴패리슨.py에서 가져옴)
class ButterflyTransform(nn.Module):
    def __init__(self, dim, device=None):
        super(ButterflyTransform, self).__init__()
        self.dim = dim
        self.device = device
        
        # 2의 거듭제곱으로 조정
        self.log_dim = int(np.ceil(np.log2(dim)))
        self.adjusted_dim = 2 ** self.log_dim
        
        # 각 레이어별 회전 파라미터
        self.thetas = nn.Parameter(torch.randn(self.log_dim * self.adjusted_dim // 2, device=device) * 0.01)
        
        # CUDA 확장 모듈 찾기
        self._initialize_optimized_weights()
    
    def _initialize_optimized_weights(self):
        """최적화된 가중치 행렬 초기화"""
        print("최적화된 버터플라이 가중치 행렬 구축 중...")
        self.weight = nn.Parameter(torch.eye(self.adjusted_dim, device=self.device))
        
        # 행렬 구축 (for문 없는 효율적인 방식)
        self._build_butterfly_matrix_optimized()
    
    def _build_butterfly_matrix_optimized(self):
        """버터플라이 행렬 구축 (효율적인 방법)"""
        try:
            # 버터플라이 커널 찾기
            print("CUDA 확장 모듈 시도 중...")
            try:
                # 1. 직접 임포트
                from riemannian_manifold.csrc.riemannian_cuda import create_butterfly_matrix
                matrix = create_butterfly_matrix(self.thetas, self.adjusted_dim, self.log_dim)
                print("CUDA 확장에서 버터플라이 행렬 생성 성공!")
                with torch.no_grad():
                    self.weight.copy_(matrix)
                return
            except ImportError as e1:
                print(f"첫 번째 시도 실패: {e1}")
                try:
                    # 2. 대체 방법으로 임포트
                    import importlib.util
                    import sys
                    
                    # 확장 모듈 파일 직접 찾기
                    extension_paths = glob.glob("**/*.pyd", recursive=True)
                    for path in extension_paths:
                        if "riemannian" in path.lower() or "cuda" in path.lower():
                            print(f"후보 확장 파일: {path}")
                            try:
                                spec = importlib.util.spec_from_file_location("riemannian_cuda", path)
                                riemannian_cuda = importlib.util.module_from_spec(spec)
                                sys.modules["riemannian_cuda"] = riemannian_cuda
                                spec.loader.exec_module(riemannian_cuda)
                                
                                # 함수 확인
                                create_butterfly_matrix = getattr(riemannian_cuda, "create_butterfly_matrix", None)
                                if create_butterfly_matrix:
                                    print(f"파일 {path}에서 create_butterfly_matrix 함수 발견!")
                                    matrix = create_butterfly_matrix(self.thetas, self.adjusted_dim, self.log_dim)
                                    print("CUDA 확장에서 버터플라이 행렬 생성 성공!")
                                    with torch.no_grad():
                                        self.weight.copy_(matrix)
                                    return
                            except Exception as e:
                                print(f"파일 {path} 로드 시도 중 오류: {e}")
                    
                    print("적합한 확장 파일을 찾을 수 없습니다. PyTorch 구현으로 계속합니다.")
                except Exception as e2:
                    print(f"대체 로드 방법 실패: {e2}")
        except Exception as e:
            print(f"CUDA 확장 로드 오류: {e}")
        
        print("순수 PyTorch 구현으로 행렬 생성 중...")
        weight = torch.eye(self.adjusted_dim, device=self.device)
        
        # log_dim 레이어의 버터플라이 변환 적용
        for layer in range(self.log_dim):
            block_size = 2 ** layer
            num_blocks = self.adjusted_dim // (2 * block_size)
            
            # 현재 레이어의 가중치
            layer_weight = torch.eye(self.adjusted_dim, device=self.device)
            
            # theta 인덱스 계산
            param_offset = 0
            for l in range(layer):
                param_offset += (self.adjusted_dim // (2 * (2 ** l)))
            
            # 블록별 처리
            for b in range(num_blocks):
                theta_idx = param_offset + b
                if theta_idx < self.thetas.size(0):
                    # 텐서에서 스칼라 값을 가져올 때는 item() 사용
                    theta_val = self.thetas[theta_idx].item()
                    cos_val = math.cos(theta_val)  # 스칼라 값에 math.cos 사용
                    sin_val = math.sin(theta_val)  # 스칼라 값에 math.sin 사용
                    
                    block_start = b * 2 * block_size
                    
                    # 회전 행렬 생성
                    for i in range(block_size):
                        idx1 = block_start + i
                        idx2 = block_start + block_size + i
                        
                        if idx1 < self.adjusted_dim and idx2 < self.adjusted_dim:
                            # 단일 회전 행렬
                            rotation = torch.eye(self.adjusted_dim, device=self.device)
                            rotation[idx1, idx1] = cos_val
                            rotation[idx1, idx2] = sin_val
                            rotation[idx2, idx1] = -sin_val
                            rotation[idx2, idx2] = cos_val
                            
                            # 현재 레이어 가중치에 회전 적용
                            layer_weight = rotation @ layer_weight
            
            # 전체 가중치에 현재 레이어 가중치를 적용
            weight = layer_weight @ weight
        
        # 최종 가중치 설정
        with torch.no_grad():
            self.weight.copy_(weight)
        print("순수 PyTorch로 버터플라이 행렬 생성 완료")
    
    def forward(self, x):
        """버터플라이 변환 적용"""
        # 배치 차원 추가
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = x.size(0)
        
        # 입력 크기 맞추기
        if x.size(1) < self.adjusted_dim:
            padding = torch.zeros(batch_size, self.adjusted_dim - x.size(1), device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :self.adjusted_dim]
        
        # 단일 행렬 곱셈으로 변환 적용
        output = x_padded @ self.weight.t()
        
        # 원래 크기로 자르기
        if single_input:
            return output[0, :x.size(1)]
        else:
            return output[:, :x.size(1)]

# CUDA 모듈 테스트
if cuda_modules:
    print("\nCUDA 모듈 테스트 중...")
    for i, module in enumerate(cuda_modules):
        print(f"\n모듈 {i+1}: {module} (파일: {cuda_module_files[i] if i < len(cuda_module_files) else '알 수 없음'})")
        
        # 함수 확인
        for func_name in ["create_butterfly_matrix", "unified_butterfly_transform", "butterfly_factor"]:
            if hasattr(module, func_name):
                print(f"  함수 {func_name} 존재")
                
                # 테스트 실행
                try:
                    if func_name == "create_butterfly_matrix":
                        dim = 8
                        log_dim = int(np.ceil(np.log2(dim)))
                        thetas = torch.randn(log_dim * dim // 2, device="cuda" if torch.cuda.is_available() else "cpu")
                        
                        # 함수 호출
                        start_time = time.time()
                        result = getattr(module, func_name)(thetas, dim, log_dim)
                        elapsed = time.time() - start_time
                        print(f"  결과: 형태 {result.shape}, 시간 {elapsed:.6f}초")
                    elif func_name == "unified_butterfly_transform":
                        # 테스트 데이터 생성
                        dim = 8
                        batch_size = 2
                        x = torch.randn(batch_size, dim, device="cuda" if torch.cuda.is_available() else "cpu")
                        weight = torch.eye(dim, device=x.device)
                        
                        # 함수 호출
                        start_time = time.time()
                        result = getattr(module, func_name)(x, weight)
                        elapsed = time.time() - start_time
                        print(f"  결과: 형태 {result.shape}, 시간 {elapsed:.6f}초")
                except Exception as e:
                    print(f"  테스트 실행 오류: {e}")
            else:
                print(f"  함수 {func_name} 없음")
else:
    print("\nCUDA 모듈을 찾을 수 없습니다.")

# 버터플라이 변환 성능 테스트
print("\n버터플라이 변환 성능 테스트 중...")
try:
    # 대상 차원 및 배치 크기 설정
    dims = [64, 128, 256]  # 작은 크기로 조정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for dim in dims:
        # 입력 데이터 생성
        batch_size = 32
        x = torch.randn(batch_size, dim, device=device)
        
        # 버터플라이 변환 인스턴스 생성
        print(f"\n차원 {dim} 테스트 중...")
        butterfly = ButterflyTransform(dim, device=device)
        butterfly.to(device)
        
        # 성능 측정
        iterations = 5
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(iterations):
            output = butterfly(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        elapsed = time.time() - start_time
        print(f"차원 {dim}: {iterations}회 반복 평균 {elapsed/iterations:.6f}초, 출력 형태: {output.shape}")
except Exception as e:
    print(f"성능 테스트 오류: {e}")
    traceback.print_exc()

# 종합 결과
print("\n종합 결과:")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"PyTorch CUDA 버전: {torch.version.cuda}")
print(f"확장 모듈 로드 성공: {'성공' if cuda_modules else '실패'}")
if cuda_modules:
    print(f"로드된 모듈 수: {len(cuda_modules)}")
    for i, module in enumerate(cuda_modules):
        print(f"  모듈 {i+1}: {module}")
print(f"버터플라이 변환 테스트: {'성공' if 'output' in locals() else '실패'}")
print("\n프로그램 종료") 