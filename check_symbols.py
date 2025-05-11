import os
import torch

# PyTorch가 사용하는 CUDA 버전 확인
print(f"PyTorch CUDA version: {torch.version.cuda}")

# PyTorch lib 디렉토리 내용 확인
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
print(f"\nPyTorch lib directory: {torch_lib}")

# .lib 파일들 찾기
lib_files = [f for f in os.listdir(torch_lib) if f.endswith('.lib')]
print("\nAvailable .lib files:")
for lib in sorted(lib_files):
    print(f"  {lib}")

# CUDA 설치 디렉토리들 확인
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
]

for cuda_path in cuda_paths:
    lib_path = os.path.join(cuda_path, "lib", "x64")
    if os.path.exists(lib_path):
        print(f"\n{cuda_path} libraries:")
        cuda_libs = [f for f in os.listdir(lib_path) if f.endswith('.lib') and 'cudart' in f]
        for lib in cuda_libs:
            print(f"  {lib}")