# debug.py
import os
import subprocess
import sys

print("=== 현재 환경변수 ===")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not Set')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not Set')}")
print(f"DISTUTILS_USE_SDK: {os.environ.get('DISTUTILS_USE_SDK', 'Not Set')}")

print("\n=== 서브프로세스에서 환경변수 ===")
result = subprocess.run([sys.executable, "-c", "import os; print(os.environ.get('CUDA_HOME', 'Not Found'))"], 
                       capture_output=True, text=True)
print(f"Subprocess CUDA_HOME: {result.stdout.strip()}")