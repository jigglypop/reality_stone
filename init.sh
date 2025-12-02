cd /e/reality_stone

# 1) .venv 안에 pip 먼저 깔기
./.venv/Scripts/python -m ensurepip --upgrade

# 2) CPU 토치 제거
./.venv/Scripts/python -m pip uninstall -y torch torchvision torchaudio

# 3) CUDA 12.1 빌드 설치
./.venv/Scripts/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121

# 4) CUDA 인식 확인
./.venv/Scripts/python - << 'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name", torch.cuda.get_device_name(0))
PY

# 5) (4)에서 cuda_available=True 나오면, 같은 python으로 테스트 실행
./.venv/Scripts/python experiments/test_gpt2_conversion.py