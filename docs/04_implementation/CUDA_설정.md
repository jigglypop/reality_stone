# Reality Stone CUDA 설정 가이드

이 문서는 Reality Stone을 CUDA 지원 모드로 빌드하고 실행하는 방법을 설명합니다.

## 사전 요구사항

### 1. NVIDIA GPU 및 드라이버
- CUDA 지원 NVIDIA GPU 필요
- 최신 NVIDIA 드라이버 설치 필수
- 드라이버 확인:
  ```bash
  nvidia-smi
  ```

### 2. CUDA Toolkit
- CUDA 12.1 이상 권장
- 설치 경로: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- 환경 변수 확인:
  ```bash
  echo $CUDA_PATH
  ```
