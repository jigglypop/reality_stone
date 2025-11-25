# HuggingFace 공통 루트
export HF_HOME=E:/hf-cache

# transformers / hub 캐시 경로 분리 (선택이지만 추천)
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"
source ./.venv/Scripts/activate