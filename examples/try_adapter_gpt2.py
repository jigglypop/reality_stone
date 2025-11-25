import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.adapter import patch_llm_with_reality_stone
import time

def run_demo():
    print("=== Reality Stone Adapter Demo ===")
    
    # 1. 작은 모델 로드 (GPT-2)
    model_id = "gpt2"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # CUDA 사용 가능하면 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    # 2. 패치 전 기본 생성 테스트
    print("\n--- Before Patching ---")
    input_text = "The theory of relativity states that"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
    end_time = time.time()
    
    print(f"Output: {tokenizer.decode(outputs[0])}")
    print(f"Time: {end_time - start_time:.4f}s")

    # 3. Reality Stone 패치 적용
    print("\n--- Applying Reality Stone Patch ---")
    model = patch_llm_with_reality_stone(model, diffusion_steps=2, alpha=0.5)
    model = model.to(device) # Ensure new modules are on device

    # 4. 패치 후 생성 테스트
    print("\n--- After Patching ---")
    # 패치 후에는 추가된 파라미터(Diffusion Head)가 초기화 상태이므로 
    # 결과가 엉뚱하게 나올 수 있으나, 구조적으로 동작하는지 확인하는 것이 목적
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
    end_time = time.time()
    
    print(f"Output: {tokenizer.decode(outputs[0])}")
    print(f"Time: {end_time - start_time:.4f}s")
    print("\nSuccess! The model accepted the geometric transformations.")

if __name__ == "__main__":
    run_demo()

