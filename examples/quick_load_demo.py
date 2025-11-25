import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.metric_extraction import apply_metric_structure_only
import os
import time

def quick_load_demo():
    print("=== Reality Stone: Fast Loading Demo ===")
    
    model_id = "Qwen/Qwen2.5-0.5B"
    save_path = "./reality-stone-qwen-tuned" # 이전 단계에서 저장된 경로
    
    # 저장된 파일이 있는지 확인
    if not os.path.exists(save_path):
        print("Tuned model not found. Please run fine-tuning first.")
        return

    print("1. Loading Base Model (CPU/RAM)...")
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    # 빈 깡통 로드 (빠름)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    print("2. Applying Structure (Skip SVD)...")
    start = time.time()
    # SVD 계산 없이 구조만 변경 (0.1초 소요)
    model = apply_metric_structure_only(model, target_dim=64)
    print(f"Structure Applied: {time.time() - start:.4f}s")
    
    print("3. Loading Learned Weights...")
    # 저장된 가중치 덮어쓰기
    # pytorch_model.bin에서 state_dict 로드
    state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
    # strict=False를 주어 불필요한 키 무시 (혹시 모를 버전 차이 대비)
    model.load_state_dict(state_dict, strict=False)
    
    print("4. Moving to GPU & Inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    quick_load_demo()

