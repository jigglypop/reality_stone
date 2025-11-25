import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.metric_extraction import extract_riemannian_metric
import time
import os

def run_llama_qwen_extraction():
    print("=== Reality Stone: Large Model Extraction Test ===")
    
    # 1. 모델 선택 (로컬에 있는 가벼운 모델 권장, 없으면 자동 다운로드)
    # Qwen2.5-0.5B 또는 Llama-3.2-1B 같은 작은 모델로 테스트
    model_id = "Qwen/Qwen2.5-0.5B" # 또는 "meta-llama/Llama-3.2-1B"
    
    print(f"Loading model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have access to the model or choose a public one.")
        return

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original Params: {orig_params:,}")

    # 2. 메트릭 추출 (Rank 64)
    # 0.5B 모델이라도 64차원이면 매우 큰 압축률을 보임
    print("\n--- Extracting Riemannian Metrics (This may take a moment) ---")
    start_time = time.time()
    
    # Qwen/Llama는 Linear 레이어 이름이 다를 수 있음 (e.g. gate_proj, up_proj)
    # metric_extraction.py는 nn.Linear를 감지하므로 자동으로 작동해야 함
    model = extract_riemannian_metric(model, target_dim=64)
    
    extraction_time = time.time() - start_time
    print(f"Extraction Time: {extraction_time:.2f}s")
    
    new_params = sum(p.numel() for p in model.parameters())
    print(f"Extracted Params: {new_params:,}")
    print(f"Compression Ratio: {(1 - new_params/orig_params)*100:.2f}%")

    # 3. 추론 테스트 (압축 직후)
    print("\n--- Inference Test (Zero-Shot after Compression) ---")
    input_text = "The future of artificial intelligence involves"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
    
    print(f"Input: {input_text}")
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    
    # 4. 저장 (선택)
    save_path = "./reality-stone-qwen-extracted"
    print(f"\nSaving to {save_path}...")
    try:
        # safe_serialization=False is needed due to shared tensors in some models
        model.save_pretrained(save_path, safe_serialization=False)
        tokenizer.save_pretrained(save_path)
        print("Saved successfully.")
        
        # 사이즈 확인
        total_size = 0
        for dirpath, _, filenames in os.walk(save_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"Disk Size: {total_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Save failed: {e}")

if __name__ == "__main__":
    run_llama_qwen_extraction()

