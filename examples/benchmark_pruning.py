import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.pruning import convert_to_pruned_riemannian

def benchmark_pruning():
    print("=== Reality Stone: Pruning & Speed Benchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. 모델 준비
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. 베이스라인 (Original) 측정
    print("\n--- 1. Original Model ---")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    speed_orig, output_orig = measure_speed(model, tokenizer, device, "Original")
    del model
    torch.cuda.empty_cache()

    # 3. 30% 가지치기 측정
    print("\n--- 2. Pruned Model (30% Removed) ---")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = convert_to_pruned_riemannian(model, curvature=1e-5, prune_ratio=0.3)
    model = model.to(device)
    speed_pruned_30, output_30 = measure_speed(model, tokenizer, device, "Pruned 30%")
    del model
    torch.cuda.empty_cache()
    
    # 4. 50% 가지치기 측정
    print("\n--- 3. Pruned Model (50% Removed) ---")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = convert_to_pruned_riemannian(model, curvature=1e-5, prune_ratio=0.5)
    model = model.to(device)
    speed_pruned_50, output_50 = measure_speed(model, tokenizer, device, "Pruned 50%")
    
    # 5. 결과 비교
    print("\n=== Summary ===")
    print(f"Original Speed: {speed_orig:.2f} tok/s")
    print(f"Pruned 30% Speed: {speed_pruned_30:.2f} tok/s (Diff: {speed_pruned_30 - speed_orig:.2f})")
    print(f"Pruned 50% Speed: {speed_pruned_50:.2f} tok/s (Diff: {speed_pruned_50 - speed_orig:.2f})")
    
    print("\nNote: PyTorch Dense Tensor 연산에서는 0을 곱해도 연산량은 동일하여 속도 향상이 즉시 나타나지 않습니다.")
    print("실제 속도 향상을 위해서는 'Sparse Tensor' 커널이나 전용 하드웨어 지원이 필요합니다.")
    print("하지만, 모델의 '용량(파라미터 수)' 관점에서는 0인 가중치를 저장하지 않음으로써 압축 가능합니다.")

def measure_speed(model, tokenizer, device, name):
    input_text = "The theory of relativity states"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Warmup
    model.generate(**inputs, max_new_tokens=10)
    
    # Benchmark
    start_time = time.time()
    num_tokens = 100
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=num_tokens, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    duration = end_time - start_time
    speed = num_tokens / duration
    
    print(f"[{name}] Speed: {speed:.2f} tokens/sec")
    print(f"[{name}] Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)[:100]}...")
    return speed, outputs

if __name__ == "__main__":
    benchmark_pruning()

