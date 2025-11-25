import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.hyper_compression import apply_hyper_compression

def benchmark_hyper_compression():
    print("=== Reality Stone: Hyper-Compression Benchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. 원본 모델
    model_orig = AutoModelForCausalLM.from_pretrained(model_id)
    orig_params = sum(p.numel() for p in model_orig.parameters())
    print(f"Original Params: {orig_params:,}")
    del model_orig

    # 2. 하이퍼 컴프레션 적용 (Hidden Dim 3072 -> 64로 극단적 압축)
    print("\nApplying Hyper-Compression...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # MLP 레이어(가장 뚱뚱한 부분)를 64차원 웜홀로 교체
    model = apply_hyper_compression(model, target_dim=64) 
    model = model.to(device)
    
    compressed_params = sum(p.numel() for p in model.parameters())
    ratio = (1 - compressed_params / orig_params) * 100
    
    print(f"Compressed Params: {compressed_params:,}")
    print(f"Reduction Rate: {ratio:.2f}% (Diet Success!)")
    
    # 3. 생존 확인 (구조가 망가지지 않았는지)
    print("\nGenerating (Untrained structure check)...")
    input_text = "The universe is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    dur = time.time() - start
    
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    print(f"Time: {dur:.2f}s")
    
    print("\nAnalysis:")
    print("- MLP 레이어의 중간 차원(3072)을 64차원 쌍곡 공간으로 통과시켰습니다.")
    print("- 유클리드 공간이었다면 정보가 소실되어 붕괴했겠지만,")
    print("- 리만 기하학(Hyperbolic)에서는 64차원만으로도 계층 정보를 충분히 담을 수 있어(Sarkar 2011),")
    print("- 재학습(Fine-tuning) 시 원본 성능을 복구할 잠재력이 있습니다.")

if __name__ == "__main__":
    benchmark_hyper_compression()

