import argparse
import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_compression(model_id="microsoft/phi-2", device="cuda"):
    from reality_stone.compression import convert_to_manifold_compressed
    print(f"=== Reality Stone: Riemannian Compression Benchmark ({model_id}) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Loading & Compressing Model...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original Params: {orig_params:,}")
    
    model = convert_to_manifold_compressed(model, curvature=1e-4, compression_ratio=0.3)
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Manifold Params: {trainable_params:,}")
    
    run_inference(model, tokenizer, device, "Compression")

def benchmark_hyper_compression(model_id="microsoft/phi-2", device="cuda"):
    from reality_stone.hyper_compression import apply_hyper_compression
    print(f"=== Reality Stone: Hyper-Compression Benchmark ({model_id}) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    orig_params = sum(p.numel() for p in model.parameters())
    
    print("\nApplying Hyper-Compression (Target Dim: 64)...")
    model = apply_hyper_compression(model, target_dim=64)
    model = model.to(device)
    
    compressed_params = sum(p.numel() for p in model.parameters())
    ratio = (1 - compressed_params / orig_params) * 100
    print(f"Original: {orig_params:,} -> Compressed: {compressed_params:,} (Reduc: {ratio:.2f}%)")
    
    run_inference(model, tokenizer, device, "Hyper-Compression")

def benchmark_pruning(model_id="microsoft/phi-2", device="cuda"):
    from reality_stone.pruning import convert_to_pruned_riemannian
    print(f"=== Reality Stone: Pruning Benchmark ({model_id}) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("\n--- 1. Original Model ---")
    model_orig = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    measure_speed(model_orig, tokenizer, device, "Original")
    del model_orig
    torch.cuda.empty_cache()

    print("\n--- 2. Pruned Model (30%) ---")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = convert_to_pruned_riemannian(model, curvature=1e-5, prune_ratio=0.3)
    model = model.to(device)
    measure_speed(model, tokenizer, device, "Pruned 30%")
    del model
    torch.cuda.empty_cache()
    
    print("\n--- 3. Pruned Model (50%) ---")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = convert_to_pruned_riemannian(model, curvature=1e-5, prune_ratio=0.5)
    model = model.to(device)
    measure_speed(model, tokenizer, device, "Pruned 50%")

def benchmark_extraction(model_id="gpt2", device="cuda"):
    from reality_stone.metric_extraction import extract_riemannian_metric
    print(f"=== Reality Stone: Metric Extraction Benchmark ({model_id}) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    orig_params = sum(p.numel() for p in model.parameters())
    
    print("Extracting Riemannian Metric (Rank 32)...")
    model = extract_riemannian_metric(model, target_dim=32)
    model = model.to(device)
    
    new_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {orig_params:,} -> {new_params:,}")
    
    run_inference(model, tokenizer, device, "Extraction")

def benchmark_tuned_model(model_path="./reality-stone-phi2-tuned", device="cuda"):
    from reality_stone.conversion import convert_to_full_riemannian
    print(f"=== Benchmarking Tuned Model: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found. Run training first.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Loading base model and re-applying surgery...")
        # Assuming base was phi-2, as per training script
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        model = convert_to_full_riemannian(model, curvature=1e-5)
        
        # Load state dict if needed, or rely on from_pretrained if saved correctly
        # For robustness, we reload weights from the tuned path if compatible
        # simple load for now:
        model_tuned = AutoModelForCausalLM.from_pretrained(model_path)
        # Note: Ideally we should transfer weights if classes differ, 
        # but here we assume direct load works or we use the loaded model directly if classes match.
        model = model_tuned 
        
    except Exception as e:
        print(f"Load failed: {e}")
        return

    model = model.to(device)
    measure_speed(model, tokenizer, device, "Tuned Model")

def run_inference(model, tokenizer, device, tag):
    input_text = "The nature of reality is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    print(f"\nGenerating ({tag})...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
    dur = time.time() - start
    
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print(f"Time: {dur:.2f}s")

def measure_speed(model, tokenizer, device, name):
    input_text = "The theory of relativity states"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Warmup
    model.generate(**inputs, max_new_tokens=5)
    
    start_time = time.time()
    num_tokens = 50
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=num_tokens, 
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    speed = num_tokens / (end_time - start_time)
    print(f"[{name}] Speed: {speed:.2f} tok/s | Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)[:50]}...")
    return speed

def main():
    from reality_stone.utils.misc import get_device
    parser = argparse.ArgumentParser(description="Reality Stone Benchmarks")
    parser.add_argument("mode", choices=["compression", "hyper", "pruning", "extraction", "tuned", "all"], help="Benchmark mode")
    parser.add_argument("--model", default="microsoft/phi-2", help="Base model ID")
    parser.add_argument("--path", default="./reality-stone-phi2-tuned", help="Path for tuned model")
    
    args = parser.parse_args()
    device = get_device()
    print(f"Device: {device}")

    if args.mode in ["compression", "all"]:
        benchmark_compression(args.model, device)
    if args.mode in ["hyper", "all"]:
        benchmark_hyper_compression(args.model, device)
    if args.mode in ["pruning", "all"]:
        benchmark_pruning(args.model, device)
    if args.mode in ["extraction", "all"]:
        benchmark_extraction("gpt2", device) # GPT2 default for extraction demo
    if args.mode in ["tuned", "all"]:
        benchmark_tuned_model(args.path, device)

if __name__ == "__main__":
    main()

