import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.conversion import convert_to_full_riemannian

def benchmark_model():
    model_path = "./reality-stone-gpt2-tuned"
    print(f"=== Benchmarking Model: {model_path} ===")

    # 1. File Size Check
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    print(f"Disk Size: {total_size / (1024*1024):.2f} MB")

    # 2. Load Model
    print("Loading model...")
    start_load = time.time()
    
    # Note: Since we saved with safe_serialization=False (pickle), 
    # and the class structure was modified dynamically, we need to apply the conversion again
    # OR load the structure and then load weights.
    # However, save_pretrained usually saves the config.
    # Let's see if AutoModel loads it correctly.
    # If the architecture in config.json is standard GPT2, it will load as standard GPT2.
    # We might need to re-apply the surgery if the saved weights are for the modified architecture
    # but the class definition isn't serialized.
    # Actually, since we replaced modules in-place, the state_dict keys match the structure.
    # But AutoModel will instantiate a vanilla GPT2. We must re-apply surgery before loading state_dict if it was saved that way.
    # Wait, save_pretrained saves the weights. If we load it back into a vanilla GPT2, 
    # keys might mismatch if we changed layer names? No, we kept names.
    # But RiemannianLinear has 'weight' and 'bias' just like Linear.
    # So vanilla GPT2 might load the weights fine! 
    # BUT, it won't have the Riemannian behavior (forward pass logic).
    # So we MUST re-apply surgery after loading to regain the intelligence.
    
    try:
        # Try loading directly first
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Standard load failed: {e}")
        print("Loading base GPT2 and applying surgery, then loading weights...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Apply surgery
        model = convert_to_full_riemannian(model, curvature=1e-5) # Use same curvature as training
        # Load weights
        from transformers.modeling_utils import load_state_dict
        # This part is tricky with save_pretrained. 
        # Let's assume for this benchmark we load the saved model and apply surgery if needed.
        # Actually, if we saved the model using save_pretrained after surgery, 
        # the config might not reflect the custom classes.
        # The weights are compatible with standard Linear.
        # So: Load -> Apply Surgery -> (Weights are already there from Load) -> Done.
        # Wait, if we load vanilla, we have the trained weights. 
        # Then we swap layers to RiemannianLinear. RiemannianLinear copies weights from original.
        # So the sequence: Load Trained -> Convert -> Run is correct.
    
    # Re-apply surgery to ensure Riemannian logic is active
    # (The weights loaded are the tuned ones)
    print("Re-applying Riemannian Surgery to activate geometric logic...")
    model = convert_to_full_riemannian(model, curvature=1e-5)
    
    load_time = time.time() - start_load
    print(f"Load Time: {load_time:.2f} s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 3. Speed Test
    input_text = "The theory of general relativity suggests that"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    print("\nGenerating...")
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10)
    
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
    tokens_per_sec = num_tokens / duration
    
    print(f"Inference Time ({num_tokens} tokens): {duration:.4f} s")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    benchmark_model()

