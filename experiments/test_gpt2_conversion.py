import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from reality_stone.models.transformer_converter import RSULFTransformerConverter, RSULFModel
import time

def analyze_layer_fidelity(original_model, rs_model, tokenizer, prompt, device):
    original_model.eval()
    rs_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = original_model.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = list(outputs.hidden_states)
    x = hidden_states[0]
    rs_outputs = []
    h = x
    for wrapper in rs_model.wrappers:
        h = wrapper(h)
        rs_outputs.append(h.detach())
    n_layers = min(len(hidden_states) - 1, len(rs_outputs))
    print("\n[RS-ULF] Layer-wise similarity")
    for i in range(n_layers):
        o = hidden_states[i + 1]
        r = rs_outputs[i]
        o_flat = o.view(-1, o.size(-1))
        r_flat = r.view(-1, r.size(-1))
        cos = F.cosine_similarity(o_flat, r_flat, dim=-1).mean().item()
        rel = (o_flat - r_flat).norm() / (o_flat.norm() + 1e-8)
        print(f"   layer {i:02d}: cos={cos:.4f}, rel_l2={rel:.4f}")

def test_gpt2_conversion():
    print("=== [Reality Stone] GPT-2 Conversion Test Start ===")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Device: cpu (CUDA not available)")
        print("WARNING: User requested CUDA but it is not available. Running on CPU.")
    # 1. Load Original GPT-2
    print("\n1. Loading Original GPT-2...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    original_model.eval()

    prompt = "The secret of the universe is"
    # Fix: Generate attention_mask
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2. Generate with Original
    print("   Generating with Original...")
    start = time.time()
    with torch.no_grad():
        # Fix: Pass attention_mask and set pad_token_id explicitly
        out_ids = original_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=30, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Prevent "unified, unified" loop
        )
    orig_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"   [Original]: {orig_text}")
    print(f"   Time: {time.time() - start:.4f}s")
    # 3. Convert to RS-ULF (Structure Mapping Mode)
    print("\n2. Converting to RS-ULF (Structure Mapping Mode)...")
    config = {
        "d_model": original_model.config.n_embd,
        "r": max(256, original_model.config.n_embd // 3),
        "eta": 0.005,
        "alpha": 0.01,
        "beta": 0.0,
        "gamma": 0.99,
        "seq_len": 64,
        "window": 4,
        "verbose": True
    }
    full_rank_r = config["r"]

    # Fix: Use **config to unpack arguments, otherwise config dict is passed as d_model
    converter = RSULFTransformerConverter(**config)
    rs_layers = converter.convert_model(original_model)
    rs_layers = rs_layers.to(device)

    analyze_layer_fidelity(original_model, rs_layers, tokenizer, prompt, device)
    
    def rsulf_generate_text(rs_model_stack, text_prompt, max_tokens=30):
        curr_ids = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
        
        # Use embedding and final norm/head from original model
        wte = original_model.transformer.wte
        wpe = original_model.transformer.wpe
        lm_head = original_model.lm_head
        ln_f = original_model.transformer.ln_f
        
        generated = curr_ids
        start_gen = time.time()
        
        for _ in range(max_tokens):
            seq_len = generated.size(1)
            pos = torch.arange(seq_len, dtype=torch.long, device=device)
            
            with torch.no_grad():
                tok_emb = wte(generated)
                pos_emb = wpe(pos)
                x = tok_emb + pos_emb
                
                # Fix: RS-ULF model stack handles the full layer transition.
                # Do NOT inject original LayerNorms or manual residual connections here
                # unless RS-ULF specifically requires it (which RSULFModel.forward does not).
                
                # Sync before transferring to CPU-heavy Rust layer to prevent async errors
                if x.is_cuda:
                    torch.cuda.synchronize()
                    
                h = rs_model_stack(x)
                
                # 3. Final Norm & Head
                h_last = ln_f(h)
                logits = lm_head(h_last)
                next_token_logits = logits[:, -1, :]
                
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        gen_time = time.time() - start_gen
        return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time

    rs_text, rs_time = rsulf_generate_text(rs_layers, prompt)
    print(f"   [RS-ULF Converted]: {rs_text}")
    print(f"   Time: {rs_time:.4f}s")

    # 5. Convert to RS-ULF (Compressed Mode: SVD Rank 64)
    print("\n3. Converting to RS-ULF (High Compression Mode - Rank 64)...")
    config["r"] = 64 # Change rank
    
    converter_svd = RSULFTransformerConverter(**config)
    rs_layers_svd = converter_svd.convert_model(original_model)
    rs_layers_svd = rs_layers_svd.to(device)

    analyze_layer_fidelity(original_model, rs_layers_svd, tokenizer, prompt, device)
    
    rs_svd_text, rs_svd_time = rsulf_generate_text(rs_layers_svd, prompt)
    print(f"   [RS-ULF SVD]: {rs_svd_text}")
    print(f"   Time: {rs_svd_time:.4f}s")

    # Summary
    print("\n=== Summary ===")
    print(f"Prompt: {prompt}")
    print(f"1. Original:         {orig_text.strip()}")
    print(f"2. RS-ULF (r={full_rank_r}):   {rs_text.strip()}")
    print(f"3. RS-ULF (r=64):    {rs_svd_text.strip()}")

if __name__ == "__main__":
    try:
        test_gpt2_conversion()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

