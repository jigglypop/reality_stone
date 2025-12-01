import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from reality_stone.models.transformer_converter import RSULFTransformerConverter, RSULFModel
import time

def test_gpt2_conversion():
    print("=== [Reality Stone] GPT-2 Conversion Test Start ===")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Device: cpu (CUDA not available)")
        # Raise warning but continue, as user asked for CUDA
        print("WARNING: User requested CUDA but it is not available. Running on CPU.")

    # 1. Load Original GPT-2
    print("\n1. Loading Original GPT-2...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Fix: Set pad_token to eos_token to avoid warnings and generation loops
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

    # 3. Convert to RS-ULF (Fast Mode: No SVD, Pure Structure Mapping)
    print("\n2. Converting to RS-ULF (Fast Mode - Structure Only)...")
    # Config for GPT-2 with corrected Geodesic Flow implementation
    # 
    # CRITICAL: Without Attention, we need Graph Diffusion to provide token interaction.
    # beta controls the Graph Laplacian term which replaces Attention's role.
    # 
    # eta=1.0: FFN output scale (1.0 = Transformer equivalent for FFN part)
    # alpha: Mean diffusion (regularization)
    # beta: Graph Laplacian diffusion (REPLACES ATTENTION - must be non-zero!)
    config = {
        "rank": 128,           # Ignored in fast_mode
        "lr": 0.8,             # eta: FFN output scale (Reduced to balance with beta)
        "alpha": 0.05,         # Mean diffusion (Increased for stability)
        "beta": 0.8,           # Graph diffusion (Increased to 0.8 to match Attention strength)
        "gamma": 0.99,         # Memory decay (for Bellman accumulator)
        "fast_mode": True,     # Bypass SVD for structure test
        "verbose": False
    }
    
    converter = RSULFTransformerConverter(config)
    # Note: RSULFModel replaces the internal layers but we need a wrapper for the full generation loop
    # The converter returns an RSULFModel which mimics the layer stack.
    # We need to hook it into the generation loop.
    
    rs_layers = converter.convert_model(original_model, device=device)
    
    print("   Conversion Complete.")
    
    # 4. Simple Generation Loop for RS-ULF
    # Since RSULFModel is just the stack of layers, we need to manually run embeddings -> layers -> lm_head
    print("   Generating with RS-ULF (Fast Mode)...")
    
    def rsulf_generate_text(rs_model_stack, text_prompt, max_tokens=30):
        curr_ids = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
        
        # Get embeddings and head from original model (they are not converted, only layers are)
        wte = original_model.transformer.wte
        wpe = original_model.transformer.wpe
        lm_head = original_model.lm_head
        ln_f = original_model.transformer.ln_f
        
        # Get LayerNorms from original model (GPT-2 uses Pre-LN)
        # Each layer has ln_1 (before attn) and ln_2 (before mlp)
        layer_norms = [(layer.ln_1, layer.ln_2) for layer in original_model.transformer.h]
        
        generated = curr_ids
        
        start_gen = time.time()
        for _ in range(max_tokens):
            seq_len = generated.size(1)
            pos = torch.arange(seq_len, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # 1. Embeddings
                tok_emb = wte(generated)
                pos_emb = wpe(pos)
                x = tok_emb + pos_emb
                
                # 2. Run RS-ULF Stack with LayerNorms
                # GPT-2 structure: x' = x + Attn(LN1(x)) + MLP(LN2(x'))
                # RS-ULF replaces Attn with Graph Diffusion and MLP with Geodesic FFN
                # We apply LN2 before FFN (the MLP norm)
                h = x
                for i, rs_layer in enumerate(rs_model_stack.layers):
                    ln_1, ln_2 = layer_norms[i]
                    # Apply LayerNorm before FFN (like GPT-2's ln_2 before MLP)
                    h_normed = ln_2(h)
                    
                    if hasattr(rs_layer, 'inner'):
                        # Rust Wrapper (CPU/Numpy)
                        # RS-ULF forward expects numpy, returns numpy
                        h_np = h_normed.cpu().float().numpy()
                        if h_np.ndim == 3:
                            b, l, d = h_np.shape
                            h_flat = h_np.reshape(b * l, d)
                        else:
                            h_flat = h_np
                        out_np, _ = rs_layer.inner.forward(h_flat, None)
                        if h_np.ndim == 3:
                            out_np = out_np.reshape(b, l, d)
                        h_out = torch.from_numpy(out_np).to(device)
                    else:
                        # PyTorch/CUDA Implementation (Tensor)
                        # rs_layer is nn.Module (RSULFLayerCUDA)
                        # It expects Tensor input (B, D) or (B, L, D) if it handles it
                        # RSULFLayerCUDA.forward handles (B, D) primarily
                        b, l, d = h_normed.shape
                        h_flat = h_normed.reshape(b * l, d)
                        h_out_flat, _ = rs_layer(h_flat, None)
                        h_out = h_out_flat.reshape(b, l, d)

                    # Residual connection
                    h = h + (h_out - h_normed)  # Add the delta from RS-ULF
                
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
    print(f"   [RS-ULF Fast]: {rs_text}")
    print(f"   Time: {rs_time:.4f}s")

    # 5. Convert to RS-ULF (Compressed Mode: SVD Rank 128)
    print("\n3. Converting to RS-ULF (Compressed Mode - SVD Rank 128)...")
    config["fast_mode"] = False
    config["rank"] = 128 # GPT-2 d_model is 768, so 128 is ~6x compression on linear maps
    
    converter_svd = RSULFTransformerConverter(config)
    rs_layers_svd = converter_svd.convert_model(original_model, device=device)
    
    rs_svd_text, rs_svd_time = rsulf_generate_text(rs_layers_svd, prompt)
    
    print(f"   [RS-ULF SVD]: {rs_svd_text}")
    print(f"   Time: {rs_svd_time:.4f}s")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Prompt: {prompt}")
    print(f"1. Original:    {orig_text.strip()}")
    print(f"2. RS-ULF Fast: {rs_text.strip()}")
    print(f"3. RS-ULF SVD:  {rs_svd_text.strip()}")
    
    if orig_text.strip() == rs_text.strip():
        print("\n[SUCCESS] Fast Mode perfectly matched Original (or extremely close).")
    else:
        print("\n[INFO] Fast Mode output differs. Check eta/alpha calibration.")

if __name__ == "__main__":
    try:
        test_gpt2_conversion()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

