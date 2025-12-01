import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
import torch
from tqdm import tqdm

from reality_stone._rust import PyRSULFLayer
from reality_stone.models.transformer_converter import RSULFTransformerConverter, RSULFStack


def test_llm_compression_inference():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    d_model = model.config.hidden_size
    num_layers = len(model.model.layers)
    
    r = max(128, d_model // 8)
    
    layer0 = model.model.layers[0]
    wq_shape = layer0.self_attn.q_proj.weight.shape
    wk_shape = layer0.self_attn.k_proj.weight.shape
    
    converter = RSULFTransformerConverter(
        d_model=wq_shape[0],
        r=r,
        eta=0.005,
        alpha=0.01,
        beta=0.005,
        gamma=0.95,
        seq_len=64,
        window=4,
        verbose=True,
    )
    
    rsulf_layers, stats = converter.convert_model(model)
    
    if stats.converted == 0:
        raise RuntimeError(f"No layers converted. Errors: {stats.errors}")
    
    if stats.errors:
        error_analysis = converter.analyze_errors()
        for err_type, layers in error_analysis["by_type"].items():
            print(f"Error '{err_type}': layers {layers[:3]}...")
    
    ratio = stats.original_params / max(stats.compressed_params, 1)
    
    rs_stack = RSULFStack(rsulf_layers)
    
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)
        
        x = embeds.squeeze(0).numpy().astype(np.float32)
        v_mem = None
        
        for i, layer in enumerate(rsulf_layers):
            x, v_mem = layer._layer.forward(x, v_mem)
        
        hidden = torch.from_numpy(x).unsqueeze(0).float()
        logits = model.lm_head(hidden)
        
        generated_ids = input_ids.tolist()[0]
        
        for step in range(20):
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated_ids.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
            
            next_embed = model.model.embed_tokens(torch.tensor([[next_token]]))
            x_next = next_embed.squeeze(0).numpy().astype(np.float32)
            
            for layer in rsulf_layers:
                x_next, v_mem = layer._layer.forward(x_next, v_mem)
            
            hidden = torch.from_numpy(x_next).unsqueeze(0).float()
            logits = model.lm_head(hidden)
    
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "output": output_text,
        "compression_ratio": ratio,
        "converted_layers": stats.converted,
        "total_layers": stats.total_layers,
        "original_params": stats.original_params,
        "compressed_params": stats.compressed_params,
    }


if __name__ == "__main__":
    result = test_llm_compression_inference()
    
    print("=" * 60)
    print("RS-ULF LLM COMPRESSION TEST")
    print("=" * 60)
    print(f"Prompt: {result['prompt']}")
    print(f"Output: {result['output']}")
    print(f"Compression: {result['compression_ratio']:.1f}x")
    print(f"Layers: {result['converted_layers']}/{result['total_layers']}")
    print(f"Params: {result['compressed_params']:,} / {result['original_params']:,}")
    print("=" * 60)

