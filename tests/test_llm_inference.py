import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import torch
from reality_stone._rust import PyRSULFLayer
from reality_stone.models.transformer_converter import RSULFTransformerConverter, build_rsulf_causal_lm


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
    r = max(128, d_model // 8)
    converter = RSULFTransformerConverter(
        d_model=d_model,
        r=r,
        eta=0.005,
        alpha=0.01,
        beta=0.005,
        gamma=0.95,
        seq_len=64,
        window=4,
        verbose=True,
    )
    rs_lm = build_rsulf_causal_lm(model, converter)
    stats = rs_lm.rsulf.stats
    if stats is not None and stats.converted == 0:
        raise RuntimeError(f"No layers converted. Errors: {stats.errors if stats else None}")
    ratio = (stats.original_params / max(stats.compressed_params, 1)) if stats is not None else 0.0
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        generated = rs_lm.generate(input_ids, max_new_tokens=20)
    
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "output": output_text,
        "compression_ratio": ratio,
        "converted_layers": stats.converted if stats is not None else 0,
        "total_layers": stats.total_layers if stats is not None else 0,
        "original_params": stats.original_params if stats is not None else 0,
        "compressed_params": stats.compressed_params if stats is not None else 0,
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

