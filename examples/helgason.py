import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import math
from tqdm import tqdm

try:
    import reality_stone as rs
    print("Reality Stone loaded")
    REALITY_STONE_AVAILABLE = True
except ImportError as e:
    print(f"Reality Stone load failed: {e}")
    REALITY_STONE_AVAILABLE = False

class RealityStoneRiemannCompressor:
    def __init__(self, compression_level="medium", curvature=1.0):
        self.compression_level = compression_level
        self.curvature = curvature
        if compression_level == "extreme":
            self.coeff_ratio = 0.15  # 15% 보존
            self.poincare_scale = 0.85
        elif compression_level == "medium":
            self.coeff_ratio = 0.25  # 25% 보존
            self.poincare_scale = 0.9
    
    def riemann_compress(self, matrix):
        if not REALITY_STONE_AVAILABLE:
            raise RuntimeError("Reality Stone required")
        original_shape = matrix.shape
        matrix = matrix.float()
        matrix_flat = matrix.flatten()
        max_val = torch.max(torch.abs(matrix_flat))
        if max_val > 0:
            normalized = matrix_flat / max_val
            poincare_vals = torch.tanh(normalized) * self.poincare_scale
        else:
            poincare_vals = matrix_flat
            max_val = torch.tensor(1.0, device=matrix.device)
        n = len(poincare_vals)
        pad_n = 2 ** int(math.ceil(math.log2(n)))
        if n < pad_n:
            poincare_vals = F.pad(poincare_vals, (0, pad_n - n))
        fft_result = torch.fft.rfft(poincare_vals)
        magnitudes = torch.abs(fft_result)
        energies = magnitudes ** 2
        keep_count = max(10, int(len(fft_result) * self.coeff_ratio))
        topk_values, topk_indices = torch.topk(energies, keep_count)
        important_coeffs = fft_result[topk_indices]
        return {
            'coeffs': important_coeffs,
            'indices': topk_indices,
            'pad_n': pad_n,
            'original_n': n,
            'original_shape': original_shape,
            'scale': max_val,
            'fft_len': len(fft_result)
        }
    
    def riemann_decompress(self, compressed):
        if not REALITY_STONE_AVAILABLE:
            raise RuntimeError("Reality Stone required")
        coeffs = compressed['coeffs']
        indices = compressed['indices']
        pad_n = compressed['pad_n']
        original_n = compressed['original_n']
        original_shape = compressed['original_shape']
        scale = compressed['scale']
        fft_len = compressed['fft_len']
        full_fft = torch.zeros(fft_len, dtype=torch.complex64, device=coeffs.device)
        full_fft[indices] = coeffs
        restored = torch.fft.irfft(full_fft, n=pad_n)
        restored = restored[:original_n]
        total_size = torch.prod(torch.tensor(original_shape)).item()
        if len(restored) != total_size:
            if len(restored) > total_size:
                restored = restored[:total_size]
            else:
                restored = F.pad(restored, (0, total_size - len(restored)))
        restored_matrix = restored.view(original_shape) * scale
        return restored_matrix

class CompressedAttention(nn.Module):
    def __init__(self, original_attn, compression_level="medium", curvature=1.0):
        super().__init__()
        self.compressor = RealityStoneRiemannCompressor(compression_level, curvature)
        self.n_head = original_attn.n_head
        self.split_size = original_attn.split_size
        self.scale = original_attn.scale
        start_time = time.time()
        self.c_attn_compressed = self.compressor.riemann_compress(original_attn.c_attn.weight.data)
        self.c_proj_compressed = self.compressor.riemann_compress(original_attn.c_proj.weight.data)
        compress_time = time.time() - start_time
        if hasattr(original_attn.c_attn, 'bias') and original_attn.c_attn.bias is not None:
            self.c_attn_bias = nn.Parameter(original_attn.c_attn.bias.data.clone())
        else:
            self.register_parameter('c_attn_bias', None)
        if hasattr(original_attn.c_proj, 'bias') and original_attn.c_proj.bias is not None:
            self.c_proj_bias = nn.Parameter(original_attn.c_proj.bias.data.clone())
        else:
            self.register_parameter('c_proj_bias', None)
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout
        original_params = original_attn.c_attn.weight.numel() + original_attn.c_proj.weight.numel()
        compressed_params = len(self.c_attn_compressed['coeffs']) * 2 + len(self.c_proj_compressed['coeffs']) * 2
        print(f"Attention: {original_params:,} -> {compressed_params:,} ({compressed_params/original_params:.1%}) in {compress_time:.2f}s")
    
    def forward(self, x):
        c_attn_weight = self.compressor.riemann_decompress(self.c_attn_compressed)
        c_proj_weight = self.compressor.riemann_decompress(self.c_proj_compressed)
        query_key_value = F.linear(x, c_attn_weight.T, self.c_attn_bias)
        query, key, value = query_key_value.split(self.split_size, dim=2)
        query = self._split_heads(query, self.n_head, query.size(-1) // self.n_head)
        key = self._split_heads(key, self.n_head, key.size(-1) // self.n_head)
        value = self._split_heads(value, self.n_head, value.size(-1) // self.n_head)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(value.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.n_head, value.size(-1))
        attn_output = F.linear(attn_output, c_proj_weight.T, self.c_proj_bias)
        attn_output = self.resid_dropout(attn_output)
        return attn_output
    
    def _split_heads(self, tensor, num_heads, head_size):
        new_shape = tensor.size()[:-1] + (num_heads, head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def _merge_heads(self, tensor, num_heads, head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_size,)
        return tensor.view(*new_shape)

class RealityStoneCompressedMLP(nn.Module):
    def __init__(self, original_mlp, compression_level="medium", curvature=1.0):
        super().__init__()
        self.compressor = RealityStoneRiemannCompressor(compression_level, curvature)
        start_time = time.time()
        self.c_fc_compressed = self.compressor.riemann_compress(original_mlp.c_fc.weight.data)
        self.c_proj_compressed = self.compressor.riemann_compress(original_mlp.c_proj.weight.data)
        compress_time = time.time() - start_time
        if original_mlp.c_fc.bias is not None:
            self.c_fc_bias = nn.Parameter(original_mlp.c_fc.bias.data.clone())
        else:
            self.register_parameter('c_fc_bias', None)
        if original_mlp.c_proj.bias is not None:
            self.c_proj_bias = nn.Parameter(original_mlp.c_proj.bias.data.clone())
        else:
            self.register_parameter('c_proj_bias', None)
        self.activation = nn.GELU()
        original_params = original_mlp.c_fc.weight.numel() + original_mlp.c_proj.weight.numel()
        compressed_params = len(self.c_fc_compressed['coeffs']) * 2 + len(self.c_proj_compressed['coeffs']) * 2
        print(f"MLP: {original_params:,} -> {compressed_params:,} ({compressed_params/original_params:.1%}) in {compress_time:.2f}s")
    
    def forward(self, x):
        c_fc_weight = self.compressor.riemann_decompress(self.c_fc_compressed)
        h = F.linear(x, c_fc_weight.T, self.c_fc_bias)
        h = self.activation(h)
        c_proj_weight = self.compressor.riemann_decompress(self.c_proj_compressed)
        output = F.linear(h, c_proj_weight.T, self.c_proj_bias)
        return output

def apply_reality_stone_compression(model, compression_level="medium", curvature=1.0, num_layers=4, compress_attention=True):
    if not REALITY_STONE_AVAILABLE:
        raise RuntimeError("Reality Stone required")
    total_layers = len(model.transformer.h)
    target_layers = list(range(total_layers - num_layers, total_layers))
    compressed_count = 0
    print(f"Compressing {num_layers} layers with {compression_level} level")
    for layer_idx in tqdm(target_layers, desc="Compressing layers"):
        if layer_idx < len(model.transformer.h):
            try:
                original_mlp = model.transformer.h[layer_idx].mlp
                reality_stone_mlp = RealityStoneCompressedMLP(original_mlp, compression_level, curvature)
                model.transformer.h[layer_idx].mlp = reality_stone_mlp
                if compress_attention:
                    original_attn = model.transformer.h[layer_idx].attn
                    compressed_attn = CompressedAttention(original_attn, compression_level, curvature)
                    model.transformer.h[layer_idx].attn = compressed_attn
                compressed_count += 1
            except Exception as e:
                print(f"Layer {layer_idx} failed: {e}")
    return model, compressed_count

def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * 4
    total_gb = total_bytes / (1024**3)
    return total_params, total_gb

def test_korean_quality(model, tokenizer):
    test_cases = [
        ("한국의", ["수도", "문화", "역사", "전통", "음식"]),
        ("안녕", ["하세요", "하십니까", "히", "하신가요"]),
        ("김치는", ["한국", "발효", "맛있", "전통"]),
        ("서울은", ["수도", "도시", "대한민국", "한국"])
    ]
    total_score = 0
    for prompt, expected_words in tqdm(test_cases, desc="Testing"):
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_length=20, 
                                       temperature=0.8, do_sample=True, 
                                       pad_token_id=tokenizer.eos_token_id,
                                       repetition_penalty=1.2)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            score = 0
            if len(generated) > len(prompt):
                score += 0.5
            generated_suffix = generated[len(prompt):].lower()
            if any(word in generated_suffix for word in expected_words):
                score += 0.5
            total_score += score
            print(f"'{prompt}' -> '{generated}'")
        except Exception as e:
            print(f"'{prompt}' -> error: {e}")
    return total_score / len(test_cases)

def main():
    if not REALITY_STONE_AVAILABLE:
        print("Reality Stone required")
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "skt/kogpt2-base-v2"
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Model load failed: {e}")
        return
    original_params, original_gb = calculate_model_size(model)
    print(f"\nOriginal: {original_params:,} params, {original_gb:.3f} GB")
    original_quality = test_korean_quality(model, tokenizer)
    print(f"Original quality: {original_quality:.1%}")
    configs = [
        {"name": "Medium (MLP only)", "level": "medium", "curvature": 1.0, "layers": 4, "attn": False},
        {"name": "Medium (MLP+Attention)", "level": "medium", "curvature": 1.0, "layers": 4, "attn": True},
        {"name": "Extreme (MLP+Attention)", "level": "extreme", "curvature": 1.5, "layers": 6, "attn": True}
    ]
    for config in configs:
        print(f"\n{config['name']} ({config['layers']} layers)")
        print("-" * 50)
        test_model = copy.deepcopy(model)
        start_time = time.time()
        compressed_model, count = apply_reality_stone_compression(
            test_model, config['level'], config['curvature'], config['layers'], config['attn']
        )
        compress_time = time.time() - start_time
        compressed_params, compressed_gb = calculate_model_size(compressed_model)
        print(f"Compressed: {compressed_params:,} params, {compressed_gb:.3f} GB")
        print(f"Size: {original_gb:.3f} GB -> {compressed_gb:.3f} GB ({(1-compressed_gb/original_gb)*100:.1f}% saved)")
        print(f"Compression time: {compress_time:.2f}s")
        quality = test_korean_quality(compressed_model, tokenizer)
        print(f"Quality: {quality:.1%}")

if __name__ == "__main__":
    main()