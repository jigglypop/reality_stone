import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time
import warnings
warnings.filterwarnings("ignore")

try:
    import reality_stone as rs
    print("Reality Stone loaded")
    REALITY_STONE_AVAILABLE = True
except ImportError as e:
    print(f"Reality Stone load failed: {e}")
    REALITY_STONE_AVAILABLE = False

class RealityStoneRiemannCompressor:
    def __init__(self, compression_level="riemann_ultra", curvature=1.0):
        self.compression_level = compression_level
        self.curvature = curvature
        if compression_level == "riemann_ultra":
            self.coeff_ratio = 0.08
            self.poincare_scale = 0.75
        elif compression_level == "riemann_maximum":
            self.coeff_ratio = 0.05
            self.poincare_scale = 0.70
        elif compression_level == "riemann_extreme":
            self.coeff_ratio = 0.12
            self.poincare_scale = 0.80
    
    def riemann_compress(self, matrix):
        if not REALITY_STONE_AVAILABLE:
            raise RuntimeError("Reality Stone required")
        original_shape = matrix.shape
        matrix = matrix.float()
        matrix_flat = matrix.flatten()
        max_val = torch.max(torch.abs(matrix_flat))
        if max_val > 0:
            normalized = matrix_flat / max_val
            poincare_vals = torch.tanh(normalized * 1.5) * self.poincare_scale
        else:
            poincare_vals = matrix_flat
            max_val = torch.tensor(1.0, device=matrix.device)
        n = len(poincare_vals)
        if n % 2 == 1:
            poincare_vals = torch.cat([poincare_vals, torch.zeros(1, device=matrix.device)])
            n += 1
        real_part = poincare_vals[::2]
        imag_part = poincare_vals[1::2]
        fft_result = torch.fft.fft(torch.complex(real_part, imag_part))
        magnitudes = torch.abs(fft_result)
        energies = magnitudes ** 2
        total_energy = torch.sum(energies)
        sorted_energies, sorted_indices = torch.sort(energies, descending=True)
        cumsum_energies = torch.cumsum(sorted_energies, dim=0)
        energy_ratios = cumsum_energies / total_energy
        energy_threshold = 0.02
        keep_count = torch.sum(energy_ratios < (1 - energy_threshold)).item() + 1
        min_coeffs = max(int(len(energies) * self.coeff_ratio), 2)
        max_coeffs = int(len(energies) * 0.5)
        keep_count = max(min_coeffs, min(keep_count, max_coeffs))
        important_indices = sorted_indices[:keep_count]
        important_coeffs = fft_result[important_indices]
        compression_ratio = len(important_coeffs) / len(energies)
        return {
            'coeffs': important_coeffs,
            'indices': important_indices,
            'original_length': len(real_part),
            'original_shape': original_shape,
            'scale': max_val,
            'compression_ratio': compression_ratio,
            'energy_kept': energy_ratios[keep_count-1].item() if keep_count > 0 else 1.0
        }
    
    def riemann_decompress(self, compressed):
        if not REALITY_STONE_AVAILABLE:
            raise RuntimeError("Reality Stone required")
        coeffs = compressed['coeffs']
        indices = compressed['indices']
        original_length = compressed['original_length']
        original_shape = compressed['original_shape']
        scale = compressed['scale']
        full_fft = torch.zeros(original_length, dtype=torch.complex64, device=coeffs.device)
        full_fft[indices] = coeffs
        restored_complex = torch.fft.ifft(full_fft)
        real_parts = restored_complex.real
        imag_parts = restored_complex.imag
        restored_flat = torch.stack([real_parts, imag_parts], dim=1).flatten()
        total_size = torch.prod(torch.tensor(original_shape)).item()
        if len(restored_flat) > total_size:
            restored_flat = restored_flat[:total_size]
        elif len(restored_flat) < total_size:
            padding = torch.zeros(total_size - len(restored_flat), device=restored_flat.device)
            restored_flat = torch.cat([restored_flat, padding])
        restored_matrix = restored_flat.view(original_shape) * scale
        return restored_matrix

class RealityStoneCompressedMLP(nn.Module):
    def __init__(self, original_mlp, compression_level="riemann_ultra", curvature=1.0):
        super().__init__()
        self.compression_level = compression_level
        self.curvature = curvature
        self.compressor = RealityStoneRiemannCompressor(compression_level, curvature)
        self.c_fc_compressed = self.compressor.riemann_compress(original_mlp.c_fc.weight.data)
        self.c_proj_compressed = self.compressor.riemann_compress(original_mlp.c_proj.weight.data)
        if original_mlp.c_fc.bias is not None:
            self.c_fc_bias = nn.Parameter(original_mlp.c_fc.bias.data.clone())
        else:
            self.register_parameter('c_fc_bias', None)
        if original_mlp.c_proj.bias is not None:
            self.c_proj_bias = nn.Parameter(original_mlp.c_proj.bias.data.clone())
        else:
            self.register_parameter('c_proj_bias', None)
        self.activation = nn.GELU()
    
    def forward(self, x):
        c_fc_weight = self.compressor.riemann_decompress(self.c_fc_compressed)
        h = F.linear(x, c_fc_weight.T, self.c_fc_bias)
        h = self.activation(h)
        c_proj_weight = self.compressor.riemann_decompress(self.c_proj_compressed)
        output = F.linear(h, c_proj_weight.T, self.c_proj_bias)
        return output

def apply_reality_stone_compression(model, compression_level="riemann_ultra", curvature=1.0, target_layers=None):
    if not REALITY_STONE_AVAILABLE:
        raise RuntimeError("Reality Stone required")
    total_layers = len(model.transformer.h)
    original_params = sum(p.numel() for p in model.parameters())
    if target_layers is None:
        target_layers = list(range(max(0, total_layers - 3), total_layers))
    compressed_count = 0
    for layer_idx in target_layers:
        if layer_idx < len(model.transformer.h):
            try:
                original_mlp = model.transformer.h[layer_idx].mlp
                reality_stone_mlp = RealityStoneCompressedMLP(original_mlp, compression_level, curvature)
                model.transformer.h[layer_idx].mlp = reality_stone_mlp
                compressed_count += 1
            except Exception as e:
                print(f"Layer {layer_idx} compression failed: {e}")
    final_params = sum(p.numel() for p in model.parameters())
    total_compression = final_params / original_params
    memory_saved = (1 - total_compression) * 100
    print(f"Compressed layers: {compressed_count}")
    print(f"Parameters: {original_params:,} -> {final_params:,}")
    print(f"Compression ratio: {total_compression:.4f}")
    print(f"Memory saved: {memory_saved:.1f}%")
    return model, total_compression

def reality_stone_korean_test(model, tokenizer, test_name=""):
    test_prompts = ["한국의", "안녕", "김치", "서울"]
    scores = []
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 4, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_longer = len(generated) > len(prompt)
            is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in generated[len(prompt):])
            score = (is_longer + is_korean) / 2
            scores.append(score)
            print(f"'{prompt}' -> '{generated}'")
        except Exception as e:
            print(f"'{prompt}' -> error: {str(e)[:50]}")
            scores.append(0)
    quality = sum(scores) / len(scores) if scores else 0
    return quality, 0

def main():
    if not REALITY_STONE_AVAILABLE:
        print("Reality Stone required")
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "skt/kogpt2-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Model load failed: {e}")
        return
    original_quality, _ = reality_stone_korean_test(model, tokenizer, "original")
    reality_stone_configs = [
        {"level": "riemann_extreme", "curvature": 0.5, "layers": [11], "name": "1 layer"},
        {"level": "riemann_ultra", "curvature": 1.0, "layers": [10, 11], "name": "2 layers"},
        {"level": "riemann_maximum", "curvature": 1.5, "layers": [9, 10, 11], "name": "3 layers"},
        {"level": "riemann_ultra", "curvature": 2.0, "layers": [8, 9, 10, 11], "name": "4 layers"},
    ]
    for config in reality_stone_configs:
        print(f"\nTesting {config['name']}")
        try:
            test_model = copy.deepcopy(model)
            compressed_model, compression_ratio = apply_reality_stone_compression(test_model, compression_level=config['level'], curvature=config['curvature'], target_layers=config['layers'])
            compressed_quality, _ = reality_stone_korean_test(compressed_model, tokenizer, config['name'])
            quality_retention = compressed_quality / original_quality if original_quality > 0 else compressed_quality
            memory_saved = (1 - compression_ratio) * 100
            print(f"Memory saved: {memory_saved:.1f}%")
            print(f"Quality retention: {quality_retention:.1%}")
        except Exception as e:
            print(f"Compression failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()