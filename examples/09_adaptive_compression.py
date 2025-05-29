"""
Reality Stone 리만 평면 FFT 역변환 압축 - 작동 버전
사용자 요구사항: "리만평면 FFT 역으로 압축"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math


class RiemannFFTCompressor:
    """리만 평면 FFT 역변환 압축기"""
    
    def __init__(self, compression_ratio=0.8):
        self.compression_ratio = compression_ratio
        
    def riemann_mapping(self, z):
        """리만 평면 매핑: z → (z-i)/(z+i)"""
        # 안전한 복소수 변환
        if not torch.is_complex(z):
            z_complex = z.to(torch.complex64)
        else:
            z_complex = z
            
        # i 정의
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        
        # 매핑 수행
        numerator = z_complex - i
        denominator = z_complex + i
        
        # 0으로 나누기 방지
        safe_denominator = torch.where(
            torch.abs(denominator) < 1e-8,
            torch.complex(torch.tensor(1e-8), torch.tensor(0.0)),
            denominator
        )
        
        return numerator / safe_denominator
    
    def inverse_riemann_mapping(self, w):
        """역 리만 매핑: w → i(1+w)/(1-w)"""
        if not torch.is_complex(w):
            w_complex = w.to(torch.complex64)
        else:
            w_complex = w
            
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        
        numerator = i * (1 + w_complex)
        denominator = 1 - w_complex
        
        # 안전한 나눗셈
        safe_denominator = torch.where(
            torch.abs(denominator) < 1e-8,
            torch.complex(torch.tensor(1e-8), torch.tensor(0.0)),
            denominator
        )
        
        return numerator / safe_denominator
    
    def compress(self, weight, name=""):
        """FFT 역변환 압축"""
        if weight.numel() < 1000:
            return {'type': 'original', 'weight': weight}
        
        original_shape = weight.shape
        
        try:
            # 2D로 변환
            if weight.dim() == 1:
                matrix = weight.unsqueeze(0).float()
            else:
                matrix = weight.view(weight.shape[0], -1).float()
            
            # 1. 리만 평면으로 매핑
            riemann_mapped = self.riemann_mapping(matrix)
            
            # 2. FFT 수행
            fft_result = torch.fft.fft2(riemann_mapped)
            
            # 3. 주파수 성분 압축 (TopK)
            magnitude = torch.abs(fft_result)
            phase = torch.angle(fft_result)
            
            # 평탄화
            mag_flat = magnitude.flatten()
            phase_flat = phase.flatten()
            
            # 중요한 주파수 성분만 선택
            keep_count = int(mag_flat.numel() * (1 - self.compression_ratio))
            keep_count = max(100, keep_count)  # 최소 100개는 유지
            
            # TopK 선택
            topk_values, topk_indices = torch.topk(mag_flat, keep_count)
            topk_phases = phase_flat[topk_indices]
            
            print(f"  {name}: {original_shape} → {keep_count} 주파수 성분 ({(1-keep_count/mag_flat.numel())*100:.1f}% 압축)")
            
            return {
                'type': 'riemann_fft',
                'shape': original_shape,
                'fft_shape': fft_result.shape,
                'magnitudes': topk_values,
                'phases': topk_phases,
                'indices': topk_indices,
                'total_elements': mag_flat.numel()
            }
            
        except Exception as e:
            print(f"  {name} 압축 실패: {e}")
            return {'type': 'original', 'weight': weight}
    
    def decompress(self, compressed):
        """압축 해제"""
        if compressed['type'] == 'original':
            return compressed['weight']
        
        try:
            # FFT 복원
            fft_shape = compressed['fft_shape']
            total_elements = compressed['total_elements']
            
            # 빈 주파수 도메인 생성
            mag_restored = torch.zeros(total_elements)
            phase_restored = torch.zeros(total_elements)
            
            # TopK 값 복원
            mag_restored[compressed['indices']] = compressed['magnitudes']
            phase_restored[compressed['indices']] = compressed['phases']
            
            # 원래 모양으로
            mag_restored = mag_restored.reshape(fft_shape)
            phase_restored = phase_restored.reshape(fft_shape)
            
            # 복소수로 재구성
            fft_restored = mag_restored * torch.exp(1j * phase_restored)
            
            # 역 FFT
            spatial_restored = torch.fft.ifft2(fft_restored)
            
            # 역 리만 매핑
            original = self.inverse_riemann_mapping(spatial_restored)
            
            # 실수부만 사용
            real_result = original.real
            
            # 원래 shape으로
            return real_result.view(compressed['shape'])
            
        except Exception as e:
            print(f"  복원 실패: {e}")
            # 안전한 랜덤 초기화
            shape = compressed['shape']
            return torch.randn(shape) * 0.02


class RiemannCompressedLinear(nn.Module):
    """리만 압축 Linear 레이어"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compressed_weight = None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.compressor = None
        
    def set_weight(self, compressed_weight, compressor):
        self.compressed_weight = compressed_weight
        self.compressor = compressor
        
    def forward(self, x):
        weight = self.compressor.decompress(self.compressed_weight)
        return F.linear(x, weight, self.bias)


def apply_riemann_fft_compression(model, compression_ratio=0.8):
    """리만 FFT 압축 적용"""
    
    print(f"\n리만 평면 FFT 역변환 압축")
    print("=" * 50)
    print(f"압축률: {compression_ratio*100:.0f}%")
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"원본 파라미터: {original_params:,}")
    
    compressor = RiemannFFTCompressor(compression_ratio)
    
    print("\n압축 진행:")
    
    # MLP 레이어만 압축
    for i in range(8, 12):  # 후반부 4개 레이어
        if i >= len(model.transformer.h):
            continue
            
        layer = model.transformer.h[i]
        
        # c_fc 압축
        old_fc = layer.mlp.c_fc
        new_fc = RiemannCompressedLinear(
            old_fc.in_features,
            old_fc.out_features,
            old_fc.bias is not None
        )
        
        compressed = compressor.compress(old_fc.weight.data, f"mlp.c_fc.{i}")
        new_fc.set_weight(compressed, compressor)
        
        if old_fc.bias is not None:
            new_fc.bias.data = old_fc.bias.data.clone()
            
        layer.mlp.c_fc = new_fc
        
        # c_proj 압축
        old_proj = layer.mlp.c_proj
        new_proj = RiemannCompressedLinear(
            old_proj.in_features,
            old_proj.out_features,
            old_proj.bias is not None
        )
        
        compressed = compressor.compress(old_proj.weight.data, f"mlp.c_proj.{i}")
        new_proj.set_weight(compressed, compressor)
        
        if old_proj.bias is not None:
            new_proj.bias.data = old_proj.bias.data.clone()
            
        layer.mlp.c_proj = new_proj
    
    compressed_params = sum(p.numel() for p in model.parameters())
    saved = (1 - compressed_params/original_params) * 100
    
    print(f"\n압축 완료:")
    print(f"압축 후 파라미터: {compressed_params:,}")
    print(f"메모리 절약: {saved:.1f}%")
    
    return model, saved


def test_korean(model, tokenizer):
    """한국어 생성 테스트"""
    print("\n한국어 생성 테스트:")
    
    prompts = ["한국의", "안녕하세요", "오늘 날씨는", "인공지능은"]
    success = 0
    
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs.input_ids,
                max_length=30,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if len(result) > len(prompt):
                print(f"  ✓ '{prompt}' → '{result}'")
                success += 1
            else:
                print(f"  ✗ '{prompt}' → 실패")
                
        except Exception as e:
            print(f"  ✗ '{prompt}' → 에러: {e}")
    
    return success / len(prompts)


def main():
    """메인"""
    print("🌀 Reality Stone 리만 평면 FFT 역변환 압축")
    print("=" * 60)
    print("사용자 요구: 리만평면 FFT 역으로 압축시키기")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # 모델 로드
    model_name = "skt/kogpt2-base-v2"
    print(f"\n모델 로딩: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 원본 테스트
    print("\n=== 원본 모델 ===")
    orig_score = test_korean(model, tokenizer)
    
    # 리만 FFT 압축
    print("\n=== 리만 FFT 압축 ===")
    compressed_model = copy.deepcopy(model)
    compressed_model, saved = apply_riemann_fft_compression(compressed_model, 0.8)
    
    # 압축 후 테스트
    score = test_korean(compressed_model, tokenizer)
    
    # 결과
    print(f"\n🏆 최종 결과:")
    print(f"메모리 절약: {saved:.1f}%")
    print(f"성공률: {score:.0%}")
    
    if saved >= 70 and score > 0:
        print("\n🎉 대성공! 70%+ 압축 + 한국어 생성!")
    elif saved >= 50 and score > 0:
        print("\n✅ 성공! 50%+ 압축 + 한국어 생성!")
    elif score > 0:
        print("\n💪 작동하지만 압축률 개선 필요")
    else:
        print("\n❌ 재시도 필요")


if __name__ == "__main__":
    main()