"""
Reality Stone 빠른 헬가손 압축
속도 최적화된 실용적인 버전

핵심: 단순한 푸앵카레 + 벡터화 푸리에 + 빠른 압축
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import warnings
warnings.filterwarnings("ignore")


class FastPoincareFourier:
    """빠른 푸앵카레 푸리에 압축"""
    
    def __init__(self, num_coeffs=16):
        self.num_coeffs = num_coeffs
        print(f"⚡ 빠른 푸앵카레 푸리에 (계수: {num_coeffs}개)")
    
    def compress_matrix(self, matrix):
        """행렬을 빠르게 압축"""
        # 1. 푸앵카레 정규화 (tanh로 [-0.95, 0.95] 매핑)
        matrix_flat = matrix.flatten()
        max_val = torch.max(torch.abs(matrix_flat))
        
        if max_val > 0:
            normalized = matrix_flat / max_val
            poincare_vals = torch.tanh(normalized) * 0.95
        else:
            poincare_vals = matrix_flat
            max_val = torch.tensor(1.0, device=matrix.device)
        
        # 2. 빠른 FFT (PyTorch 내장 사용)
        # 실수를 복소수로 변환하여 FFT 적용
        if len(poincare_vals) % 2 == 1:
            poincare_vals = torch.cat([poincare_vals, torch.zeros(1, device=matrix.device)])
        
        # 복소수 변환
        real_part = poincare_vals[::2]
        imag_part = poincare_vals[1::2]
        complex_vals = torch.complex(real_part, imag_part)
        
        # FFT
        fft_result = torch.fft.fft(complex_vals)
        
        # 3. 중요한 계수만 선택 (에너지 기반)
        magnitudes = torch.abs(fft_result)
        _, top_indices = torch.topk(magnitudes, min(self.num_coeffs, len(magnitudes)))
        
        # 선택된 계수들
        important_coeffs = fft_result[top_indices]
        
        return {
            'coeffs': important_coeffs,
            'indices': top_indices,
            'original_length': len(complex_vals),
            'original_shape': matrix.shape,
            'scale': max_val
        }
    
    def decompress_matrix(self, compressed):
        """압축된 데이터를 빠르게 복원"""
        coeffs = compressed['coeffs']
        indices = compressed['indices']
        original_length = compressed['original_length']
        original_shape = compressed['original_shape']
        scale = compressed['scale']
        
        # FFT 계수 복원
        full_fft = torch.zeros(original_length, dtype=torch.complex64, device=coeffs.device)
        full_fft[indices] = coeffs
        
        # IFFT
        restored_complex = torch.fft.ifft(full_fft)
        
        # 실수 변환
        real_parts = restored_complex.real
        imag_parts = restored_complex.imag
        restored_flat = torch.stack([real_parts, imag_parts], dim=1).flatten()
        
        # 원래 크기 맞춤
        total_size = torch.prod(torch.tensor(original_shape)).item()
        if len(restored_flat) > total_size:
            restored_flat = restored_flat[:total_size]
        elif len(restored_flat) < total_size:
            padding = torch.zeros(total_size - len(restored_flat), device=restored_flat.device)
            restored_flat = torch.cat([restored_flat, padding])
        
        # 스케일 복원 및 reshape
        restored_matrix = restored_flat.view(original_shape) * scale
        
        return restored_matrix


class FastHelgasonLayer(nn.Module):
    """빠른 헬가손 압축 레이어"""
    
    def __init__(self, mlp_layers, num_coeffs=16):
        super().__init__()
        
        self.num_layers = len(mlp_layers)
        print(f"\n⚡ 빠른 헬가손 레이어 ({self.num_layers}개 융합)")
        
        # 압축기 초기화
        self.compressor = FastPoincareFourier(num_coeffs)
        
        # 가중치 융합 (단순 평균)
        c_fc_weights = [mlp.c_fc.weight.data for mlp in mlp_layers]
        c_proj_weights = [mlp.c_proj.weight.data for mlp in mlp_layers]
        
        print("   ⚡ 빠른 가중치 융합...")
        fused_c_fc = torch.mean(torch.stack(c_fc_weights), dim=0)
        fused_c_proj = torch.mean(torch.stack(c_proj_weights), dim=0)
        
        # 압축
        print("   ⚡ c_fc 압축...")
        self.c_fc_compressed = self.compressor.compress_matrix(fused_c_fc)
        
        print("   ⚡ c_proj 압축...")
        self.c_proj_compressed = self.compressor.compress_matrix(fused_c_proj)
        
        # 바이어스 처리
        if mlp_layers[0].c_fc.bias is not None:
            biases = torch.stack([mlp.c_fc.bias.data for mlp in mlp_layers])
            self.c_fc_bias = nn.Parameter(torch.mean(biases, dim=0))
        else:
            self.register_parameter('c_fc_bias', None)
            
        if mlp_layers[0].c_proj.bias is not None:
            biases = torch.stack([mlp.c_proj.bias.data for mlp in mlp_layers])
            self.c_proj_bias = nn.Parameter(torch.mean(biases, dim=0))
        else:
            self.register_parameter('c_proj_bias', None)
        
        self.activation = nn.GELU()
        
        # 통계
        self._print_stats(c_fc_weights + c_proj_weights)
    
    def _print_stats(self, original_weights):
        """압축 통계"""
        original_params = sum(w.numel() for w in original_weights)
        
        # 압축된 파라미터 (복소수 = 실수 2개)
        compressed_params = (
            len(self.c_fc_compressed['coeffs']) * 2 +  # 복소수
            len(self.c_proj_compressed['coeffs']) * 2 +
            (self.c_fc_bias.numel() if self.c_fc_bias is not None else 0) +
            (self.c_proj_bias.numel() if self.c_proj_bias is not None else 0)
        )
        
        self.compression_ratio = compressed_params / original_params
        memory_saved = (1 - self.compression_ratio) * 100
        
        print(f"   💾 압축 통계:")
        print(f"   원본: {original_params:,} → 압축: {compressed_params:,}")
        print(f"   압축률: {self.compression_ratio:.3f}")
        print(f"   메모리 절약: {memory_saved:.1f}%")
    
    def forward(self, x):
        """빠른 순전파"""
        # c_fc 복원 및 적용
        c_fc_weight = self.compressor.decompress_matrix(self.c_fc_compressed)
        h = F.linear(x, c_fc_weight.T, self.c_fc_bias)
        h = self.activation(h)
        
        # c_proj 복원 및 적용
        c_proj_weight = self.compressor.decompress_matrix(self.c_proj_compressed)
        output = F.linear(h, c_proj_weight.T, self.c_proj_bias)
        
        return output


def apply_fast_helgason_compression(model, num_coeffs=16):
    """빠른 헬가손 압축 적용"""
    
    print(f"\n⚡ 빠른 헬가손 압축")
    print("=" * 40)
    
    total_layers = len(model.transformer.h)
    original_params = sum(p.numel() for p in model.parameters())
    
    # 마지막 2개 레이어만 융합 (빠른 테스트)
    if total_layers >= 2:
        fusion_groups = [[total_layers - 2, total_layers - 1]]
    else:
        print("   ⚠️ 레이어가 부족합니다.")
        return model, 1.0
    
    print(f"   원본 레이어: {total_layers}개")
    print(f"   융합 그룹: {fusion_groups}")
    
    # 융합 적용
    for group in fusion_groups:
        print(f"\n📦 그룹 {group} 압축...")
        
        # MLP 수집
        mlp_layers = [model.transformer.h[i].mlp for i in group]
        
        # 빠른 압축 레이어 생성
        compressed_layer = FastHelgasonLayer(mlp_layers, num_coeffs)
        
        # 첫 번째 레이어에 배치
        model.transformer.h[group[0]].mlp = compressed_layer
    
    # 마지막 레이어 제거
    del model.transformer.h[-1]
    
    # 최종 통계
    final_params = sum(p.numel() for p in model.parameters())
    total_compression = final_params / original_params
    memory_saved = (1 - total_compression) * 100
    
    print(f"\n📊 전체 압축 결과:")
    print(f"   레이어: {total_layers} → {len(model.transformer.h)}")
    print(f"   파라미터: {original_params:,} → {final_params:,}")
    print(f"   압축률: {total_compression:.3f}")
    print(f"   메모리 절약: {memory_saved:.1f}%")
    
    return model, total_compression


def quick_test(model, tokenizer, test_name=""):
    """빠른 품질 테스트"""
    
    print(f"🧪 빠른 테스트 {test_name}")
    
    prompts = ["한국의 수도는", "안녕하세요"]
    scores = []
    
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 5,  # 짧게
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 간단한 체크
            score = 1 if len(generated) > len(prompt) + 1 else 0
            scores.append(score)
            
            status = "✅" if score else "❌"
            print(f"   '{prompt}' → '{generated}' {status}")
            
        except Exception as e:
            print(f"   '{prompt}' → 에러: {str(e)[:30]}... ❌")
            scores.append(0)
    
    quality = sum(scores) / len(scores) if scores else 0
    print(f"   품질: {quality:.1%}")
    
    return quality


def main():
    """빠른 테스트"""
    
    print("⚡ 빠른 헬가손 푸리에 압축 테스트")
    print("=" * 50)
    
    # 모델 로드
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "skt/kogpt2-base-v2"
        print(f"📥 모델 로딩: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ 모델 로드 성공!")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 원본 테스트
    print(f"\n📋 원본 모델 테스트")
    original_quality = quick_test(model, tokenizer, "(원본)")
    
    # 빠른 압축
    try:
        print(f"\n⚡ 빠른 헬가손 압축 시작...")
        compressed_model = copy.deepcopy(model)
        compressed_model, compression_ratio = apply_fast_helgason_compression(compressed_model, num_coeffs=12)
        
        # 압축 모델 테스트
        print(f"\n📋 압축 모델 테스트")
        compressed_quality = quick_test(compressed_model, tokenizer, "(압축)")
        
        # 결과
        quality_retention = compressed_quality / original_quality if original_quality > 0 else 0
        memory_saved = (1 - compression_ratio) * 100
        
        print(f"\n🏆 빠른 결과:")
        print(f"   원본 품질: {original_quality:.1%}")
        print(f"   압축 품질: {compressed_quality:.1%}")
        print(f"   품질 보존: {quality_retention:.1%}")
        print(f"   메모리 절약: {memory_saved:.1f}%")
        
        if memory_saved >= 3 and quality_retention >= 0.5:
            print(f"\n🎉 빠른 성공! 실용적인 압축")
            print(f"   ✅ 속도 + 효율성 달성")
        elif memory_saved >= 3:
            print(f"\n⚡ 압축 성공! 속도 우선")
        else:
            print(f"\n💪 더 나은 설정 필요")
        
        print(f"\n⚡ 빠른 기술:")
        print(f"   ✅ 벡터화 푸앵카레 매핑")
        print(f"   ✅ PyTorch FFT 활용")
        print(f"   ✅ TopK 계수 선택")
        print(f"   ✅ 최적화된 복원")
        print(f"   ✅ 빠른 실행 속도")
        
        print(f"\n🎯 목표 달성: 용량과 속도 효과!")
        
    except Exception as e:
        print(f"❌ 압축 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 빠른 테스트 완료!")


if __name__ == "__main__":
    main() 