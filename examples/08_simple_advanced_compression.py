"""
Reality Stone 간단한 고급 압축 테스트
SVD + FFT Hybrid 압축 기능만 검증

목표: 높은 압축률 + 정확도 보존 확인
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")


class SimpleHybridSuperLayer(nn.Module):
    """간단한 SVD + FFT Hybrid 압축 기술 기반 Super Layer"""
    
    def __init__(self, mlp_layers, layer_indices, svd_rank_ratio=0.5, fft_quality=0.95):
        super().__init__()
        
        self.layer_indices = layer_indices
        self.svd_rank_ratio = svd_rank_ratio
        self.fft_quality = fft_quality
        
        print(f"\n🔬 Simple Hybrid Super Layer (SVD + FFT)")
        print(f"   융합 레이어: {layer_indices}")
        print(f"   SVD rank ratio: {svd_rank_ratio}")
        print(f"   FFT 품질: {fft_quality:.1%}")
        
        # 가중치 수집
        all_c_fc_weights = [mlp.c_fc.weight.data.clone() for mlp in mlp_layers]
        all_c_proj_weights = [mlp.c_proj.weight.data.clone() for mlp in mlp_layers]
        
        # Hybrid 압축 적용
        self.c_fc_U, self.c_fc_S, self.c_fc_V = self._create_hybrid_compressed_layer(
            all_c_fc_weights, "c_fc"
        )
        
        self.c_proj_U, self.c_proj_S, self.c_proj_V = self._create_hybrid_compressed_layer(
            all_c_proj_weights, "c_proj"
        )
        
        # 바이어스 처리
        if mlp_layers[0].c_fc.bias is not None:
            all_c_fc_bias = torch.stack([mlp.c_fc.bias.data for mlp in mlp_layers])
            self.c_fc_bias = nn.Parameter(torch.mean(all_c_fc_bias, dim=0))
        else:
            self.register_parameter('c_fc_bias', None)
            
        if mlp_layers[0].c_proj.bias is not None:
            all_c_proj_bias = torch.stack([mlp.c_proj.bias.data for mlp in mlp_layers])
            self.c_proj_bias = nn.Parameter(torch.mean(all_c_proj_bias, dim=0))
        else:
            self.register_parameter('c_proj_bias', None)
        
        self.activation = nn.GELU()
        
        # 압축률 계산
        original_total = sum(w.numel() for w in all_c_fc_weights + all_c_proj_weights)
        compressed_total = (self.c_fc_U.numel() + self.c_fc_S.numel() + self.c_fc_V.numel() + 
                          self.c_proj_U.numel() + self.c_proj_S.numel() + self.c_proj_V.numel())
        
        self.compression_ratio = compressed_total / original_total
        
        print(f"   🎯 Hybrid 압축 완료:")
        print(f"   원본 파라미터: {original_total:,}")
        print(f"   압축 파라미터: {compressed_total:,}")
        print(f"   압축률: {self.compression_ratio:.3f} ({(1-self.compression_ratio)*100:.1f}% 절약)")
        
    def _create_hybrid_compressed_layer(self, weight_list, layer_type):
        """SVD + FFT 하이브리드 압축"""
        
        print(f"\n   🔬 {layer_type} Hybrid 압축 중...")
        
        # 1. FFT 기반 레이어 융합
        fft_layers = []
        for weight in weight_list:
            weight_fft = torch.fft.fft2(weight.float())
            fft_layers.append(weight_fft)
            
        fft_stack = torch.stack(fft_layers, dim=0)
        magnitude_stack = torch.abs(fft_stack)
        avg_magnitude = torch.mean(magnitude_stack, dim=0)
        
        # 중요한 주파수 성분 선택
        h, w = avg_magnitude.shape
        magnitude_flat = avg_magnitude.flatten()
        sorted_indices = torch.argsort(magnitude_flat, descending=True)
        
        keep_coeffs = int(len(magnitude_flat) * self.fft_quality)
        important_indices = sorted_indices[:keep_coeffs]
        
        mask = torch.zeros_like(magnitude_flat, dtype=torch.bool)
        mask[important_indices] = True
        mask = mask.reshape(h, w)
        
        # 가중 평균으로 융합 - 레이어 깊이에 따른 적응적 가중치
        layer_depths = torch.arange(len(weight_list), dtype=torch.float32)
        layer_weights = torch.softmax(layer_depths / 2.0, dim=0)  # 깊은 레이어에 더 많은 가중치
        
        # 주파수 도메인에서 학습 가능한 가중치 적용
        weighted_fft = torch.zeros_like(fft_stack[0])
        phase_consensus = torch.zeros_like(fft_stack[0])
        
        for i, weight in enumerate(layer_weights):
            # 위상 정보도 고려
            phase = torch.angle(fft_stack[i])
            phase_consensus += phase * weight
            weighted_fft += fft_stack[i] * weight * mask
        
        # 위상 보정 적용
        magnitude = torch.abs(weighted_fft)
        weighted_fft = magnitude * torch.exp(1j * phase_consensus)
        
        # IFFT로 복원
        fused_weight = torch.fft.ifft2(weighted_fft).real
        
        # 2. SVD 압축 적용
        U, S, V = torch.svd(fused_weight)
        
        # 적응적 rank 계산 - 에너지 기반 + 최소 rank 보장
        energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        
        # 더 스마트한 rank 선택: 에너지 보존 + gradient 기반
        energy_threshold = self.svd_rank_ratio
        rank = torch.sum(energy < energy_threshold).item() + 1
        
        # gradient 기반 rank 조정
        if rank > 1:
            energy_diff = energy[1:] - energy[:-1]
            largest_gaps = torch.argsort(energy_diff, descending=True)[:5]
            if len(largest_gaps) > 0:
                optimal_rank = largest_gaps[0].item() + 1
                rank = min(rank, optimal_rank)
        
        # 최소/최대 rank 제한
        min_rank = max(int(min(fused_weight.shape) * 0.05), 32)  # 최소 5% 또는 32
        max_rank = int(min(fused_weight.shape) * 0.9)  # 최대 90%
        rank = max(min_rank, min(rank, max_rank))
        
        print(f"   SVD rank: {min(fused_weight.shape)} → {rank} ({rank/min(fused_weight.shape):.1%})")
        print(f"   에너지 보존: {energy[rank-1]:.1%}")
        
        # 압축된 성분들
        U_compressed = U[:, :rank]
        S_compressed = S[:rank]
        V_compressed = V[:, :rank]
        
        return (nn.Parameter(U_compressed.to(weight_list[0].dtype).to(weight_list[0].device)),
                nn.Parameter(S_compressed.to(weight_list[0].dtype).to(weight_list[0].device)),
                nn.Parameter(V_compressed.to(weight_list[0].dtype).to(weight_list[0].device)))
        
    def forward(self, x):
        """Hybrid Super Layer 순전파"""
        # c_fc: SVD 복원 후 적용
        c_fc_weight = torch.mm(self.c_fc_U * self.c_fc_S.unsqueeze(0), self.c_fc_V.T)
        h = F.linear(x, c_fc_weight.T, self.c_fc_bias)
        h = self.activation(h)
        
        # c_proj: SVD 복원 후 적용
        c_proj_weight = torch.mm(self.c_proj_U * self.c_proj_S.unsqueeze(0), self.c_proj_V.T)
        output = F.linear(h, c_proj_weight.T, self.c_proj_bias)
        
        return output


def apply_hybrid_compression(model, svd_ratio=0.5, fft_quality=0.95, target_layers=None):
    """하이브리드 압축 적용"""
    
    print(f"\n🚀 하이브리드 압축 적용")
    print(f"   SVD ratio: {svd_ratio}")
    print(f"   FFT 품질: {fft_quality:.1%}")
    
    total_layers = len(model.transformer.h)
    
    if target_layers is None:
        # 후반부 레이어들을 융합 (절반부터)
        target_layers = list(range(total_layers // 2, total_layers))
    
    print(f"   융합 대상: {target_layers}")
    
    # MLP들 수집
    mlp_layers = [model.transformer.h[i].mlp for i in target_layers]
    
    # Super Layer 생성
    super_layer = SimpleHybridSuperLayer(
        mlp_layers, target_layers, svd_ratio, fft_quality
    )
    
    # 첫 번째 융합 레이어에 Super Layer 배치
    model.transformer.h[target_layers[0]].mlp = super_layer
    
    # 나머지 융합 레이어들 제거
    for i in reversed(target_layers[1:]):
        del model.transformer.h[i]
    
    return model, super_layer.compression_ratio


def test_accuracy_preservation(model, tokenizer):
    """정확도 보존 테스트"""
    
    print("📊 정확도 테스트")
    
    tests = [
        ("한국의 수도는", ["서울", "Seoul"]),
        ("안녕하세요", ["안녕", "반갑", "좋"]), 
        ("인공지능", ["AI", "기술", "컴퓨터"]),
        ("김치", ["음식", "한국", "먹"]),
        ("서울", ["한국", "수도", "도시"])
    ]
    
    correct = 0
    for prompt, expected_list in tests:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 관련성 체크
            score = 1 if any(exp in generated for exp in expected_list) else 0
            correct += score
            
            print(f"   '{prompt}' → '{generated[:40]}...' ({'✅' if score else '❌'})")
            
        except Exception as e:
            print(f"   '{prompt}' → 오류: {e} (❌)")
    
    accuracy = correct / len(tests)
    print(f"   정확도: {accuracy:.1%}")
    
    return accuracy


def simple_advanced_compression_test():
    """간단한 고급 압축 테스트"""
    
    print("🎯 Reality Stone 간단한 고급 압축 테스트")
    print("=" * 80)
    print("   목표: SVD + FFT Hybrid 압축 기능 검증")
    
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
    
    original_params = sum(p.numel() for p in model.parameters())
    original_layers = len(model.transformer.h)
    
    print(f"\n📊 원본 모델:")
    print(f"   레이어 수: {original_layers}")
    print(f"   파라미터: {original_params:,}")
    print(f"   크기: {original_params * 4 / (1024**2):.1f}MB")
    
    # 원본 정확도 측정
    print(f"\n📋 원본 모델 테스트")
    original_accuracy = test_accuracy_preservation(model, tokenizer)
    
    # 다양한 압축 설정 테스트
    compression_configs = [
        {'name': 'Light Compression', 'svd_ratio': 0.8, 'fft_quality': 0.98, 'target_layers': [10, 11]},
        {'name': 'Medium Compression', 'svd_ratio': 0.6, 'fft_quality': 0.95, 'target_layers': [8, 9, 10, 11]},
        {'name': 'High Compression', 'svd_ratio': 0.4, 'fft_quality': 0.90, 'target_layers': [6, 7, 8, 9, 10, 11]},
        {'name': 'Extreme Compression', 'svd_ratio': 0.3, 'fft_quality': 0.85, 'target_layers': [4, 5, 6, 7, 8, 9, 10, 11]},
    ]
    
    best_result = None
    
    for config in compression_configs:
        print(f"\n🎯 {config['name']}")
        print("=" * 60)
        
        try:
            # 모델 복사 및 압축
            compressed_model = copy.deepcopy(model)
            compressed_model, compression_ratio = apply_hybrid_compression(
                compressed_model, 
                svd_ratio=config['svd_ratio'],
                fft_quality=config['fft_quality'],
                target_layers=config['target_layers']
            )
            
            # 압축 후 통계
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            compressed_layers = len(compressed_model.transformer.h)
            actual_compression_ratio = compressed_params / original_params
            memory_saved = (original_params - compressed_params) * 4 / (1024**2)
            
            print(f"\n📊 압축 후 모델:")
            print(f"   레이어 수: {original_layers} → {compressed_layers}")
            print(f"   파라미터: {original_params:,} → {compressed_params:,}")
            print(f"   실제 압축률: {actual_compression_ratio:.3f}")
            print(f"   메모리 절약: {memory_saved:.1f}MB ({(1-actual_compression_ratio)*100:.1f}%)")
            
            # 압축 모델 정확도 테스트
            print(f"\n📋 압축 모델 테스트")
            compressed_accuracy = test_accuracy_preservation(compressed_model, tokenizer)
            
            # 정확도 보존율
            accuracy_retention = compressed_accuracy / original_accuracy if original_accuracy > 0 else 0
            
            print(f"\n📈 {config['name']} 결과:")
            print(f"   원본 정확도: {original_accuracy:.1%}")
            print(f"   압축 정확도: {compressed_accuracy:.1%}")  
            print(f"   정확도 보존: {accuracy_retention:.1%}")
            print(f"   메모리 절약: {(1-actual_compression_ratio)*100:.1f}%")
            print(f"   레이어 절약: {original_layers - compressed_layers}개")
            
            # 성과 평가
            high_compression = (1-actual_compression_ratio) >= 0.50  # 50%+ 절약
            good_accuracy = accuracy_retention >= 0.80  # 80%+ 보존
            
            if high_compression and good_accuracy:
                best_result = {
                    'name': config['name'],
                    'compression_ratio': actual_compression_ratio,
                    'accuracy_retention': accuracy_retention,
                    'memory_saved': 1-actual_compression_ratio,
                    'layers_saved': original_layers - compressed_layers
                }
                print(f"   🎉 우수한 성과! (50%+ 압축 + 80%+ 정확도)")
            
        except Exception as e:
            print(f"   ❌ 압축 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과
    print(f"\n🏆 간단한 고급 압축 최종 결과")
    print("=" * 80)
    
    if best_result:
        print(f"🥇 최고 성과: {best_result['name']}")
        print(f"   메모리 절약: {best_result['memory_saved']:.1%}")
        print(f"   정확도 보존: {best_result['accuracy_retention']:.1%}")
        print(f"   레이어 절약: {best_result['layers_saved']}개")
        print(f"   압축률: {best_result['compression_ratio']:.3f}")
        
        print(f"\n🎯 혁신적 성과:")
        print(f"   ✅ SVD + FFT Hybrid 압축 성공")
        print(f"   ✅ 구조적 압축: 여러 레이어 융합")
        print(f"   ✅ 높은 압축률 달성")
        print(f"   ✅ 정확도 상당 부분 보존")
        
        print(f"\n🚀 SVD + FFT Hybrid 압축 기술 검증 완료!")
    else:
        print("💪 압축 기능 검증 완료, 더 나은 파라미터 조정 필요")
    
    print(f"\n✅ 간단한 고급 압축 테스트 완료!")


if __name__ == "__main__":
    simple_advanced_compression_test() 