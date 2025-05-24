"""
Reality Stone 초고속 압축 엔진
사전 계산된 가중치로 런타임 오버헤드 완전 제거

이전 문제: SVD 재구성으로 인한 속도 저하 (0.5x)
해결책: 초기화 시점에 압축된 가중치 완전 계산 후 고정
목표: 1.5-2x 속도 향상 + 25-35% 압축률 + 90%+ 품질
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Reality Stone 백엔드 로드
import sys
sys.path.insert(0, '.')

try:
    import reality_stone
    print("✅ Reality Stone 백엔드 로드 성공!")
    REALITY_STONE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Reality Stone 백엔드 로드 실패: {e}")
    REALITY_STONE_AVAILABLE = False


class UltraFastCompressedLinear(nn.Module):
    """초고속 압축 선형 레이어 - 런타임 재구성 없음"""
    
            def __init__(self, original_linear, compression_ratio=0.4, layer_name="unknown"):        super().__init__()                self.layer_name = layer_name                # 원본 가중치 및 바이어스 (Conv1D 처리!)        original_weight = original_linear.weight.data.clone()        original_bias = original_linear.bias.data.clone() if original_linear.bias is not None else None                device = original_weight.device        dtype = original_weight.dtype                print(f"   UltraFast {layer_name}: {original_weight.shape} (압축률: {compression_ratio:.1%})")                # Conv1D는 (out_features, in_features), Linear는 (out_features, in_features)          # 하지만 SVD를 위해 (in_features, out_features)로 전치        if len(original_weight.shape) == 2:            weight_for_svd = original_weight.T  # (in, out) for SVD        else:            weight_for_svd = original_weight                # 1. 빠른 SVD 압축        U, S, V = torch.svd(weight_for_svd.float())
        
        # 2. 적응적 랭크 선택 (에너지 + 압축률 고려)
        total_rank = min(U.shape[1], V.shape[0])
        
        # 에너지 기반 중요 랭크
        energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        energy_rank = torch.sum(energy < 0.99).item() + 1  # 99% 에너지 보존
        
        # 압축률 기반 랭크
        target_rank = max(8, int(total_rank * compression_ratio))
        
        # 둘 중 작은 값 선택 (품질과 압축의 균형)
        final_rank = min(energy_rank, target_rank)
        final_rank = max(final_rank, 8)  # 최소 8개 보장
        
        print(f"   랭크 선택: 에너지({energy_rank}) vs 타겟({target_rank}) → 최종({final_rank})")
        
        # 3. 압축된 가중치 사전 계산 (핵심!)
        compressed_weight = U[:, :final_rank] @ torch.diag(S[:final_rank]) @ V[:, :final_rank].T
        
        # 4. 사전 계산된 가중치를 파라미터로 저장
        self.weight = nn.Parameter(compressed_weight.to(dtype).to(device))
        
        # 5. 바이어스 처리
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.to(dtype).to(device))
        else:
            self.register_parameter('bias', None)
        
        # 6. 압축 통계
        original_params = original_weight.numel() + (original_bias.numel() if original_bias is not None else 0)
        compressed_params = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
        self.actual_compression_ratio = compressed_params / original_params
        
        print(f"   파라미터: {original_params:,} → {compressed_params:,} ({self.actual_compression_ratio:.3f})")
        print(f"   에너지 보존: {energy[final_rank-1]:.3f}")
        
    def forward(self, x):
        """초고속 순전파 - 재구성 없음, 직접 linear 연산"""
        return F.linear(x, self.weight, self.bias)


class SmartCompressedMLP(nn.Module):
    """스마트 압축 MLP - 전략적 레이어별 압축"""
    
    def __init__(self, original_mlp, layer_idx=0, aggressive=False):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # 레이어 위치에 따른 압축 전략
        if layer_idx < 4:  # 초기 레이어 (품질 중요)
            compression_ratio = 0.6 if not aggressive else 0.4
        elif layer_idx < 8:  # 중간 레이어 (균형)
            compression_ratio = 0.5 if not aggressive else 0.3  
        else:  # 후반 레이어 (압축 중요)
            compression_ratio = 0.4 if not aggressive else 0.2
        
        print(f"\n📐 Layer {layer_idx} MLP 스마트 압축 (압축률: {compression_ratio:.1%})")
        
        # c_fc 압축 (입력 → 중간층)
        if hasattr(original_mlp, 'c_fc'):
            self.c_fc = UltraFastCompressedLinear(
                original_mlp.c_fc, compression_ratio, f"L{layer_idx}_c_fc"
            )
        
        # c_proj 압축 (중간층 → 출력) - 더 보수적
        if hasattr(original_mlp, 'c_proj'):
            conservative_ratio = compression_ratio * 1.2  # 20% 더 보수적
            self.c_proj = UltraFastCompressedLinear(
                original_mlp.c_proj, conservative_ratio, f"L{layer_idx}_c_proj"
            )
        
        # 활성화 함수
        self.activation = nn.GELU()
        
    def forward(self, x):
        """스마트 MLP 순전파"""
        # 표준 MLP 플로우: c_fc → activation → c_proj
        h = self.c_fc(x)
        h = self.activation(h)
        output = self.c_proj(h)
        return output


def apply_ultra_fast_compression(model, aggressive=False, target_layers=None):
    """초고속 압축 적용"""
    
    mode = "공격적" if aggressive else "균형적"
    print(f"\n🚀 초고속 압축 적용 ({mode} 모드)")
    print("   전략: 사전 계산된 가중치, 런타임 오버헤드 제거")
    
    if target_layers is None:
        # 전체 레이어 압축 (더 많은 메모리 절약)
        target_layers = list(range(len(model.transformer.h)))
    
    print(f"   대상 레이어: {len(target_layers)}개 (전체)")
    
    compressed_count = 0
    total_original = 0
    total_compressed = 0
    
    for layer_idx in target_layers:
        if layer_idx < len(model.transformer.h):
            layer = model.transformer.h[layer_idx]
            
            try:
                # 원본 MLP 파라미터 수
                original_mlp = layer.mlp
                original_params = sum(p.numel() for p in original_mlp.parameters())
                
                # SmartCompressedMLP로 교체
                compressed_mlp = SmartCompressedMLP(
                    original_mlp, layer_idx, aggressive
                )
                
                # MLP 교체
                layer.mlp = compressed_mlp
                
                # 압축된 파라미터 수
                compressed_params = sum(p.numel() for p in compressed_mlp.parameters())
                
                total_original += original_params
                total_compressed += compressed_params
                compressed_count += 1
                
                actual_ratio = compressed_params / original_params
                print(f"   ✅ Layer {layer_idx}: {original_params:,} → {compressed_params:,} ({actual_ratio:.1%})")
                
            except Exception as e:
                print(f"   ❌ Layer {layer_idx} 압축 실패: {e}")
    
    # 전체 모델 압축률 계산
    total_model_params = sum(p.numel() for p in model.parameters())
    mlp_compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
    overall_compression_ratio = (total_model_params - total_original + total_compressed) / total_model_params
    memory_saved = (total_original - total_compressed) * 4 / (1024**2)
    
    print(f"\n🎯 초고속 압축 완료:")
    print(f"   압축된 레이어: {compressed_count}개")
    print(f"   MLP 압축률: {mlp_compression_ratio:.1%}")
    print(f"   전체 모델 압축률: {overall_compression_ratio:.1%}")
    print(f"   메모리 절약: {memory_saved:.1f}MB")
    
    return model, overall_compression_ratio


def load_korean_model():
    """한글 모델 로드"""
    
    print("📥 한글 모델 로딩...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "skt/kogpt2-base-v2"
        print(f"   로딩: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"   ✅ 로드 성공!")
        return model, tokenizer, model_name
        
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")
        return None, None, None


def benchmark_speed(model, tokenizer, num_runs=50):
    """정밀 속도 벤치마크"""
    
    try:
        # 다양한 입력 길이로 테스트
        test_inputs = [
            "안녕",
            "안녕하세요 오늘은",
            "안녕하세요 오늘은 정말 좋은 날씨네요"
        ]
        
        all_times = []
        
        for test_input in test_inputs:
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # 워밍업
            with torch.no_grad():
                for _ in range(10):
                    _ = model(**inputs)
            
            # 측정
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(**inputs)
                times.append((time.time() - start_time) * 1000)
            
            avg_time = np.mean(times)
            all_times.append(avg_time)
            print(f"   '{test_input}': {avg_time:.2f}ms")
        
        overall_avg = np.mean(all_times)
        return overall_avg
        
    except Exception as e:
        print(f"   ❌ 속도 측정 실패: {e}")
        return 0.0


def test_generation_quality(model, tokenizer, prompts):
    """생성 품질 테스트"""
    
    print("📝 생성 품질 테스트")
    results = []
    times = []
    
    for i, prompt in enumerate(prompts):
        try:
            print(f"\n{i+1}. '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 25,  # 더 짧게
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    top_p=0.85
                )
            
            gen_time = (time.time() - start_time) * 1000
            times.append(gen_time)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(generated)
            
            print(f"   → {generated}")
            print(f"   시간: {gen_time:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ 생성 실패: {e}")
            results.append(f"[실패: {e}]")
            times.append(0)
    
    avg_time = np.mean(times) if times else 0
    print(f"\n⏱️ 평균 생성 시간: {avg_time:.1f}ms")
    
    return results, avg_time


def ultra_fast_compression_test():
    """초고속 압축 테스트"""
    
    print("🚀 Reality Stone 초고속 압축 테스트")
    print("=" * 80)
    print("   목표: 1.5-2x 속도 향상 + 25-35% 압축률 + 90%+ 품질")
    
    # 모델 로드
    model, tokenizer, model_name = load_korean_model()
    if model is None:
        return
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 모델 정보:")
    print(f"   모델: {model_name}")
    print(f"   파라미터: {original_params:,}")
    print(f"   크기: {original_params * 4 / (1024**2):.1f}MB")
    
    # 테스트 프롬프트 (더 짧게)
    test_prompts = [
        "안녕하세요",
        "인공지능",
        "한국 문화"
    ]
    
    # 원본 모델 테스트
    print(f"\n📋 원본 모델 벤치마크")
    print("-" * 60)
    
    print("⏱️ 원본 모델 속도 측정")
    original_speed = benchmark_speed(model, tokenizer)
    print(f"   평균 추론 시간: {original_speed:.2f}ms")
    
    original_results, original_gen_time = test_generation_quality(model, tokenizer, test_prompts)
    
    # 두 가지 모드로 테스트
    modes = [
        {"name": "균형", "aggressive": False},
        {"name": "공격적", "aggressive": True}
    ]
    
    best_result = None
    best_score = 0
    
    for mode in modes:
        print(f"\n🔧 {mode['name']} 모드 테스트")
        print("=" * 60)
        
        try:
            # 모델 복사 및 압축
            compressed_model = copy.deepcopy(model)
            compressed_model, actual_compression = apply_ultra_fast_compression(
                compressed_model, mode['aggressive']
            )
            
            # 압축 모델 속도 측정
            print("\n⏱️ 압축 모델 속도 측정")
            compressed_speed = benchmark_speed(compressed_model, tokenizer)
            speed_improvement = original_speed / compressed_speed if compressed_speed > 0 else 1.0
            print(f"   평균 추론 시간: {compressed_speed:.2f}ms")
            print(f"   속도 향상: {speed_improvement:.2f}x")
            
            # 압축 모델 생성 테스트
            compressed_results, compressed_gen_time = test_generation_quality(compressed_model, tokenizer, test_prompts)
            gen_speed_improvement = original_gen_time / compressed_gen_time if compressed_gen_time > 0 else 1.0
            
            # 품질 평가 (텍스트 길이 + 한글 비율)
            quality_score = 0
            korean_ratio = 0
            
            for orig, comp in zip(original_results, compressed_results):
                if isinstance(comp, str) and len(comp) > 5:
                    # 길이 비율
                    length_ratio = min(len(comp) / len(orig), 1.0) if len(orig) > 0 else 0
                    
                    # 한글 비율
                    korean_chars = sum(1 for c in comp if '가' <= c <= '힣')
                    total_chars = len(comp.replace(' ', ''))
                    kr_ratio = korean_chars / total_chars if total_chars > 0 else 0
                    
                    quality_score += length_ratio
                    korean_ratio += kr_ratio
            
            quality_score = quality_score / len(test_prompts) if test_prompts else 0
            korean_ratio = korean_ratio / len(test_prompts) if test_prompts else 0
            
            # 종합 점수 (속도 * 압축률 * 품질)
            memory_saved_ratio = 1 - actual_compression
            overall_score = speed_improvement * memory_saved_ratio * quality_score
            
            print(f"\n📊 {mode['name']} 모드 결과:")
            print(f"   실제 압축률: {actual_compression:.1%}")
            print(f"   메모리 절약: {memory_saved_ratio:.1%}")
            print(f"   추론 속도 향상: {speed_improvement:.2f}x")
            print(f"   생성 속도 향상: {gen_speed_improvement:.2f}x")
            print(f"   품질 점수: {quality_score:.3f}")
            print(f"   한글 비율: {korean_ratio:.3f}")
            print(f"   종합 점수: {overall_score:.3f}")
            
            # 최고 성능 기록
            if overall_score > best_score:
                best_score = overall_score
                best_result = {
                    'mode': mode['name'],
                    'actual_compression': actual_compression,
                    'memory_saved': memory_saved_ratio,
                    'speed_improvement': speed_improvement,
                    'gen_speed_improvement': gen_speed_improvement,
                    'quality_score': quality_score,
                    'korean_ratio': korean_ratio,
                    'overall_score': overall_score
                }
            
        except Exception as e:
            print(f"   ❌ 압축 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과 요약
    print(f"\n🏆 초고속 압축 최종 결과")
    print("=" * 80)
    
    if best_result:
        print(f"🥇 최고 성능 ({best_result['mode']} 모드):")
        print(f"   실제 압축률: {best_result['actual_compression']:.1%}")
        print(f"   메모리 절약: {best_result['memory_saved']:.1%}")
        print(f"   추론 속도 향상: {best_result['speed_improvement']:.2f}x")
        print(f"   생성 속도 향상: {best_result['gen_speed_improvement']:.2f}x")
        print(f"   품질 점수: {best_result['quality_score']:.3f}")
        print(f"   한글 비율: {best_result['korean_ratio']:.3f}")
        print(f"   종합 점수: {best_result['overall_score']:.3f}")
        
        print(f"\n🎯 목표 달성도:")
        speed_ok = best_result['speed_improvement'] >= 1.5
        compress_ok = best_result['memory_saved'] >= 0.25
        quality_ok = best_result['quality_score'] >= 0.9
        
        print(f"   속도 개선: {'✅' if speed_ok else '⚠️'} (목표: 1.5-2x, 달성: {best_result['speed_improvement']:.1f}x)")
        print(f"   압축률: {'✅' if compress_ok else '⚠️'} (목표: 25-35%, 달성: {best_result['memory_saved']:.1%})")
        print(f"   품질 유지: {'✅' if quality_ok else '⚠️'} (목표: 90%+, 달성: {best_result['quality_score']:.1%})")
        
        if speed_ok and compress_ok and quality_ok:
            print(f"\n🎉 모든 목표 달성! 초고속 압축 성공!")
        else:
            print(f"\n🔄 일부 목표 미달성, 추가 최적화 필요")
    else:
        print("❌ 성공적인 압축 결과 없음")
    
    print(f"\n✅ 초고속 압축 테스트 완료!")


if __name__ == "__main__":
    ultra_fast_compression_test() 