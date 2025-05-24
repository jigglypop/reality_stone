"""
Reality Stone 고속 품질 압축 엔진
속도 최적화 + 품질 개선에 중점을 둔 헬가손 압축

이전 성과: 55% 압축률 달성
문제점: 23배 속도 저하, 품질 저하
목표: 30-40% 압축률 + 2-3배 속도 향상 + 품질 유지
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


class FastHelgasonMLP(nn.Module):
    """고속 헬가손 MLP - 사전 계산된 가중치 사용"""
    
    def __init__(self, original_weight, compression_ratio=0.3, layer_name="unknown"):
        super().__init__()
        
        self.layer_name = layer_name
        self.compression_ratio = compression_ratio
        
        device = original_weight.device
        dtype = original_weight.dtype
        
        print(f"   FastHelgason {layer_name}: {original_weight.shape} (압축률: {compression_ratio:.1%})")
        
        # 1. 빠른 SVD 압축 (하이퍼볼릭 대신 효율적인 SVD)
        U, S, V = torch.svd(original_weight.float())
        
        # 2. 압축 랭크 계산 (30% 압축 목표)
        total_rank = min(U.shape[1], V.shape[0])
        compressed_rank = max(4, int(total_rank * compression_ratio))
        
        print(f"   압축 랭크: {total_rank} → {compressed_rank}")
        
        # 3. 중요도 기반 특이값 선택
        # 특이값의 누적 에너지로 중요한 성분만 선택
        energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        energy_threshold = 0.95  # 95% 에너지 보존
        important_rank = torch.sum(energy < energy_threshold).item() + 1
        
        # 압축 랭크와 중요도 랭크 중 작은 값 선택
        final_rank = min(compressed_rank, important_rank)
        final_rank = max(final_rank, 4)  # 최소 4개 유지
        
        print(f"   최종 랭크: {final_rank} (에너지 보존: {energy[final_rank-1]:.3f})")
        
        # 4. 압축된 행렬 저장 (사전 계산)
        self.U_compressed = nn.Parameter(U[:, :final_rank].to(dtype).to(device))
        self.S_compressed = nn.Parameter(S[:final_rank].to(dtype).to(device))
        self.V_compressed = nn.Parameter(V[:, :final_rank].to(dtype).to(device))
        
        # 5. 압축 통계
        original_params = original_weight.numel()
        compressed_params = self.U_compressed.numel() + self.S_compressed.numel() + self.V_compressed.numel()
        self.actual_compression_ratio = compressed_params / original_params
        
        print(f"   파라미터: {original_params:,} → {compressed_params:,} ({self.actual_compression_ratio:.3f})")
        
    def forward(self, x):
        """고속 순전파 - 사전 계산된 가중치 사용"""
        # SVD 재구성: W = U @ diag(S) @ V^T
        weight = self.U_compressed @ torch.diag(self.S_compressed) @ self.V_compressed.T
        return F.linear(x, weight)


class QualityPreservingMLP(nn.Module):
    """품질 보존 MLP - 핵심 레이어만 압축"""
    
    def __init__(self, original_mlp, compression_ratio=0.3, layer_idx=0):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # c_fc (입력 → 중간층) 압축
        if hasattr(original_mlp, 'c_fc'):
            c_fc_weight = original_mlp.c_fc.weight.data.clone()
            self.c_fc_compressed = FastHelgasonMLP(
                c_fc_weight.T, compression_ratio, f"Layer{layer_idx}_c_fc"
            )
            self.c_fc_bias = nn.Parameter(original_mlp.c_fc.bias.data.clone())
        
        # c_proj (중간층 → 출력) - 더 보수적으로 압축
        if hasattr(original_mlp, 'c_proj'):
            c_proj_weight = original_mlp.c_proj.weight.data.clone()
            # c_proj는 품질에 더 중요하므로 덜 압축
            conservative_ratio = compression_ratio * 1.5  # 30% → 45%
            self.c_proj_compressed = FastHelgasonMLP(
                c_proj_weight.T, conservative_ratio, f"Layer{layer_idx}_c_proj"
            )
            self.c_proj_bias = nn.Parameter(original_mlp.c_proj.bias.data.clone())
        
        # 활성화 함수
        self.activation = nn.GELU()
        
    def forward(self, x):
        """품질 보존 순전파"""
        # c_fc (압축)
        h = self.c_fc_compressed(x) + self.c_fc_bias
        
        # 활성화
        h = self.activation(h)
        
        # c_proj (보수적 압축)
        output = self.c_proj_compressed(h) + self.c_proj_bias
        
        return output


def apply_fast_quality_compression(model, compression_ratio=0.3, target_layers=None):
    """고속 품질 압축 적용"""
    
    print(f"\n🚀 고속 품질 압축 적용 (압축률: {compression_ratio:.1%})")
    print("   전략: MLP만 압축, 어텐션 보존, 사전 계산된 가중치")
    
    if target_layers is None:
        # 후반부 레이어만 압축 (품질 영향 최소화)
        total_layers = len(model.transformer.h)
        target_layers = list(range(total_layers//2, total_layers))  # 후반 절반
    
    print(f"   대상 레이어: {target_layers}")
    
    compressed_count = 0
    total_original = 0
    total_compressed = 0
    
    for layer_idx in target_layers:
        if layer_idx < len(model.transformer.h):
            layer = model.transformer.h[layer_idx]
            
            print(f"\n📐 Layer {layer_idx} MLP 압축 중...")
            
            try:
                # 원본 MLP 파라미터 수
                original_mlp = layer.mlp
                original_params = sum(p.numel() for p in original_mlp.parameters())
                
                # QualityPreservingMLP로 교체
                compressed_mlp = QualityPreservingMLP(
                    original_mlp, compression_ratio, layer_idx
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
                import traceback
                traceback.print_exc()
    
    # 전체 모델 압축률 계산
    total_model_params = sum(p.numel() for p in model.parameters())
    mlp_compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
    overall_compression_ratio = (total_model_params - total_original + total_compressed) / total_model_params
    memory_saved = (total_original - total_compressed) * 4 / (1024**2)
    
    print(f"\n🎯 고속 품질 압축 완료:")
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


def test_generation(model, tokenizer, prompts, max_new_tokens=30):
    """생성 테스트"""
    
    print("📝 한글 생성 테스트")
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
                    max_length=len(inputs.input_ids[0]) + max_new_tokens,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
            
            gen_time = (time.time() - start_time) * 1000
            times.append(gen_time)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(generated)
            
            print(f"   생성: {generated}")
            print(f"   시간: {gen_time:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ 생성 실패: {e}")
            results.append(f"[실패: {e}]")
            times.append(0)
    
    avg_time = np.mean(times) if times else 0
    print(f"\n⏱️ 평균 생성 시간: {avg_time:.1f}ms")
    
    return results, avg_time


def measure_inference_speed(model, tokenizer, test_prompt="안녕하세요", num_runs=20):
    """정확한 추론 속도 측정"""
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # 워밍업 (5회)
        print("   워밍업 중...")
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inputs)
        
        # 실제 측정
        print(f"   측정 중... ({num_runs}회)")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(**inputs)
        
        avg_time = (time.time() - start_time) / num_runs * 1000
        return avg_time
        
    except Exception as e:
        print(f"   ❌ 속도 측정 실패: {e}")
        return 0.0


def fast_quality_compression_test():
    """고속 품질 압축 테스트"""
    
    print("🚀 Reality Stone 고속 품질 압축 테스트")
    print("=" * 80)
    print("   목표: 속도 2-3배 향상 + 품질 유지 + 30-40% 압축")
    
    # 모델 로드
    model, tokenizer, model_name = load_korean_model()
    if model is None:
        return
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 모델 정보:")
    print(f"   모델: {model_name}")
    print(f"   파라미터: {original_params:,}")
    print(f"   크기: {original_params * 4 / (1024**2):.1f}MB")
    
    # 테스트 프롬프트
    test_prompts = [
        "안녕하세요, 오늘은",
        "인공지능의 발전으로",
        "한국의 전통 문화",
        "미래 기술 전망"
    ]
    
    # 원본 모델 테스트
    print(f"\n📋 원본 모델 테스트")
    print("-" * 60)
    
    # 원본 속도 측정
    print("⏱️ 원본 모델 추론 속도 측정")
    original_speed = measure_inference_speed(model, tokenizer)
    print(f"   평균 추론 시간: {original_speed:.2f}ms")
    
    # 원본 생성 테스트
    original_results, original_gen_time = test_generation(model, tokenizer, test_prompts)
    
    # 다양한 압축률로 테스트
    compression_ratios = [0.2, 0.3, 0.4]  # 20%, 30%, 40% 압축
    
    best_result = None
    best_score = 0  # 압축률 * 속도향상 * 품질지수
    
    for compression_ratio in compression_ratios:
        print(f"\n🔧 압축률 {compression_ratio:.1%} 테스트")
        print("=" * 60)
        
        try:
            # 모델 복사 및 압축
            compressed_model = copy.deepcopy(model)
            compressed_model, actual_compression = apply_fast_quality_compression(
                compressed_model, compression_ratio
            )
            
            # 압축 모델 속도 측정
            print("\n⏱️ 압축 모델 추론 속도 측정")
            compressed_speed = measure_inference_speed(compressed_model, tokenizer)
            speed_improvement = original_speed / compressed_speed if compressed_speed > 0 else 1.0
            print(f"   평균 추론 시간: {compressed_speed:.2f}ms")
            print(f"   속도 향상: {speed_improvement:.2f}x")
            
            # 압축 모델 생성 테스트
            compressed_results, compressed_gen_time = test_generation(compressed_model, tokenizer, test_prompts)
            gen_speed_improvement = original_gen_time / compressed_gen_time if compressed_gen_time > 0 else 1.0
            
            # 품질 평가 (간단한 길이 기반)
            quality_score = 0
            for orig, comp in zip(original_results, compressed_results):
                if isinstance(comp, str) and len(comp) > 10:
                    # 생성된 텍스트 길이 비율로 품질 추정
                    length_ratio = min(len(comp) / len(orig), 1.0) if len(orig) > 0 else 0
                    quality_score += length_ratio
            
            quality_score = quality_score / len(test_prompts) if test_prompts else 0
            
            # 종합 점수 계산
            memory_saved_ratio = 1 - actual_compression
            overall_score = memory_saved_ratio * speed_improvement * quality_score
            
            print(f"\n📊 압축률 {compression_ratio:.1%} 결과:")
            print(f"   실제 압축률: {actual_compression:.1%}")
            print(f"   메모리 절약: {memory_saved_ratio:.1%}")
            print(f"   추론 속도 향상: {speed_improvement:.2f}x")
            print(f"   생성 속도 향상: {gen_speed_improvement:.2f}x")
            print(f"   품질 점수: {quality_score:.3f}")
            print(f"   종합 점수: {overall_score:.3f}")
            
            # 최고 성능 기록
            if overall_score > best_score:
                best_score = overall_score
                best_result = {
                    'compression_ratio': compression_ratio,
                    'actual_compression': actual_compression,
                    'memory_saved': memory_saved_ratio,
                    'speed_improvement': speed_improvement,
                    'gen_speed_improvement': gen_speed_improvement,
                    'quality_score': quality_score,
                    'overall_score': overall_score
                }
            
        except Exception as e:
            print(f"   ❌ 압축 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과 요약
    print(f"\n🏆 고속 품질 압축 최종 결과")
    print("=" * 80)
    
    if best_result:
        print(f"🥇 최고 성능:")
        print(f"   압축률: {best_result['compression_ratio']:.1%} (실제: {best_result['actual_compression']:.1%})")
        print(f"   메모리 절약: {best_result['memory_saved']:.1%}")
        print(f"   추론 속도 향상: {best_result['speed_improvement']:.2f}x")
        print(f"   생성 속도 향상: {best_result['gen_speed_improvement']:.2f}x")
        print(f"   품질 점수: {best_result['quality_score']:.3f}")
        print(f"   종합 점수: {best_result['overall_score']:.3f}")
        
        print(f"\n🎯 목표 달성도:")
        print(f"   속도 개선: {'✅' if best_result['speed_improvement'] >= 2 else '⚠️'} (목표: 2-3x, 달성: {best_result['speed_improvement']:.1f}x)")
        print(f"   압축률: {'✅' if best_result['memory_saved'] >= 0.3 else '⚠️'} (목표: 30-40%, 달성: {best_result['memory_saved']:.1%})")
        print(f"   품질 유지: {'✅' if best_result['quality_score'] >= 0.8 else '⚠️'} (목표: 80%+, 달성: {best_result['quality_score']:.1%})")
    else:
        print("❌ 성공적인 압축 결과 없음")
    
    print(f"\n✅ 고속 품질 압축 테스트 완료!")


if __name__ == "__main__":
    fast_quality_compression_test() 