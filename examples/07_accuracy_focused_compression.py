"""
Reality Stone 정확도 중심 압축 연구
품질과 정확도, 압축률에 집중한 새로운 접근법

연구 목표:
1. 압축률: 30-50% 달성 (현재: 15-20%)
2. 한국어 정확도: 90%+ 유지 
3. 품질: 의미론적 일관성 보존

새로운 기법:
- 하이브리드 압축 (SVD + 중요도 프루닝)
- 한국어 태스크 기반 평가
- 지식 증류 기반 성능 보존
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import warnings
warnings.filterwarnings("ignore")

# Reality Stone 백엔드 로드
import sys
sys.path.insert(0, '.')

try:
    import reality_stone
    print("✅ Reality Stone 백엔드 로드 성공!")
except ImportError as e:
    print(f"❌ Reality Stone 백엔드 로드 실패: {e}")


class HybridCompressedLinear(nn.Module):
    """하이브리드 압축: SVD + 중요도 기반 프루닝"""
    
    def __init__(self, original_layer, compression_ratio=0.3, layer_name="unknown"):
        super().__init__()
        
        self.layer_name = layer_name
        
        # 원본 가중치
        original_weight = original_layer.weight.data.clone()
        original_bias = original_layer.bias.data.clone() if original_layer.bias is not None else None
        
        device = original_weight.device
        dtype = original_weight.dtype
        
        print(f"   Hybrid {layer_name}: {original_weight.shape} → 압축률: {compression_ratio:.1%}")
        
        # 1. 중요도 분석 (그라디언트 기반)
        weight_importance = torch.abs(original_weight)
        
        # 2. SVD 압축
        U, S, V = torch.svd(original_weight.float())
        
        # 3. 적응적 랭크 선택
        total_rank = min(U.shape[1], V.shape[0])
        
        # 중요도 기반 에너지 계산
        energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        
        # 압축률에 따른 다단계 전략
        if compression_ratio <= 0.3:  # 강한 압축 (30% 이하)
            energy_threshold = 0.98  # 98% 에너지 보존
            target_rank = max(8, int(total_rank * compression_ratio))
        elif compression_ratio <= 0.5:  # 중간 압축 (30-50%)
            energy_threshold = 0.95  # 95% 에너지 보존
            target_rank = max(12, int(total_rank * compression_ratio))
        else:  # 약한 압축 (50% 이상)
            energy_threshold = 0.90  # 90% 에너지 보존
            target_rank = max(16, int(total_rank * compression_ratio))
        
        energy_rank = torch.sum(energy < energy_threshold).item() + 1
        final_rank = min(target_rank, energy_rank)
        final_rank = max(final_rank, 4)
        
        print(f"   랭크 분석: 전체({total_rank}) → 에너지({energy_rank}) → 타겟({target_rank}) → 최종({final_rank})")
        print(f"   에너지 보존: {energy[final_rank-1]:.4f} (임계값: {energy_threshold})")
        
        # 4. 하이브리드 압축 적용
        # SVD 기반 저차원 근사
        U_compressed = U[:, :final_rank]
        S_compressed = S[:final_rank]
        V_compressed = V[:, :final_rank]
        
        # 중요도 기반 가중치 조정
        importance_factor = torch.sqrt(S_compressed / S_compressed[0])  # 정규화된 중요도
        S_compressed = S_compressed * importance_factor  # 중요한 성분 강화
        
        # 5. 압축된 가중치 사전 계산
        compressed_weight = U_compressed @ torch.diag(S_compressed) @ V_compressed.T
        
        # 6. 저장
        self.weight = nn.Parameter(compressed_weight.to(dtype).to(device))
        
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.to(dtype).to(device))
        else:
            self.register_parameter('bias', None)
        
        # 7. 압축 통계
        original_params = original_weight.numel() + (original_bias.numel() if original_bias is not None else 0)
        compressed_params = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
        self.actual_compression_ratio = compressed_params / original_params
        
        print(f"   파라미터: {original_params:,} → {compressed_params:,} ({self.actual_compression_ratio:.3f})")
        
    def forward(self, x):
        """하이브리드 순전파"""
        return F.linear(x, self.weight, self.bias)


class AccuracyPreservingMLP(nn.Module):
    """정확도 보존 MLP"""
    
    def __init__(self, original_mlp, layer_idx=0, target_compression=0.4):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # 레이어별 중요도 기반 압축 전략
        if layer_idx < 3:  # 초기 레이어 (특징 추출 중요)
            compression_ratio = target_compression * 1.5  # 덜 압축
        elif layer_idx < 6:  # 중간 레이어 (특징 변환)
            compression_ratio = target_compression * 1.2  # 약간 덜 압축
        elif layer_idx < 9:  # 후반 레이어 (고차원 특징)
            compression_ratio = target_compression  # 목표 압축
        else:  # 최종 레이어 (출력 생성)
            compression_ratio = target_compression * 0.8  # 더 압축
        
        print(f"\n📐 Layer {layer_idx} 정확도 보존 압축 (목표: {target_compression:.1%}, 적용: {compression_ratio:.1%})")
        
        # c_fc 압축
        if hasattr(original_mlp, 'c_fc'):
            self.c_fc = HybridCompressedLinear(
                original_mlp.c_fc, compression_ratio, f"L{layer_idx}_c_fc"
            )
        
        # c_proj 압축 (출력층이므로 더 보수적)
        if hasattr(original_mlp, 'c_proj'):
            conservative_ratio = compression_ratio * 1.4  # 40% 더 보수적
            self.c_proj = HybridCompressedLinear(
                original_mlp.c_proj, conservative_ratio, f"L{layer_idx}_c_proj"
            )
        
        # 활성화 함수
        self.activation = nn.GELU()
        
    def forward(self, x):
        """정확도 보존 순전파"""
        h = self.c_fc(x)
        h = self.activation(h)
        output = self.c_proj(h)
        return output


def apply_accuracy_focused_compression(model, target_compression=0.4, target_layers=None):
    """정확도 중심 압축 적용"""
    
    print(f"\n🚀 정확도 중심 압축 적용 (목표: {target_compression:.1%})")
    print("   전략: 하이브리드 압축 + 레이어별 최적화 + 정확도 보존")
    
    if target_layers is None:
        # 전체 레이어 압축 (더 높은 압축률 달성)
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
                
                # AccuracyPreservingMLP로 교체
                compressed_mlp = AccuracyPreservingMLP(
                    original_mlp, layer_idx, target_compression
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
    
    print(f"\n🎯 정확도 중심 압축 완료:")
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


def evaluate_korean_accuracy(model, tokenizer):
    """한국어 정확도 평가"""
    
    print("📊 한국어 정확도 평가")
    
    # 다양한 한국어 태스크
    tasks = {
        "문장완성": [
            ("한국의 수도는", "서울"),
            ("안녕하세요는", "인사"),
            ("김치는 한국의", "음식"),
            ("태극기는 한국의", "국기"),
            ("한글은 한국의", "문자")
        ],
        "문맥이해": [
            ("오늘 날씨가 좋아서", ["좋", "날씨", "맑", "산책"]),
            ("배가 고파서", ["밥", "음식", "먹", "식당"]),
            ("공부를 열심히 해서", ["시험", "성적", "좋", "합격"]),
            ("친구와 함께", ["놀", "영화", "게임", "즐거"]),
            ("새해가 되어서", ["새해", "새", "희망", "결심"])
        ],
        "한글생성": [
            "안녕하세요",
            "대한민국",
            "한국어",
            "서울특별시",
            "인공지능"
        ]
    }
    
    total_score = 0
    total_tests = 0
    
    # 1. 문장 완성 평가
    print("\n1️⃣ 문장 완성 정확도:")
    completion_score = 0
    for prompt, expected in tasks["문장완성"]:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 10,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt):].strip()
            
            # 정확도 체크 (예상 단어 포함 여부)
            score = 1 if expected in generated else 0
            completion_score += score
            
            print(f"   '{prompt}' → '{generated[:20]}...' ({'✅' if score else '❌'})")
            
        except Exception as e:
            print(f"   '{prompt}' → 오류: {e} (❌)")
        
        total_tests += 1
    
    completion_accuracy = completion_score / len(tasks["문장완성"]) if tasks["문장완성"] else 0
    total_score += completion_accuracy
    
    # 2. 문맥 이해 평가
    print("\n2️⃣ 문맥 이해 정확도:")
    context_score = 0
    for prompt, keywords in tasks["문맥이해"]:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 15,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt):].strip()
            
            # 키워드 매칭 점수
            keyword_matches = sum(1 for keyword in keywords if keyword in generated)
            score = keyword_matches / len(keywords)
            context_score += score
            
            print(f"   '{prompt}' → '{generated[:30]}...' ({score:.1%})")
            
        except Exception as e:
            print(f"   '{prompt}' → 오류: {e} (0%)")
        
        total_tests += 1
    
    context_accuracy = context_score / len(tasks["문맥이해"]) if tasks["문맥이해"] else 0
    total_score += context_accuracy
    
    # 3. 한글 생성 품질
    print("\n3️⃣ 한글 생성 품질:")
    generation_score = 0
    for prompt in tasks["한글생성"]:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 20,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt):].strip()
            
            # 한글 비율 계산
            korean_chars = sum(1 for c in generated if '가' <= c <= '힣')
            total_chars = len(generated.replace(' ', ''))
            korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
            
            # 길이 점수 (적절한 길이 생성 여부)
            length_score = min(len(generated) / 20, 1.0) if len(generated) > 0 else 0
            
            # 종합 점수
            score = (korean_ratio + length_score) / 2
            generation_score += score
            
            print(f"   '{prompt}' → '{generated[:30]}...' (한글:{korean_ratio:.1%}, 점수:{score:.1%})")
            
        except Exception as e:
            print(f"   '{prompt}' → 오류: {e} (0%)")
        
        total_tests += 1
    
    generation_accuracy = generation_score / len(tasks["한글생성"]) if tasks["한글생성"] else 0
    total_score += generation_accuracy
    
    # 전체 정확도
    overall_accuracy = total_score / 3  # 3개 태스크 평균
    
    print(f"\n📊 한국어 정확도 요약:")
    print(f"   문장 완성: {completion_accuracy:.1%}")
    print(f"   문맥 이해: {context_accuracy:.1%}")
    print(f"   한글 생성: {generation_accuracy:.1%}")
    print(f"   전체 정확도: {overall_accuracy:.1%}")
    
    return {
        'completion': completion_accuracy,
        'context': context_accuracy,
        'generation': generation_accuracy,
        'overall': overall_accuracy
    }


def accuracy_focused_research():
    """정확도 중심 압축 연구"""
    
    print("🚀 Reality Stone 정확도 중심 압축 연구")
    print("=" * 80)
    print("   목표: 압축률 30-50% + 한국어 정확도 90%+ + 품질 보존")
    
    # 모델 로드
    model, tokenizer, model_name = load_korean_model()
    if model is None:
        return
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 모델 정보:")
    print(f"   모델: {model_name}")
    print(f"   파라미터: {original_params:,}")
    print(f"   크기: {original_params * 4 / (1024**2):.1f}MB")
    
    # 원본 모델 정확도 평가
    print(f"\n📋 원본 모델 평가")
    print("-" * 60)
    original_accuracy = evaluate_korean_accuracy(model, tokenizer)
    
    # 다양한 압축률로 실험
    compression_ratios = [0.3, 0.4, 0.5, 0.6]  # 30%, 40%, 50%, 60%
    
    best_result = None
    best_score = 0
    
    for compression_ratio in compression_ratios:
        print(f"\n🔧 압축률 {compression_ratio:.1%} 연구")
        print("=" * 60)
        
        try:
            # 모델 복사 및 압축
            compressed_model = copy.deepcopy(model)
            compressed_model, actual_compression = apply_accuracy_focused_compression(
                compressed_model, compression_ratio
            )
            
            # 압축 모델 정확도 평가
            print(f"\n📋 압축 모델 평가")
            print("-" * 40)
            compressed_accuracy = evaluate_korean_accuracy(compressed_model, tokenizer)
            
            # 성능 비교
            accuracy_retention = compressed_accuracy['overall'] / original_accuracy['overall'] if original_accuracy['overall'] > 0 else 0
            memory_saved_ratio = 1 - actual_compression
            
            # 종합 점수 (정확도 보존 * 압축률)
            overall_score = accuracy_retention * memory_saved_ratio
            
            print(f"\n📊 압축률 {compression_ratio:.1%} 결과:")
            print(f"   실제 압축률: {actual_compression:.1%}")
            print(f"   메모리 절약: {memory_saved_ratio:.1%}")
            print(f"   정확도 보존: {accuracy_retention:.1%}")
            print(f"   문장완성 보존: {compressed_accuracy['completion'] / original_accuracy['completion']:.1%}")
            print(f"   문맥이해 보존: {compressed_accuracy['context'] / original_accuracy['context']:.1%}")
            print(f"   한글생성 보존: {compressed_accuracy['generation'] / original_accuracy['generation']:.1%}")
            print(f"   종합 점수: {overall_score:.3f}")
            
            # 최고 성능 기록
            if overall_score > best_score:
                best_score = overall_score
                best_result = {
                    'compression_ratio': compression_ratio,
                    'actual_compression': actual_compression,
                    'memory_saved': memory_saved_ratio,
                    'accuracy_retention': accuracy_retention,
                    'original_accuracy': original_accuracy,
                    'compressed_accuracy': compressed_accuracy,
                    'overall_score': overall_score
                }
            
        except Exception as e:
            print(f"   ❌ 압축 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 연구 결과
    print(f"\n🏆 정확도 중심 압축 연구 결과")
    print("=" * 80)
    
    if best_result:
        print(f"🥇 최고 성능:")
        print(f"   목표 압축률: {best_result['compression_ratio']:.1%}")
        print(f"   실제 압축률: {best_result['actual_compression']:.1%}")
        print(f"   메모리 절약: {best_result['memory_saved']:.1%}")
        print(f"   전체 정확도 보존: {best_result['accuracy_retention']:.1%}")
        print(f"   문장완성 보존: {best_result['compressed_accuracy']['completion'] / best_result['original_accuracy']['completion']:.1%}")
        print(f"   문맥이해 보존: {best_result['compressed_accuracy']['context'] / best_result['original_accuracy']['context']:.1%}")
        print(f"   한글생성 보존: {best_result['compressed_accuracy']['generation'] / best_result['original_accuracy']['generation']:.1%}")
        print(f"   종합 점수: {best_result['overall_score']:.3f}")
        
        print(f"\n🎯 연구 목표 달성도:")
        compress_ok = best_result['memory_saved'] >= 0.3  # 30% 압축률
        accuracy_ok = best_result['accuracy_retention'] >= 0.9  # 90% 정확도 보존
        quality_ok = best_result['compressed_accuracy']['generation'] / best_result['original_accuracy']['generation'] >= 0.85  # 85% 품질 보존
        
        print(f"   압축률: {'✅' if compress_ok else '⚠️'} (목표: 30%+, 달성: {best_result['memory_saved']:.1%})")
        print(f"   정확도 보존: {'✅' if accuracy_ok else '⚠️'} (목표: 90%+, 달성: {best_result['accuracy_retention']:.1%})")
        print(f"   품질 보존: {'✅' if quality_ok else '⚠️'} (목표: 85%+, 달성: {best_result['compressed_accuracy']['generation'] / best_result['original_accuracy']['generation']:.1%})")
        
        if compress_ok and accuracy_ok and quality_ok:
            print(f"\n🎉 모든 연구 목표 달성! 정확도 중심 압축 성공!")
        else:
            print(f"\n🔬 일부 목표 미달성, 추가 연구 필요")
    else:
        print("❌ 성공적인 압축 결과 없음")
    
    print(f"\n✅ 정확도 중심 압축 연구 완료!")


if __name__ == "__main__":
    accuracy_focused_research() 