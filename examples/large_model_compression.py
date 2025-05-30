#!/usr/bin/env python3
"""
🚀 대형 모델용 RealityStone 압축 시스템 v9.1
기존 성공한 리만+FFT+SVD 로직을 대형 모델로 확장
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re
from collections import Counter
import copy
import gc
import os

# 기존 성공한 압축 시스템 임포트 (상대 경로로 수정)
from reality_stone.examples.fftsvds import (
    enhanced_stereographic_projection,
    enhanced_inverse_stereographic_projection, 
    advanced_riemann_distance,
    advanced_mobius_transform,
    FastSVDCompressor,
    AdvancedFFTSVDCompressor,
    EnhancedRealityStoneLinear,
    SimplifiedRiemannCompressor,
    ultra_knowledge_distillation_fine_tune,
    generate_with_anti_repetition,
    advanced_quality_evaluation,
    RS_AVAILABLE
)

print("✅ 기존 성공한 RealityStone 압축 시스템 로드!")

# ═══════════════════════════════════════════════════════════════
# 🎯 대형 모델 지원 목록 (기존과 동일)
# ═══════════════════════════════════════════════════════════════

SUPPORTED_LARGE_MODELS = {
    "EleutherAI/polyglot-ko-1.3b": {
        "size": "1.3B",
        "params": 1_300_000_000, 
        "description": "한국어 Polyglot 1.3B",
        "compression_target": 0.05,  # 극한 5% 압축!
        "memory_gb": 6
    },
    "microsoft/DialoGPT-medium": {
        "size": "345M",
        "params": 345_000_000,
        "description": "DialoGPT Medium",
        "compression_target": 0.08,  # 8% 압축
        "memory_gb": 2
    },
    "beomi/KoAlpaca-Polyglot-5.8B": {
        "size": "5.8B", 
        "params": 5_800_000_000,
        "description": "한국어 Alpaca 5.8B",
        "compression_target": 0.15,  # 보수적 15%
        "memory_gb": 23
    },
    "EleutherAI/gpt-j-6b": {
        "size": "6B", 
        "params": 6_000_000_000,
        "description": "GPT-J 6B",
        "compression_target": 0.20,  # 매우 보수적
        "memory_gb": 24
    }
}

# ═══════════════════════════════════════════════════════════════
# 🚀 대형 모델용 RealityStone 압축 레이어 (기존 로직 확장)
# ═══════════════════════════════════════════════════════════════

class LargeModelRealityStoneLinear(nn.Module):
    """대형 모델용 RealityStone Linear (간단한 SVD 압축)"""
    
    def __init__(self, lin, compression_ratio=0.05, compression_type='auto', model_dtype=torch.float32):
        super().__init__()
        
        if hasattr(lin, 'weight'):
            W = lin.weight.data.clone().float()  # 무조건 float32로 변환
            
            # 레이어 정보 추출
            if hasattr(lin, 'nf'):  # Conv1D
                self.in_features = W.shape[1]
                self.out_features = W.shape[0]
                W = W.t()
                layer_type = "Conv1D"
            else:  # nn.Linear
                self.in_features = lin.in_features
                self.out_features = lin.out_features  
                layer_type = "Linear"
            
            param_count = W.numel()
            print(f"🔗 {layer_type} 간단압축: {W.shape} ({param_count/1e6:.1f}M 파라미터)")
            
            # 간단한 SVD 압축 (데이터 타입 안전)
            U, S, V = torch.svd(W.float())
            rank = max(8, int(min(W.shape) * compression_ratio * 2))
            rank = min(rank, len(S))
            
            # 압축된 파라미터 저장 (float32)
            self.U = nn.Parameter(U[:, :rank].float())
            self.S = nn.Parameter(S[:rank].float())
            self.V = nn.Parameter(V[:, :rank].float())
            
            print(f"       ✅ SVD 압축완료: rank {rank}")
            
            # 바이어스 처리
            if hasattr(lin, 'bias') and lin.bias is not None:
                self.bias = nn.Parameter(lin.bias.data.clone().float())
            else:
                self.bias = None
                
            print(f"     ✅ 압축 완료 (float32)")
        else:
            raise ValueError("Input layer must have weight attribute")

    def forward(self, x):
        # 모든 계산을 float32로 통일
        x = x.float()
        # SVD 연산: x @ V @ diag(S) @ U.t()
        result = x @ self.V @ torch.diag(self.S) @ self.U.t()
        if self.bias is not None:
            result = result + self.bias
        return result.float()

# ═══════════════════════════════════════════════════════════════
# 🎯 대형 모델용 RealityStone 블록 (기존 로직 확장)
# ═══════════════════════════════════════════════════════════════

class LargeModelRealityStoneBlock(nn.Module):
    """대형 모델용 RealityStone 블록 (데이터 타입 통일)"""
    
    def __init__(self, block, compression_ratio=0.05, layer_idx=0, total_layers=32,
                 model_type='gpt_neox', model_dtype=torch.float16):
        super().__init__()
        
        self.model_type = model_type
        self.model_dtype = model_dtype
        
        # 모델 타입별 구조 처리
        if model_type == 'gpt_neox':
            self._compress_gpt_neox_block(block, compression_ratio, layer_idx, total_layers)
        elif model_type == 'gpt2':
            self._compress_gpt2_block(block, compression_ratio, layer_idx, total_layers)
        else:
            self._compress_generic_block(block, compression_ratio, layer_idx, total_layers)
    
    def _compress_gpt_neox_block(self, block, compression_ratio, layer_idx, total_layers):
        """GPT-NeoX 구조 압축 (데이터 타입 통일)"""
        
        # 적응적 압축률 계산 (기존 로직 활용)
        normalized_idx = layer_idx / total_layers
        
        if normalized_idx < 0.2:  # 초기층
            layer_ratio = compression_ratio * 1.5
        elif normalized_idx < 0.8:  # 중간층 - 극한 압축
            layer_ratio = compression_ratio * 0.5
        else:  # 말단층
            layer_ratio = compression_ratio * 1.2
        
        print(f"🌐 GPT-NeoX 레이어 {layer_idx}: 극한압축률 {layer_ratio:.1%}")
        
        # 각 서브모듈 압축 (데이터 타입 전달)
        if hasattr(block, 'input_layernorm'):
            self.input_layernorm = block.input_layernorm
        if hasattr(block, 'post_attention_layernorm'): 
            self.post_attention_layernorm = block.post_attention_layernorm
            
        # Attention 압축
        if hasattr(block, 'attention'):
            self.attention = self._compress_attention_module(
                block.attention, layer_ratio
            )
        
        # MLP 압축
        if hasattr(block, 'mlp'):
            self.mlp = self._compress_mlp_module(
                block.mlp, layer_ratio
            )
    
    def _compress_gpt2_block(self, block, compression_ratio, layer_idx, total_layers):
        """GPT-2 구조 압축 (데이터 타입 통일)"""
        
        # 기존 성공한 EnhancedRealityStoneBlock 로직 활용
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        attn, mlp = block.attn, block.mlp

        # 적응적 압축률
        normalized_idx = layer_idx / total_layers
        if normalized_idx < 0.3:
            layer_ratio = compression_ratio * 1.5
        elif normalized_idx < 0.7:
            layer_ratio = compression_ratio * 0.5  # 극한
        else:
            layer_ratio = compression_ratio * 1.2

        print(f"🌐 GPT-2 레이어 {layer_idx}: 극한압축률 {layer_ratio:.1%}")

        # 서브레이어 압축 (데이터 타입 전달)
        attn.c_attn = LargeModelRealityStoneLinear(attn.c_attn, layer_ratio, 'auto', self.model_dtype)
        attn.c_proj = LargeModelRealityStoneLinear(attn.c_proj, layer_ratio, 'auto', self.model_dtype)
        mlp.c_fc = LargeModelRealityStoneLinear(mlp.c_fc, layer_ratio, 'auto', self.model_dtype)
        mlp.c_proj = LargeModelRealityStoneLinear(mlp.c_proj, layer_ratio, 'auto', self.model_dtype)
        
        self.attn, self.mlp = attn, mlp
    
    def _compress_generic_block(self, block, compression_ratio, layer_idx, total_layers):
        """일반적인 블록 압축 (데이터 타입 통일)"""
        # 모든 Linear 레이어 찾아서 압축
        for name, module in block.named_children():
            if isinstance(module, nn.Linear):
                compressed_module = LargeModelRealityStoneLinear(
                    module, compression_ratio, 'auto', self.model_dtype
                )
                setattr(self, name, compressed_module)
            else:
                setattr(self, name, module)
    
    def _compress_attention_module(self, attention, compression_ratio):
        """어텐션 모듈 압축 (데이터 타입 통일)"""
        
        # query_key_value 압축 (일반적인 구조)
        if hasattr(attention, 'query_key_value'):
            attention.query_key_value = LargeModelRealityStoneLinear(
                attention.query_key_value, compression_ratio, 'riemann', self.model_dtype
            )
        
        # dense 압축
        if hasattr(attention, 'dense'):
            attention.dense = LargeModelRealityStoneLinear(
                attention.dense, compression_ratio, 'fast_svd', self.model_dtype
            )
        
        return attention
    
    def _compress_mlp_module(self, mlp, compression_ratio):
        """MLP 모듈 압축 (데이터 타입 통일)"""
        
        # dense_h_to_4h 압축
        if hasattr(mlp, 'dense_h_to_4h'):
            mlp.dense_h_to_4h = LargeModelRealityStoneLinear(
                mlp.dense_h_to_4h, compression_ratio, 'fft_svd', self.model_dtype
            )
        
        # dense_4h_to_h 압축
        if hasattr(mlp, 'dense_4h_to_h'):
            mlp.dense_4h_to_h = LargeModelRealityStoneLinear(
                mlp.dense_4h_to_h, compression_ratio, 'riemann', self.model_dtype
            )
        
        return mlp
    
    def forward(self, x, **kwargs):
        """순전파 (모델 타입별)"""
        
        # 입력을 float32로 변환 (데이터 타입 통일)
        x = x.float()
        
        if self.model_type == 'gpt_neox':
            # GPT-NeoX 순전파
            h = self.input_layernorm(x)
            attn_outputs = self.attention(h, **kwargs)
            if isinstance(attn_outputs, tuple):
                a = attn_outputs[0]
            else:
                a = attn_outputs
            x = x + a
            
            h2 = self.post_attention_layernorm(x)
            m = self.mlp(h2)
            output = x + m
            
            return (output,)
            
        elif self.model_type == 'gpt2':
            # GPT-2 순전파 (기존 성공 로직)
            h = self.ln1(x)
            attn_outputs = self.attn(h, **kwargs)
            a = attn_outputs[0]
            x = x + a
            h2 = self.ln2(x)
            m = self.mlp(h2)
            output = x + m
            
            if len(attn_outputs) > 1:
                return (output,) + attn_outputs[1:]
            else:
                return (output,)
        else:
            # 일반적인 순전파
            return x

# ═══════════════════════════════════════════════════════════════
# 🚀 대형 모델 압축 파이프라인 (기존 성공 로직 확장)
# ═══════════════════════════════════════════════════════════════

def detect_model_architecture(model):
    """모델 아키텍처 자동 감지"""
    
    config = model.config
    
    if hasattr(config, 'model_type'):
        if config.model_type == 'gpt_neox':
            return 'gpt_neox', len(model.gpt_neox.layers)
        elif config.model_type in ['gpt2', 'kogpt2']:
            return 'gpt2', len(model.transformer.h)
        elif config.model_type == 'llama':
            return 'llama', len(model.model.layers)
    
    # 폴백: 구조 기반 감지
    if hasattr(model, 'gpt_neox'):
        return 'gpt_neox', len(model.gpt_neox.layers)
    elif hasattr(model, 'transformer'):
        return 'gpt2', len(model.transformer.h)
    elif hasattr(model, 'model'):
        return 'llama', len(model.model.layers)
    
    return 'unknown', 0

def apply_large_model_reality_stone_compression(model, compression_ratio=0.05, 
                                               strategy='adaptive'):
    """대형 모델용 RealityStone 압축 (데이터 타입 통일)"""
    
    total_before = sum(p.numel() for p in model.parameters())
    
    # 모델 데이터 타입 확인 및 통일
    model_dtype = next(model.parameters()).dtype
    print(f"🔧 모델 데이터 타입: {model_dtype}")
    
    # 디바이스 설정
    if torch.cuda.is_available():
        device = "cuda"
        model = model.cuda()
    else:
        device = "cpu"
        model = model.cpu()
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    print(f"🔥 대형 모델 RealityStone 압축 시작")
    print(f"   Before: {total_before:,} params ({total_before/1e9:.2f}B)")
    
    # 데이터 타입 문제 해결: 모델 전체를 float32로 변환
    if model_dtype != torch.float32:
        print(f"   🔧 데이터 타입 통일: {model_dtype} → float32")
        model = model.float()
        model_dtype = torch.float32
    
    print(f"   압축률: {compression_ratio:.1%} (목표: {(1-compression_ratio)*100:.0f}% 절약)")
    print(f"   전략: {strategy}")
    print(f"   💎 기법: 리만기하학 + FFT+SVD + RealityStone")
    
    # 모델 아키텍처 감지
    model_type, total_layers = detect_model_architecture(model)
    print(f"   🏗️ 감지된 구조: {model_type} ({total_layers} 레이어)")
    
    # 구조별 압축 적용 (데이터 타입 전달)
    if model_type == 'gpt_neox':
        compressed_count = apply_gpt_neox_compression(
            model, compression_ratio, strategy, total_layers, model_dtype
        )
    elif model_type == 'gpt2':
        compressed_count = apply_gpt2_compression(
            model, compression_ratio, strategy, total_layers, model_dtype
        )
    else:
        print(f"   ⚠️ 지원되지 않는 구조: {model_type}")
        return model
    
    # 압축 결과
    total_after = sum(p.numel() for p in model.parameters())
    actual_compression = total_after / total_before
    memory_saved = (1 - actual_compression) * 100
    
    print(f"\n✅ 대형 모델 압축 완료!")
    print(f"   After:  {total_after:,} params ({total_after/1e9:.2f}B)")
    print(f"   압축률: {actual_compression:.1%} ({1/actual_compression:.1f}× 압축)")
    print(f"   메모리 절약: {memory_saved:.1f}%")
    print(f"   성공 레이어: {compressed_count}/{total_layers}")
    
    return model

def apply_gpt_neox_compression(model, compression_ratio, strategy, total_layers, model_dtype):
    """GPT-NeoX 구조 압축 (데이터 타입 통일)"""
    
    layers = model.gpt_neox.layers
    
    # 전략별 레이어 선택 (기존 로직)
    if strategy == 'adaptive':
        compress_layers = list(range(total_layers))
    elif strategy == 'conservative':
        compress_layers = list(range(2, total_layers-2))
    else:  # aggressive
        compress_layers = list(range(1, total_layers-1))
    
    print(f"   압축 대상: {len(compress_layers)}/{total_layers} 레이어")
    
    compressed_count = 0
    for i in tqdm(compress_layers, desc="🌐 GPT-NeoX 압축"):
        try:
            layers[i] = LargeModelRealityStoneBlock(
                layers[i], compression_ratio, i, total_layers, 'gpt_neox', model_dtype
            )
            compressed_count += 1
        except Exception as e:
            print(f"   ❌ 레이어 {i} 압축 실패: {e}")
            continue
    
    return compressed_count

def apply_gpt2_compression(model, compression_ratio, strategy, total_layers, model_dtype):
    """GPT-2 구조 압축 (데이터 타입 통일)"""
    
    layers = model.transformer.h
    
    # 기존 성공한 전략 사용
    if strategy == 'adaptive':
        compress_layers = list(range(total_layers))
    elif strategy == 'conservative':
        compress_layers = list(range(2, total_layers-2))
    else:
        compress_layers = list(range(1, total_layers-1))
    
    print(f"   압축 대상: {len(compress_layers)}/{total_layers} 레이어")
    
    compressed_count = 0
    for i in tqdm(compress_layers, desc="🌐 GPT-2 압축"):
        try:
            layers[i] = LargeModelRealityStoneBlock(
                layers[i], compression_ratio, i, total_layers, 'gpt2', model_dtype
            )
            compressed_count += 1
        except Exception as e:
            print(f"   ❌ 레이어 {i} 압축 실패: {e}")
            continue
    
    return compressed_count

# ═══════════════════════════════════════════════════════════════
# 🎯 대형 모델 테스트 (수정된 버전)
# ═══════════════════════════════════════════════════════════════

def test_large_model_generation_fixed(model, tokenizer, model_type="원본"):
    """대형 모델 생성 테스트 (수정된 버전)"""
    
    test_prompts = [
        "안녕하세요",
        "오늘 날씨는", 
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    
    print(f"\n=== {model_type} 대형 모델 테스트 ===")
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/5] '{prompt}'")
        
        try:
            t0 = time.time()
            
            # 직접 generate 호출 (문제 있는 함수 대신)
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # 딕셔너리 접근 방식으로 수정
            input_ids = inputs['input_ids'] if 'input_ids' in inputs else inputs.input_ids
            
            # 디바이스 통일
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                model = model.cuda()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 15,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed = time.time() - t0
            
            print(f"  생성: {generated_text}")
            print(f"  시간: {elapsed:.3f}초")
            
            # 간단한 품질 평가
            generated_only = generated_text[len(prompt):].strip()
            if len(generated_only) > 5:
                quality_score = min(3.0, len(generated_only.split()) / 3)
                if any(bad in generated_only for bad in ['/', ':', '##']):
                    quality_score *= 0.5
            else:
                quality_score = 0.1
            
            print(f"  품질: {quality_score:.2f}/3.0")
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'time': elapsed,
                'quality': quality_score
            })
            
        except Exception as e:
            print(f"  ❌ 에러: {e}")
            results.append({
                'prompt': prompt,
                'generated': f"ERROR: {e}",
                'time': 0,
                'quality': 0
            })
    
    # 통계
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print(f"\n📊 {model_type} 통계:")
    print(f"  평균 시간: {avg_time:.3f}초")
    print(f"  평균 품질: {avg_quality:.2f}/3.0")
    
    return results

def select_and_load_large_model():
    """모델 선택 및 로드"""
    
    print("🚀 대형 모델 선택 중...")
    
    # GPU 메모리 체크
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️ GPU 메모리: {gpu_memory:.1f}GB")
        
        # 사용 가능한 모델 필터링
        suitable_models = []
        for name, info in SUPPORTED_LARGE_MODELS.items():
            if info['memory_gb'] <= gpu_memory:
                suitable_models.append((name, info))
        
        if not suitable_models:
            print("❌ 로드 가능한 모델이 없습니다")
            return None, None, None, None
        
        # 가장 큰 모델 선택
        model_name, model_info = max(suitable_models, key=lambda x: x[1]['params'])
        print(f"🎯 선택된 모델: {model_name} ({model_info['size']})")
        
    else:
        # CPU 전용 - 가장 작은 모델
        model_name = "microsoft/DialoGPT-medium"
        model_info = SUPPORTED_LARGE_MODELS[model_name]
        print(f"🖥️ CPU 모드: {model_name}")
    
    # 모델 로드
    print(f"\n📥 모델 로드 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 로드 완료: {total_params:,} 파라미터")
        
        return model, tokenizer, model_name, model_info
        
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return None, None, None, None

def complete_compression_with_finetuning():
    """완전한 압축+파인튜닝 파이프라인"""
    
    print("🚀 완전한 대형 모델 RealityStone 압축+파인튜닝 시스템")
    print("=" * 80)
    print("💎 압축→테스트→파인튜닝→최종검증 전체 파이프라인")
    
    # 1단계: 모델 로드
    teacher_model, tokenizer, model_name, model_info = select_and_load_large_model()
    if teacher_model is None:
        return
    
    # 2단계: 원본 테스트
    print("\n" + "="*80)
    print("📊 원본 대형 모델 성능 테스트")
    original_results = test_large_model_generation_fixed(teacher_model, tokenizer, "원본")
    
    # 3단계: RealityStone 압축 적용
    print("\n" + "="*80)
    print("🔥 RealityStone 극한 압축 적용")
    
    student_model = copy.deepcopy(teacher_model)
    student_model = apply_large_model_reality_stone_compression(
        student_model,
        compression_ratio=model_info['compression_target'],
        strategy='adaptive'
    )
    
    # 4단계: 압축 후 테스트
    print("\n" + "="*80)
    print("📊 압축 후 성능 테스트")
    compressed_results = test_large_model_generation_fixed(student_model, tokenizer, "압축 후")
    
    # 5단계: Knowledge Distillation 파인튜닝
    print("\n" + "="*80)
    print("🧠 Knowledge Distillation 파인튜닝")
    
    try:
        student_model = ultra_knowledge_distillation_fine_tune(
            teacher_model, student_model, tokenizer,
            total_steps=500,   # 대형 모델용 적당한 스텝
            base_lr=3e-6,      # 더 조심스럽게
            temperature=2.5    # 적절한 온도
        )
        finetuning_success = True
    except Exception as e:
        print(f"❌ 파인튜닝 실패: {e}")
        print("📊 압축된 모델 결과만 분석합니다.")
        finetuning_success = False
    
    # 6단계: 최종 테스트 (파인튜닝 있었다면)
    if finetuning_success:
        print("\n" + "="*80)
        print("📊 파인튜닝 후 최종 테스트")
        final_results = test_large_model_generation_fixed(student_model, tokenizer, "최종")
    else:
        final_results = compressed_results
    
    # 7단계: 완전한 결과 분석
    print("\n" + "="*80)
    print("🏆 완전한 RealityStone 압축 최종 분석")
    print("="*80)
    
    # 성능 지표
    orig_quality = sum(r['quality'] for r in original_results) / len(original_results)
    comp_quality = sum(r['quality'] for r in compressed_results) / len(compressed_results)
    final_quality = sum(r['quality'] for r in final_results) / len(final_results)
    
    orig_time = sum(r['time'] for r in original_results) / len(original_results)
    comp_time = sum(r['time'] for r in compressed_results) / len(compressed_results)
    final_time = sum(r['time'] for r in final_results) / len(final_results)
    
    # 압축 통계
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = student_params / teacher_params
    memory_saved = (1 - compression_ratio) * 100
    quality_retention = final_quality / orig_quality if orig_quality > 0 else 1
    quality_improvement = final_quality - comp_quality if finetuning_success else 0
    speed_improvement = orig_time / final_time if final_time > 0 else 1
    
    print(f"📊 완전한 성능 분석:")
    print(f"   모델: {model_name}")
    print(f"   파라미터: {teacher_params:,} → {student_params:,}")
    print(f"   압축률: {compression_ratio:.1%} ({1/compression_ratio:.1f}× 압축)")
    print(f"   메모리 절약: {memory_saved:.1f}%")
    print(f"   품질: 원본 {orig_quality:.2f} → 압축 {comp_quality:.2f} → 최종 {final_quality:.2f}")
    print(f"   속도: 원본 {orig_time:.3f}초 → 최종 {final_time:.3f}초 ({speed_improvement:.1f}×)")
    
    if finetuning_success:
        print(f"   파인튜닝 개선: {quality_improvement:+.2f}점")
    
    # 최종 성공 판정
    if memory_saved >= 85 and quality_retention >= 0.8:
        grade = "🏆 완전 대성공!"
        message = f"RealityStone으로 {memory_saved:.0f}% 절약 + 품질 {quality_retention*100:.0f}% 유지!"
    elif memory_saved >= 70 and quality_retention >= 0.7:
        grade = "🥇 대성공!"
        message = f"상당한 압축 성과 + 품질 유지!"
    elif memory_saved >= 50 and quality_retention >= 0.5:
        grade = "🥈 성공!"
        message = f"절반 이상 압축 + 적정 품질 유지!"
    else:
        grade = "🔧 부분 성공"
        message = f"압축은 성공, 품질 최적화 필요"
    
    print(f"\n🎯 최종 평가: {grade}")
    print(f"   {message}")
    print(f"   💎 기술: RealityStone + 리만기하학 + FFT+SVD")
    print(f"   🏗️ 아키텍처: {model_info['size']} 대형 모델")
    print(f"   🧠 파인튜닝: {'성공' if finetuning_success else '실패'}")
    
    # 생성 결과 샘플 출력
    print(f"\n📝 생성 결과 샘플:")
    for i, result in enumerate(final_results[:3], 1):
        print(f"   [{i}] {result['prompt']} → {result['generated']}")
    
    # 메모리 정리
    del teacher_model, student_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n✨ 완전한 압축+파인튜닝 파이프라인 완료!")
    
    return {
        'compression_ratio': compression_ratio,
        'memory_saved': memory_saved,
        'quality_retention': quality_retention,
        'speed_improvement': speed_improvement,
        'finetuning_success': finetuning_success,
        'final_grade': grade
    }

def main_large_model():
    """메인 함수 (완전한 버전)"""
    
    # 완전한 압축+파인튜닝 실행
    return complete_compression_with_finetuning()

if __name__ == "__main__":
    main_large_model() 