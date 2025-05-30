import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import Counter
import copy

try:
    import reality_stone as rs
    print("✅ RealityStone 라이브러리 로드 성공!")
    RS_AVAILABLE = True
except ImportError:
    print("⚠️ RealityStone 라이브러리 없음 - 기본 압축 사용")
    RS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# 🎯 보수적 고품질 압축기 (품질 우선)
# ═══════════════════════════════════════════════════════════════

class ConservativeSVDCompressor:
    """보수적 SVD 압축기 (품질 우선, 안전한 압축률)"""
    
    def __init__(self, W: torch.Tensor, compression_ratio=0.3):
        """
        Args:
            W: 가중치 행렬 [out_f, in_f] 
            compression_ratio: 보수적 압축률 (30% = 70% 파라미터 유지)
        """
        self.out_f, self.in_f = W.shape
        self.compression_ratio = compression_ratio
        
        print(f"    🔧 보수적 압축: {W.shape}, 압축률={compression_ratio:.1%}")
        
        self._apply_conservative_compression(W)
    
    def _apply_conservative_compression(self, W: torch.Tensor):
        """보수적 SVD 압축 (품질 우선)"""
        
        # SVD 분해
        U, S, V = torch.svd(W.float())
        
        # 에너지 기반 랭크 선택 (95% 에너지 보존)
        energy_cumsum = torch.cumsum(S**2, dim=0)
        total_energy = energy_cumsum[-1]
        energy_threshold = total_energy * 0.95  # 95% 에너지 보존
        
        energy_rank = torch.sum(energy_cumsum < energy_threshold).item() + 1
        
        # 보수적 랭크 (더 많은 파라미터 유지)
        conservative_rank = max(
            min(W.shape) // 2,  # 최소 절반은 유지
            int(min(W.shape) * (1 - self.compression_ratio))  # 보수적 계산
        )
        
        # 최종 랭크 (에너지와 보수적 중 큰 값)
        final_rank = min(max(energy_rank, conservative_rank), len(S))
        
        # 압축된 파라미터 저장
        self.U = nn.Parameter(U[:, :final_rank].to(W.dtype))
        self.S = nn.Parameter(S[:final_rank].to(W.dtype))
        self.V = nn.Parameter(V[:, :final_rank].to(W.dtype))
        
        # 압축 통계
        original_params = W.numel()
        compressed_params = self.U.numel() + self.S.numel() + self.V.numel()
        actual_ratio = compressed_params / original_params
        
        print(f"       ✅ 보수적 압축: rank {final_rank}, 실제 압축률 {actual_ratio:.1%}")
        print(f"          에너지 보존: {95:.0f}%, 파라미터 유지: {actual_ratio*100:.0f}%")
    
    def reconstruct(self) -> torch.Tensor:
        """압축된 가중치 복원"""
        return self.U @ torch.diag(self.S) @ self.V.t()
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """압축된 연산 적용"""
        # 3단계 효율적 계산
        step1 = x @ self.V  # [batch, rank]
        step2 = step1 * self.S.unsqueeze(0)  # [batch, rank]
        step3 = step2 @ self.U.t()  # [batch, out_features]
        return step3

# ═══════════════════════════════════════════════════════════════
# 🎯 보수적 Linear 레이어 (Conv1D 지원)
# ═══════════════════════════════════════════════════════════════

class ConservativeLinear(nn.Module):
    """보수적 Linear 레이어 (품질 우선)"""
    
    def __init__(self, lin, compression_ratio=0.3):
        super().__init__()
        
        if hasattr(lin, 'weight'):
            W = lin.weight.data.clone()
            
            # Conv1D 처리
            if hasattr(lin, 'nf'):  # GPT2 Conv1D
                self.in_features = W.shape[0]  # [768, 2304] 형태
                self.out_features = W.shape[1]
                self.is_conv1d = True
                # Conv1D는 이미 전치되어 있으므로 압축을 위해 다시 전치
                W = W.t()  # [out_features, in_features]로 변환
                print(f"🔧 Conv1D 보수적압축: in={self.in_features}, out={self.out_features}")
            else:  # nn.Linear
                self.in_features = lin.in_features
                self.out_features = lin.out_features
                self.is_conv1d = False
                print(f"🔧 Linear 보수적압축: in={self.in_features}, out={self.out_features}")
            
            # 보수적 압축기 적용
            self.compressor = ConservativeSVDCompressor(W, compression_ratio)
            
            # 바이어스 처리
            if hasattr(lin, 'bias') and lin.bias is not None:
                self.bias = nn.Parameter(lin.bias.data.clone())
            else:
                self.bias = None
        else:
            raise ValueError("Input layer must have weight attribute")
    
    def forward(self, x):
        if self.is_conv1d:
            # Conv1D: weight를 복원하고 전치하여 사용
            W_compressed = self.compressor.reconstruct()  # [out_f, in_f]
            W_conv1d = W_compressed.t()  # [in_f, out_f] Conv1D 형태
            out = x @ W_conv1d
        else:
            # Linear: 직접 적용
            out = self.compressor.apply(x)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out

# ═══════════════════════════════════════════════════════════════
# 🎯 보수적 Block (레이어별 차별 압축)
# ═══════════════════════════════════════════════════════════════

class ConservativeBlock(nn.Module):
    """보수적 Block (레이어별 차별 압축)"""
    
    def __init__(self, block, base_ratio=0.25, layer_idx=0, total_layers=12):
        super().__init__()
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        attn, mlp = block.attn, block.mlp
        
        # 레이어별 차별 압축률 (중요 레이어는 덜 압축)
        layer_ratio = self._get_layer_compression_ratio(layer_idx, total_layers, base_ratio)
        
        print(f"🔧 보수적 레이어 {layer_idx}: 압축률 {layer_ratio:.1%}")
        
        # 각 서브레이어 압축
        attn.c_attn = ConservativeLinear(attn.c_attn, layer_ratio)
        attn.c_proj = ConservativeLinear(attn.c_proj, layer_ratio)
        mlp.c_fc = ConservativeLinear(mlp.c_fc, layer_ratio)
        mlp.c_proj = ConservativeLinear(mlp.c_proj, layer_ratio)
        
        self.attn, self.mlp = attn, mlp
    
    def _get_layer_compression_ratio(self, layer_idx, total_layers, base_ratio):
        """레이어별 차별 압축률"""
        
        normalized_idx = layer_idx / total_layers
        
        if layer_idx == 0 or layer_idx == total_layers - 1:
            # 첫째/마지막 레이어: 가장 보수적
            return base_ratio * 0.5
        elif normalized_idx < 0.3 or normalized_idx > 0.7:
            # 앞쪽/뒤쪽 레이어: 보수적
            return base_ratio * 0.7
        else:
            # 중간 레이어: 기본 압축
            return base_ratio
    
    def forward(self, x, **kwargs):
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

# ═══════════════════════════════════════════════════════════════
# 🎯 보수적 압축 파이프라인
# ═══════════════════════════════════════════════════════════════

def apply_conservative_compression(model, compression_ratio=0.25):
    """보수적 압축 파이프라인 (품질 우선)"""
    
    total = sum(p.numel() for p in model.parameters())
    total_layers = len(model.transformer.h)
    
    print(f"Before: {total:,} params ({total/1e6:.1f}M)")
    print(f"🔧 보수적 압축: 목표={compression_ratio:.1%} (품질 우선)")
    
    # 보수적 레이어 선택 (가장자리 보존)
    compress_layers = list(range(1, total_layers-1))  # 첫째/마지막 제외
    
    print(f"   압축 대상: {len(compress_layers)}/{total_layers} 레이어")
    
    # 보수적 압축 진행
    for i in tqdm(compress_layers, desc="🔧 보수적 압축"):
        if i < len(model.transformer.h):
            try:
                model.transformer.h[i] = ConservativeBlock(
                    model.transformer.h[i], compression_ratio, i, total_layers
                )
            except Exception as e:
                print(f"   ⚠️ 레이어 {i} 압축 실패: {e}")
                continue
    
    total2 = sum(p.numel() for p in model.parameters())
    actual_compression = total2 / total
    
    print(f"After:  {total2:,} params ({total2/1e6:.1f}M)")
    print(f"🔧 실제 압축률: {actual_compression:.1%} ({1/actual_compression:.1f}× 압축)")
    print(f"✅ 메모리 절약: {(1-actual_compression)*100:.1f}%")
    
    return model

# ═══════════════════════════════════════════════════════════════
# 🎯 정확한 품질 평가 (깨진 텍스트 감지)
# ═══════════════════════════════════════════════════════════════

def accurate_quality_evaluation(generated_text, prompt):
    """정확한 한국어 품질 평가 (깨진 텍스트 엄격 감지)"""
    
    generated_only = generated_text[len(prompt):].strip()
    if len(generated_only) < 2:
        return 0.0
    
    score = 3.0  # 시작 점수
    
    # 1. 깨진 텍스트 감지 (가장 중요!)
    broken_patterns = [
        r'[가-힣]{1}[a-zA-Z가-힣]{1}[가-힣]{1}',  # 한글-영어-한글 패턴
        r'티아|티스|르트|병정|살을|베아|괴라|랜홀',      # 이상한 음절 조합
        r'[가-힣]{10,}',                          # 10글자 이상 연속 한글
        r'[ㄱ-ㅎㅏ-ㅣ]',                          # 불완전한 한글
    ]
    
    for pattern in broken_patterns:
        if re.search(pattern, generated_only):
            score -= 2.0  # 깨진 텍스트 발견시 큰 페널티
            break
    
    # 2. 반복 패턴 감지
    char_repeats = len(re.findall(r'(.)\1{2,}', generated_only))
    if char_repeats > 0:
        score -= min(1.0, char_repeats * 0.5)
    
    # 3. 한국어 문법 구조
    korean_endings = ['다', '요', '니다', '해요', '어요', '아요', '네요', '습니다']
    has_proper_ending = any(generated_only.endswith(ending) for ending in korean_endings)
    
    if has_proper_ending:
        score += 0.5
    
    # 4. 의미 있는 단어 포함
    meaningful_words = ['날씨', '좋', '나쁘', '안녕', '감사', '죄송', '오늘', '내일']
    has_meaningful = any(word in generated_only for word in meaningful_words)
    
    if has_meaningful:
        score += 0.3
    
    return max(0.0, min(3.0, score))

# ═══════════════════════════════════════════════════════════════
# 🎯 집중 Knowledge Distillation (더 긴 파인튜닝)
# ═══════════════════════════════════════════════════════════════

def intensive_knowledge_distillation(teacher_model, student_model, tokenizer, 
                                   total_steps=1500, base_lr=5e-6, temperature=2.5):
    """집중 Knowledge Distillation (긴 파인튜닝)"""
    
    print(f"\n🧠 집중 Knowledge Distillation 파인튜닝")
    print(f"   📊 스텝: {total_steps}, 학습률: {base_lr}, 온도: {temperature}")
    
    # 체계적인 한국어 훈련 데이터
    train_texts = [
        # 기본 인사 (완벽한 문장)
        "안녕하세요.", "안녕하세요. 반갑습니다.", "좋은 아침입니다.", "좋은 저녁입니다.",
        "안녕히 가세요.", "안녕히 계세요.", "감사합니다.", "고맙습니다.",
        
        # 날씨 표현 (완벽한 문장)
        "오늘 날씨가 맑습니다.", "오늘 날씨가 흐립니다.", "오늘 날씨가 좋습니다.",
        "비가 옵니다.", "눈이 옵니다.", "바람이 붑니다.", "날씨가 춥습니다.",
        
        # 일상 표현 (완벽한 문장) 
        "밥을 먹었습니다.", "물을 마셨습니다.", "책을 읽었습니다.", "공부를 했습니다.",
        "운동을 했습니다.", "음악을 들었습니다.", "영화를 봤습니다.", "쇼핑을 했습니다.",
        
        # 감정 표현 (완벽한 문장)
        "기분이 좋습니다.", "기분이 나쁩니다.", "행복합니다.", "슬픕니다.",
        "즐겁습니다.", "피곤합니다.", "편안합니다.", "신납니다.",
        
        # 질문과 응답 (완벽한 문장)
        "어떻게 지내세요?", "뭐 하세요?", "어디 가세요?", "언제 오세요?",
        "네, 맞습니다.", "아니요, 틀렸습니다.", "잘 모르겠습니다.", "알겠습니다.",
        
        # 복합 문장 (자연스러운)
        "오늘 날씨가 좋아서 산책을 했습니다.", "친구와 함께 영화를 봤습니다.",
        "도서관에서 책을 읽었습니다.", "맛있는 음식을 먹었습니다.",
    ]
    
    teacher_model.eval()
    student_model.train()
    
    # 단계별 학습률 스케줄
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=base_lr*0.1)
    
    total_loss = 0.0
    best_loss = float('inf')
    patience_count = 0
    
    progress_bar = tqdm(range(total_steps), desc="🧠 집중 파인튜닝")
    
    for step in progress_bar:
        # 다양한 텍스트 순환
        text = train_texts[step % len(train_texts)]
        inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding=True)
        
        if inputs.input_ids.shape[1] < 3:
            continue
            
        input_ids = inputs.input_ids
        labels = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]
        
        optimizer.zero_grad()
        
        # Teacher와 Student 출력
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids)
        
        student_outputs = student_model(input_ids)
        
        # Knowledge Distillation Loss
        teacher_probs = F.softmax(teacher_outputs.logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_outputs.logits / temperature, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
        # Language Model Loss  
        lm_loss = F.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)), 
            labels.view(-1), 
            ignore_index=-100
        )
        
        # 가중 손실 (초기에는 KD 위주, 후반에는 LM 위주)
        kd_weight = max(0.5, 0.9 - (step / total_steps) * 0.4)  # 0.9 → 0.5
        lm_weight = 1 - kd_weight
        
        total_loss_step = kd_weight * kd_loss + lm_weight * lm_loss
        total_loss += total_loss_step.item()
        
        # 역전파
        total_loss_step.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 진행 상황 및 조기 종료
        if step % 50 == 0:
            avg_loss = total_loss / (step + 1)
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'kd_w': f'{kd_weight:.2f}',
                'lm_w': f'{lm_weight:.2f}'
            })
            
            # 조기 종료 체크
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > 200:  # 200 스텝 동안 개선 없으면 조기 종료
                    print(f"\n   조기 종료: {step} 스텝에서 최적화 완료")
                    break
    
    print(f"   최종 평균 손실: {total_loss / (step + 1):.4f}")
    print("✅ 집중 Knowledge Distillation 완료!")
    
    return student_model

# ═══════════════════════════════════════════════════════════════
# 🎯 정확한 테스트 함수
# ═══════════════════════════════════════════════════════════════

def accurate_test(model, tokenizer, model_type="모델"):
    """정확한 품질 테스트 (깨진 텍스트 엄격 감지)"""
    
    test_prompts = [
        "안녕하세요",
        "오늘 날씨는", 
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    
    print(f"\n=== {model_type} 정확한 품질 테스트 ===")
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/5] '{prompt}'")
        
        try:
            t0 = time.time()
            
            # 보수적 생성 (품질 우선)
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=25,
                    do_sample=True,
                    temperature=0.7,       # 보수적 온도
                    top_p=0.9,            # 높은 확률 유지
                    top_k=50,             # 더 넓은 선택
                    repetition_penalty=1.5,  # 적당한 반복 방지
                    no_repeat_ngram_size=3,  # 3-gram 반복 방지
                    pad_token_id=tokenizer.eos_token_id,
                    min_length=len(inputs.input_ids[0]) + 3,  # 최소 길이
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed = time.time() - t0
            
            print(f"  생성: {generated_text}")
            print(f"  시간: {elapsed:.3f}초")
            
            # 정확한 품질 평가
            quality_score = accurate_quality_evaluation(generated_text, prompt)
            print(f"  품질: {quality_score:.2f}/3.0")
            
            # 깨진 텍스트 감지
            generated_only = generated_text[len(prompt):].strip()
            is_broken = bool(re.search(r'티아|티스|르트|병정|살을|베아|괴라|랜홀', generated_only))
            if is_broken:
                print(f"  ⚠️ 깨진 텍스트 감지!")
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'time': elapsed,
                'quality': quality_score,
                'is_broken': is_broken
            })
            
        except Exception as e:
            print(f"  ❌ 에러: {e}")
            results.append({
                'prompt': prompt,
                'generated': f"ERROR: {e}",
                'time': 0,
                'quality': 0,
                'is_broken': True
            })
    
    # 통계
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    broken_count = sum(1 for r in results if r['is_broken'])
    
    print(f"\n📊 {model_type} 정확한 통계:")
    print(f"  평균 시간: {avg_time:.3f}초")
    print(f"  평균 품질: {avg_quality:.2f}/3.0")
    print(f"  깨진 텍스트: {broken_count}/5개")
    print(f"  성공률: {(5-broken_count)/5*100:.0f}%")
    
    return results

# ═══════════════════════════════════════════════════════════════
# 🎯 메인 함수 (정확도 최우선)
# ═══════════════════════════════════════════════════════════════

def main():
    model_name = "skt/kogpt2-base-v2"
    print("🎯 정확도 우선 보수적 압축 시스템 v1.0")
    print("=" * 60)
    print("🔧 목표: 품질 유지 + 적당한 압축 + 깨진 텍스트 방지")
    print("Loading model…")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1단계: 원본 모델 테스트
    print("\n" + "="*60)
    print("📊 원본 모델 성능 벤치마크")
    original_results = accurate_test(teacher_model, tokenizer, "원본")

    # 2단계: 보수적 압축
    print("\n" + "="*60)
    print("🔧 보수적 압축 적용 (품질 우선)")
    
    student_model = copy.deepcopy(teacher_model)
    student_model = apply_conservative_compression(
        student_model, 
        compression_ratio=0.25  # 25% 압축 (75% 파라미터 유지)
    )
    
    # 3단계: 압축 직후 테스트
    print("\n" + "="*60)
    print("📊 압축 직후 품질 테스트")
    compressed_results = accurate_test(student_model, tokenizer, "압축후")
    
    # 4단계: 집중 파인튜닝
    print("\n" + "="*60)
    print("🧠 집중 Knowledge Distillation 파인튜닝")
    student_model = intensive_knowledge_distillation(
        teacher_model, student_model, tokenizer,
        total_steps=1500,  # 1500 스텝
        base_lr=5e-6,      # 낮은 학습률
        temperature=2.5    # 적당한 온도
    )
    
    # 5단계: 최종 테스트
    print("\n" + "="*60)
    print("📊 최종 품질 평가")
    final_results = accurate_test(student_model, tokenizer, "최종")
    
    # 6단계: 종합 분석
    print("\n" + "="*60)
    print("🏆 정확도 우선 압축 최종 분석")
    print("="*60)
    
    # 성능 지표
    orig_quality = sum(r['quality'] for r in original_results) / len(original_results)
    orig_broken = sum(1 for r in original_results if r['is_broken'])
    
    comp_quality = sum(r['quality'] for r in compressed_results) / len(compressed_results)
    comp_broken = sum(1 for r in compressed_results if r['is_broken'])
    
    final_quality = sum(r['quality'] for r in final_results) / len(final_results)
    final_broken = sum(1 for r in final_results if r['is_broken'])
    
    # 압축 통계
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = student_params / teacher_params
    
    print(f"📊 성능 비교:")
    print(f"   원본:     품질 {orig_quality:.2f}, 깨진 텍스트 {orig_broken}/5")
    print(f"   압축후:   품질 {comp_quality:.2f}, 깨진 텍스트 {comp_broken}/5")  
    print(f"   최종:     품질 {final_quality:.2f}, 깨진 텍스트 {final_broken}/5")
    
    print(f"\n📈 개선 효과:")
    quality_retention = final_quality / orig_quality
    improvement = final_quality - comp_quality
    print(f"   품질 유지율: {quality_retention*100:.1f}%")
    print(f"   파인튜닝 개선: +{improvement:.2f}점")
    print(f"   텍스트 복구: {comp_broken} → {final_broken} 깨진 텍스트")
    
    print(f"\n💾 압축 성과:")
    print(f"   파라미터: {teacher_params:,} → {student_params:,}")
    print(f"   압축 비율: {compression_ratio:.1%} ({1/compression_ratio:.1f}× 압축)")
    print(f"   메모리 절약: {(1-compression_ratio)*100:.1f}%")
    
    # 최종 판정
    if final_broken == 0 and final_quality >= orig_quality * 0.9:
        grade = "🏆 성공! (A급)"
        message = "깨진 텍스트 없이 90%+ 품질 유지!"
    elif final_broken <= 1 and final_quality >= orig_quality * 0.8:
        grade = "🥇 양호 (B급)"  
        message = "대부분 정상 텍스트, 80%+ 품질 유지"
    elif final_broken <= 2 and final_quality >= orig_quality * 0.7:
        grade = "🥈 보통 (C급)"
        message = "일부 개선 효과 있음"
    else:
        grade = "🔧 개선 필요 (D급)"
        message = "추가 최적화 필요"
    
    print(f"\n{grade}: {message}")
    
    if final_broken > 0:
        print(f"💡 권장사항: 압축률을 더 낮추거나 파인튜닝 연장")
    if quality_retention < 0.85:
        print(f"💡 권장사항: 더 보수적인 압축 전략 적용")
    
    print(f"\n🌟 최종 결론:")
    print(f"   보수적 압축으로 {(1-compression_ratio)*100:.0f}% 메모리 절약하면서")
    print(f"   원본 품질의 {quality_retention*100:.0f}%를 유지했습니다.")
    if final_broken == 0:
        print(f"   ✅ 깨진 텍스트 완전 방지 성공!")

if __name__ == "__main__":
    main() 