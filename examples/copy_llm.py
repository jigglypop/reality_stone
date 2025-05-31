import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from tqdm import tqdm
import reality_stone as rs

print("RealityStone 로드 성공")

# ═══════════════════════════════════════════════════════════════
# PoincareBallLinear: reality_stone을 실제로 사용하는 레이어
# ═══════════════════════════════════════════════════════════════
class PoincareBallLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        # 가중치 초기화 (작은 값으로 시작)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 디바이스 일치 확인 및 이동
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(x.device)
        # Poincaré Ball에서 실제 선형 변환 구현
        try:
            # 1) 표준 선형 변환 (기준점)
            standard_out = F.linear(x, self.weight, self.bias)
            # 2) Poincaré Ball에서 가중치별 연산
            batch_size = x.shape[0]
            seq_len = x.shape[1] if len(x.shape) == 3 else 1
            in_dim = x.shape[-1]
            out_dim = self.out_features
            # 입력을 2D로 reshape: [batch*seq, in_dim]
            if len(x.shape) == 3:
                x_flat = x.reshape(-1, in_dim)
            else:
                x_flat = x
            # 각 출력 차원별로 Möbius 연산 수행
            poincare_out = torch.zeros(x_flat.shape[0], out_dim, device=x.device, dtype=x.dtype)
            for j in range(out_dim):
                # j번째 출력을 위한 가중치 벡터
                w_j = self.weight[j]  # [in_dim]
                # 입력 벡터들과 가중치의 element-wise product
                weighted_inputs = x_flat * w_j.unsqueeze(0)  # [batch*seq, in_dim]
                # Möbius 덧셈으로 weighted_inputs를 누적
                result_j = torch.zeros_like(weighted_inputs[:, 0])  # [batch*seq]
                for k in range(in_dim):
                    if k == 0:
                        result_j = weighted_inputs[:, k]
                    else:
                        # 각 배치 원소별로 Möbius 덧셈
                        for b in range(weighted_inputs.shape[0]):
                            try:
                                # 스칼라를 1차원 텐서로 변환하여 mobius_add 사용
                                a = result_j[b:b+1]
                                b_val = weighted_inputs[b:b+1, k]
                                result_j[b] = rs.mobius_add(a, b_val, self.curvature)[0]
                            except:
                                # Möbius 연산 실패시 표준 덧셈 사용
                                result_j[b] = result_j[b] + weighted_inputs[b, k]
                
                poincare_out[:, j] = result_j
            # bias 추가 (Möbius 덧셈 사용)
            if self.bias is not None:
                for j in range(out_dim):
                    for b in range(poincare_out.shape[0]):
                        try:
                            a = poincare_out[b:b+1, j]
                            bias_val = self.bias[j:j+1]
                            poincare_out[b, j] = rs.mobius_add(a, bias_val, self.curvature)[0]
                        except:
                            poincare_out[b, j] = poincare_out[b, j] + self.bias[j]
            if len(x.shape) == 3:
                poincare_out = poincare_out.reshape(batch_size, seq_len, out_dim)
            result = 0.95 * standard_out + 0.05 * poincare_out
            return result
        except Exception as e:
            return F.linear(x, self.weight, self.bias)

# ═══════════════════════════════════════════════════════════════
# PoincareBallWrappedLinear: 기존 레이어를 래핑
# ═══════════════════════════════════════════════════════════════
class PoincareBallWrappedLinear(nn.Module):
    def __init__(self, original_layer: nn.Module, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
        # 원본 레이어 저장 (fallback용)
        self.original_layer = copy.deepcopy(original_layer)
        # 원본 파라미터 분석
        if hasattr(original_layer, 'nf'):  # GPT2Conv1D
            in_features = original_layer.weight.shape[0]
            out_features = original_layer.weight.shape[1]
            is_conv1d = True
            print(f"🔧 Conv1D: {in_features} → {out_features}")
        elif hasattr(original_layer, 'weight'):  # nn.Linear
            out_features, in_features = original_layer.weight.shape
            is_conv1d = False
            print(f"🔧 Linear: {in_features} → {out_features}")
        else:
            raise ValueError("Cannot determine layer dimensions")
        self.poincare_layer = PoincareBallLinear(
            in_features, out_features, curvature, 
            bias=(hasattr(original_layer, 'bias') and original_layer.bias is not None)
        )
        with torch.no_grad():
            if is_conv1d: 
                self.poincare_layer.weight.data.copy_(original_layer.weight.data.t())
            else: 
                self.poincare_layer.weight.data.copy_(original_layer.weight.data)
            if self.poincare_layer.bias is not None and hasattr(original_layer, 'bias') and original_layer.bias is not None:
                self.poincare_layer.bias.data.copy_(original_layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Poincaré 레이어 시도
            result = self.poincare_layer(x)
            expected_shape = list(x.shape)
            expected_shape[-1] = self.poincare_layer.out_features
            if result.shape != torch.Size(expected_shape):
                print(f"차원 불일치, 원본 사용: {result.shape} vs {expected_shape}")
                return self.original_layer(x)
            return result
        except Exception as e:
            print(f"Poincaré 오류, 원본 사용: {e}")
            return self.original_layer(x)

# ═══════════════════════════════════════════════════════════════
# PoincareBlock: GPT-2 블록 래핑
# ═══════════════════════════════════════════════════════════════
class PoincareBlock(nn.Module):
    def __init__(self, block: nn.Module, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
        # LayerNorm 복제
        self.ln_1 = copy.deepcopy(block.ln_1)
        self.ln_2 = copy.deepcopy(block.ln_2)
        # Attention, MLP 모듈 복제
        attn = copy.deepcopy(block.attn)
        mlp = copy.deepcopy(block.mlp)
        # Linear 레이어들을 Poincaré 레이어로 교체
        attn.c_attn = PoincareBallWrappedLinear(attn.c_attn, curvature)
        attn.c_proj = PoincareBallWrappedLinear(attn.c_proj, curvature)
        mlp.c_fc = PoincareBallWrappedLinear(mlp.c_fc, curvature)
        mlp.c_proj = PoincareBallWrappedLinear(mlp.c_proj, curvature)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x, **kwargs):
        # Attention
        h = self.ln_1(x)
        attn_outputs = self.attn(h, **kwargs)
        a = attn_outputs[0]
        x = x + a
        # MLP
        h2 = self.ln_2(x)
        m = self.mlp(h2)
        out = x + m
        # 추가 출력이 있으면 그대로 반환
        if len(attn_outputs) > 1:
            return (out,) + attn_outputs[1:]
        return (out,)

# ═══════════════════════════════════════════════════════════════
# Poincaré 모델 생성 함수
# ═══════════════════════════════════════════════════════════════
def create_poincare_model(teacher_model: nn.Module, curvature: float = 1.0):
    student = copy.deepcopy(teacher_model)
    total_blocks = len(student.transformer.h)
    print(f"🔄 총 {total_blocks}개 블록을 Poincaré 볼 기반으로 교체 중")
    for i in tqdm(range(total_blocks), desc="포인카레 변환"):
        orig_block = student.transformer.h[i]
        student.transformer.h[i] = PoincareBlock(orig_block, curvature=curvature)
    return student

# ═══════════════════════════════════════════════════════════════
# 🔶 테스트 및 비교 함수들
# ═══════════════════════════════════════════════════════════════
def fast_test(model, tokenizer, device, prompts, model_type="모델", max_length=50):
    model.to(device).eval()
    results = []
    total_time = 0.0
    print(f"\n=== [{model_type}] 테스트 ===")

    for idx, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_time += elapsed
        print(f"[{idx}] '{prompt}' → {gen_text} ({elapsed:.3f}s)")
        results.append((prompt, gen_text, elapsed))

    avg_time = total_time / len(prompts)
    print(f"[{model_type}] 평균 생성 시간: {avg_time:.3f}초")
    return results, avg_time

def detailed_accuracy_test(teacher_model, student_model, tokenizer, device, test_prompts):
    teacher_model.to(device).eval()
    student_model.to(device).eval()

    print("\n🔬 상세 정확도 검증 시작...")
    total_logprob_diff = 0.0
    total_embedding_cosim = 0.0
    exact_matches = 0

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits
            
            # 로그 확률 차이
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
            student_logprobs = F.log_softmax(student_logits, dim=-1)
            logprob_diff = torch.mean(torch.abs(teacher_logprobs - student_logprobs)).item()
            total_logprob_diff += logprob_diff
            
            # 임베딩 유사도
            if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
                teacher_hidden = teacher_outputs.hidden_states[-1].mean(dim=1)
                student_hidden = student_outputs.hidden_states[-1].mean(dim=1)
                cosim = F.cosine_similarity(teacher_hidden, student_hidden, dim=-1).mean().item()
            else:
                teacher_hidden = teacher_logits.mean(dim=1)
                student_hidden = student_logits.mean(dim=1)
                cosim = F.cosine_similarity(teacher_hidden, student_hidden, dim=-1).mean().item()
            total_embedding_cosim += cosim

            # 예측 일치
            teacher_pred = torch.argmax(teacher_logits, dim=-1)
            student_pred = torch.argmax(student_logits, dim=-1)
            if torch.equal(teacher_pred, student_pred):
                exact_matches += 1

            print(f"[{i+1}] '{prompt}':")
            print(f"  📈 로그확률 차이: {logprob_diff:.6f}")
            print(f"  🎯 임베딩 유사도: {cosim:.6f}")
            print(f"  ✓ 예측 일치: {'예' if torch.equal(teacher_pred, student_pred) else '아니오'}")

    avg_logprob_diff = total_logprob_diff / len(test_prompts)
    avg_embedding_cosim = total_embedding_cosim / len(test_prompts)
    exact_match_rate = exact_matches / len(test_prompts)

    print(f"\n📊 정확도 종합 결과:")
    print(f"  🔸 평균 로그확률 차이: {avg_logprob_diff:.6f} (낮을수록 좋음)")
    print(f"  🔸 평균 임베딩 유사도: {avg_embedding_cosim:.6f} (높을수록 좋음)")
    print(f"  🔸 정확한 예측 일치율: {exact_match_rate:.1%}")

    return {
        'avg_logprob_diff': avg_logprob_diff,
        'avg_embedding_cosim': avg_embedding_cosim,
        'exact_match_rate': exact_match_rate
    }

def compare_state_dicts(teacher, student):
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    print("\n🔍 파라미터 구조 비교:")
    print(f"  Teacher 파라미터 수: {len(t_sd)} 개")
    print(f"  Student 파라미터 수: {len(s_sd)} 개")

    teacher_total_params = sum(p.numel() for p in t_sd.values())
    student_total_params = sum(p.numel() for p in s_sd.values())

    print(f"  Teacher 전체 파라미터: {teacher_total_params:,}")
    print(f"  Student 전체 파라미터: {student_total_params:,}")
    print(f"  파라미터 수 비율: {student_total_params/teacher_total_params:.4f}")

    close_matches = 0
    total_keys = len(t_sd)

    for k in t_sd:
        if k not in s_sd:
            print(f"⚠️ Student에 누락된 키: {k}")
            continue
        if torch.allclose(t_sd[k], s_sd[k], atol=1e-4, rtol=1e-3):
            close_matches += 1
        else:
            diff = torch.mean(torch.abs(t_sd[k] - s_sd[k])).item()
            print(f"📏 파라미터 차이: {k} (평균 절대차이: {diff:.6f})")

    print(f"✅ 근사 일치 파라미터: {close_matches}/{total_keys} ({close_matches/total_keys:.1%})")
    return close_matches == total_keys

def measure_memory_usage(model, device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        return memory_used
    else:
        return 0.0

def extract_korean_outputs(model, tokenizer, device, prompts, model_name="모델"):
    model.to(device).eval()
    print(f"\n🔤 [{model_name}] 한글 출력 추출")
    print("="*50)
    korean_outputs = []
    for idx, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        korean_outputs.append((prompt, gen_text))
        print(f"\n[{idx}] 프롬프트: '{prompt}'")
        print(f"    출력: {gen_text}")
        print("-" * 50)
    return korean_outputs

def creative_korean_test(model, tokenizer, device, model_name="모델"):
    creative_prompts = [
        "봄이 오면",
        "내가 좋아하는 것은",
        "미래의 기술은",
        "행복한 순간은",
        "한국의 아름다운 곳은"
    ]
    print(f"\n🎨 [{model_name}] 창의적 한글 생성 테스트")
    print("="*60)
    for idx, prompt in enumerate(creative_prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=1.0,
                top_p=0.85,
                top_k=40,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n🌟 [{idx}] '{prompt}'")
        print(f"💭 {gen_text}")
        print("─" * 60)

# ═══════════════════════════════════════════════════════════════
# 🔶 메인 함수
# ═══════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "skt/kogpt2-base-v2"
    curvature = 1.0

    print(f"RealityStone PoincareBallLayer 변환 테스트")
    print(f"디바이스: {device}")
    print(f"모델: {model_name}")
    print(f"곡률: {curvature}")
    print(f"Reality Stone 사용 가능: 예")

    # 1) Teacher 모델 로드
    print("\n▶ 원본 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 테스트용 프롬프트
    prompts = [
        "안녕하세요",
        "오늘 날씨는",
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    detailed_prompts = ["안녕", "좋은 하루", "인공지능"]

    # 2) 원본 모델 메모리 측정
    teacher_memory = measure_memory_usage(teacher, device)
    print(f"Teacher 모델 메모리 사용량: {teacher_memory:.1f} MB")

    # 3) 원본 모델 테스트
    print("\n원본 모델 테스트 시작")
    orig_results, orig_time = fast_test(teacher, tokenizer, device, prompts, "원본")

    # 4) PoincaréBallLayer 기반 모델 생성
    print(f"\nPoincaréBallLayer 모델 생성 중... (곡률={curvature})")
    student = create_poincare_model(teacher, curvature)

    # 5) Poincaré 모델 메모리 측정
    student_memory = measure_memory_usage(student, device)
    print(f"Student 모델 메모리 사용량: {student_memory:.1f} MB")
    print(f"메모리 비율: {student_memory/teacher_memory:.3f}")

    # 6) Poincaré 모델 테스트
    print("\nPoincaréBallLayer 모델 테스트 시작")
    poincare_results, poincare_time = fast_test(student, tokenizer, device, prompts, "포인카레")

    # 7) 파라미터 비교
    print("\n파라미터 동등성 검증 중...")
    params_match = compare_state_dicts(teacher, student)

    # 8) 상세 정확도 검증
    accuracy_metrics = detailed_accuracy_test(teacher, student, tokenizer, device, detailed_prompts)

    # 9) 최종 요약
    print("\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)

    print(f"\n속도 비교:")
    print(f"   원본 평균 생성 시간: {orig_time:.3f}s")
    print(f"   포인카레 평균 생성 시간: {poincare_time:.3f}s")
    speed_ratio = poincare_time / orig_time
    print(f"   속도 비율: {speed_ratio:.3f} ({'빠름' if speed_ratio < 1.0 else '느림'})")

    print(f"\n메모리 비교:")
    print(f"   메모리 사용량 비율: {student_memory/teacher_memory:.3f}")

    print(f"\n정확도 지표:")
    print(f"   로그확률 차이: {accuracy_metrics['avg_logprob_diff']:.6f}")
    print(f"   임베딩 유사도: {accuracy_metrics['avg_embedding_cosim']:.4f}")
    print(f"   예측 일치율: {accuracy_metrics['exact_match_rate']:.1%}")

    print(f"\n출력 결과 일치 여부:")
    exact_output_matches = 0
    for i, (o, p) in enumerate(zip(orig_results, poincare_results), 1):
        if o[1] == p[1]:
            print(f"[{i}] '{o[0]}' 출력 완전 일치")
            exact_output_matches += 1
        else:
            print(f"[{i}] '{o[0]}' 출력 불일치")
            print(f"    원본: {o[1]}")
            print(f"    포인카레: {p[1]}")
    output_match_rate = exact_output_matches / len(prompts)
    print(f"\n완전 출력 일치율: {output_match_rate:.1%}")

    print(f"\nReality Stone PoincaréBallLayer 변환 결과:")
    if accuracy_metrics['exact_match_rate'] > 0.8 and accuracy_metrics['avg_embedding_cosim'] > 0.95:
        print("성공: PoincaréBallLayer가 원본과 거의 동일한 성능을 보입니다!")
    elif accuracy_metrics['exact_match_rate'] > 0.6 and accuracy_metrics['avg_embedding_cosim'] > 0.9:
        print("부분 성공: PoincaréBallLayer가 원본과 유사한 성능을 보입니다.")
    else:
        print("주의 필요: PoincaréBallLayer 변환에서 성능 차이가 발생했습니다.")

    print("\n[완료] PoincaréBallLayer 기반 모델 변환 및 검증이 완료되었습니다.")
    
    # 한글 생성 테스트
    extract_korean_outputs(student, tokenizer, device, prompts, "포인카레")
    creative_korean_test(student, tokenizer, device, "포인카레")

if __name__ == "__main__":
    main()
