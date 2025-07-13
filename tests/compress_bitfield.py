import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
import copy
from tqdm import tqdm

try:
    import reality_stone as rs
    from reality_stone.layers import BitfieldLinear
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    print("⚠️ RealityStone이 설치되지 않았습니다. 종료합니다.")
    exit(1)


class ProperBitfieldLinear(nn.Module):
    """
    논문에 충실한 비트필드 압축 선형 레이어
    - 가중치 전체를 비트필드로 압축
    - 추론 시 원래 norm을 복원하여 정보 손실 최소화
    """
    def __init__(self, linear_layer: nn.Linear, basis_size: int = 256):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        W = linear_layer.weight.data

        # 1. 각 행의 norm 계산 및 저장
        weight_norms = torch.norm(W, p=2, dim=1)
        self.register_buffer('weight_norms', weight_norms)

        # 2. 가중치 정규화 (0으로 나누는 것 방지)
        W_normalized = W / (weight_norms.unsqueeze(1) + 1e-8)
        
        # 정규화된 가중치로 임시 Linear 레이어 생성
        normalized_linear = nn.Linear(self.in_features, self.out_features, bias=False)
        normalized_linear.weight.data = W_normalized
        
        # 3. 정규화된 가중치를 비트필드로 압축
        # r_max=1.0은 norm이 1 이하인 벡터에 적합
        self.bitfield = BitfieldLinear.from_linear(
            normalized_linear,
            basis_size=basis_size,
            r_max=1.0
        )
        
        # 바이어스 처리
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # 3D 입력을 2D로 변환: (B, S, F_in) -> (B*S, F_in)
        input_shape = x.shape
        if x.dim() == 3:
            x_reshaped = x.reshape(-1, self.in_features)
        else:
            x_reshaped = x
            
        # 2D 텐서로 압축 레이어 실행 (정규화된 가중치 기준)
        output_normalized = self.bitfield(x_reshaped)
        
        # 4. 저장된 norm을 곱하여 스케일 복원
        output_restored = output_normalized * self.weight_norms
        
        # 2D 출력을 원래 3D 차원으로 복원
        if x.dim() == 3:
            batch_size, seq_len, _ = input_shape
            output = output_restored.view(batch_size, seq_len, self.out_features)
        else:
            output = output_restored
            
        # 바이어스 추가
        if self.bias is not None:
            output += self.bias
        return output


def apply_bitfield_compression_to_model(model, basis_size=256, target_layers=None):
    """
    모델의 선형 레이어를 비트필드 압축 레이어로 교체
    
    Args:
        model: 압축할 모델
        basis_size: 기저 벡터 테이블 크기 (논문: 256 권장)
        target_layers: 압축할 레이어 이름 패턴 (None이면 모두 압축)
    """
    print(f"\n🔷 비트필드 압축 시작 (basis_size={basis_size})")
    
    # 교체할 레이어 목록 수집
    layers_to_replace = []
    total_params_before = 0
    total_params_after = 0
    
    for name, module in model.named_modules():
        # Linear 레이어와 transformers의 Conv1D 레이어 대상
        if isinstance(module, (nn.Linear, Conv1D)):
            # target_layers가 None이면 모든 레이어 압축
            if target_layers is None:
                layers_to_replace.append(name)
            # target_layers가 있으면 어느 하나의 패턴이라도 name에 포함되면 압축
            elif any(pattern in name for pattern in target_layers):
                layers_to_replace.append(name)
                # 원본 파라미터 수 계산
                params = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    params += module.bias.numel()
                total_params_before += params
    
    print(f"\n압축 대상 레이어: {len(layers_to_replace)}개")
    
    # 레이어 교체
    replaced_layers = []
    for name in tqdm(layers_to_replace, desc="레이어 압축"):
        module = model.get_submodule(name)
        
        # 최상위 모듈은 건너뜀
        if '.' not in name:
            continue
            
        parent_name, child_name = name.rsplit('.', 1)
        parent_module = model.get_submodule(parent_name)
        
        # Conv1D를 Linear로 변환 (GPT-2 특수 처리)
        if isinstance(module, Conv1D):
            # transformers의 Conv1D는 실제로는 Linear
            # weight shape: [nx, nf] where nx=in_features, nf=out_features
            linear_equiv = nn.Linear(
                module.weight.shape[0],  # nx = in_features
                module.weight.shape[1],  # nf = out_features
                bias=(module.bias is not None)
            )
            # Conv1D의 weight는 이미 [in_features, out_features] 형태
            linear_equiv.weight.data = module.weight.data.t()  # Linear는 [out_features, in_features] 필요
            if module.bias is not None:
                linear_equiv.bias.data = module.bias.data
        else:
            linear_equiv = module
        
        # 비트필드 압축 적용
        compressed_layer = ProperBitfieldLinear(linear_equiv, basis_size)
        setattr(parent_module, child_name, compressed_layer)
        replaced_layers.append((name, compressed_layer))
        
        # 압축 후 파라미터 수 계산 (22비트 × out_features + 기저 테이블 공유)
        compressed_params = compressed_layer.out_features * 3  # 22비트 ≈ 3바이트
        if compressed_layer.bias is not None:
            compressed_params += compressed_layer.bias.numel() * 4  # 바이어스는 float32
        total_params_after += compressed_params
    
    # 기저 테이블 크기 추가 (모든 레이어가 공유)
    if replaced_layers:
        # 첫 번째 압축된 레이어의 in_features 가져오기
        _, first_compressed = replaced_layers[0]
        basis_table_size = basis_size * first_compressed.in_features * 4  # float32
        total_params_after += basis_table_size
    
    compression_ratio = total_params_after / total_params_before if total_params_before > 0 and total_params_after > 0 else 0
    print(f"\n압축 완료:")
    print(f"  - 원본 파라미터: {total_params_before:,} bytes")
    print(f"  - 압축 파라미터: {total_params_after:,} bytes")
    if compression_ratio > 0:
        print(f"  - 압축률: {compression_ratio:.3%} ({1/compression_ratio:.1f}x 압축)")
    else:
        print(f"  - 압축률: 계산 불가")
    
    return model


def test_model(model, tokenizer, device, prompts, model_name, max_length=50):
    """모델 테스트 및 성능 측정"""
    model.to(device).eval()
    results = []
    total_time = 0.0
    
    print(f"\n=== {model_name} 테스트 ===")
    for idx, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=max_length, 
                do_sample=False,  # 결정적 생성
                pad_token_id=tokenizer.eos_token_id
            )
        
        elapsed = time.time() - start
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_time += elapsed
        
        print(f"[{idx}] '{prompt}' -> {gen_text} ({elapsed:.3f}s)")
        results.append((prompt, gen_text, elapsed))
    
    avg_time = total_time / len(prompts) if prompts else 0
    print(f"{model_name} 평균 시간: {avg_time:.3f}초")
    
    return results, avg_time


def calculate_perplexity(model, tokenizer, device, test_text):
    """퍼플렉시티 계산"""
    model.eval()
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 모델 로드
    model_name = "skt/kogpt2-base-v2"
    print(f"\n모델 로드 중: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, use_safetensors=True).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 테스트 프롬프트
    prompts = [
        "안녕하세요",
        "오늘 날씨는", 
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    
    # 퍼플렉시티 테스트용 텍스트
    test_text = """인공지능은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현하려는 
    컴퓨터 과학의 세부 분야이다. 최근 딥러닝 기술의 발전으로 많은 진전이 있었다."""
    
    print("\n" + "="*60)
    print("비트필드 압축 성능 비교")
    print("="*60)
    
    # 원본 모델 크기
    orig_params = sum(p.numel() for p in teacher.parameters())
    print(f"원본 모델 크기: {orig_params:,} 파라미터")
    
    # 1. 비트필드 압축 (전체 레이어)
    print("\n[1] 전체 레이어 비트필드 압축")
    student_full = copy.deepcopy(teacher)
    student_full = apply_bitfield_compression_to_model(student_full, basis_size=256)
    
    comp_full_results, comp_full_time = test_model(student_full, tokenizer, device, prompts, "전체압축")
    comp_full_perplexity = calculate_perplexity(student_full, tokenizer, device, test_text)
    print(f"전체압축 퍼플렉시티: {comp_full_perplexity:.2f}")
    
    # 2. 비트필드 압축 (주요 레이어만)
    print("\n[2] 주요 레이어만 비트필드 압축 (MLP 레이어만)")
    student_partial = copy.deepcopy(teacher)
    # MLP 레이어만 압축 (일반적으로 가장 큰 레이어)
    # GPT-2는 transformer.h.*.mlp.c_fc와 transformer.h.*.mlp.c_proj 패턴
    target_layers = ['mlp.c_']  # MLP의 c_fc와 c_proj만
    student_partial = apply_bitfield_compression_to_model(
        student_partial, 
        basis_size=256,
        target_layers=target_layers
    )
    
    comp_partial_results, comp_partial_time = test_model(
        student_partial, tokenizer, device, prompts, "부분압축"
    )
    comp_partial_perplexity = calculate_perplexity(student_partial, tokenizer, device, test_text)
    print(f"부분압축 퍼플렉시티: {comp_partial_perplexity:.2f}")
    
    # 3. 원본 모델 테스트 (비교용)
    print("\n[3] 원본 모델 테스트 (비교용)")
    orig_results, orig_time = test_model(teacher, tokenizer, device, prompts, "원본")
    orig_perplexity = calculate_perplexity(teacher, tokenizer, device, test_text)
    print(f"원본 퍼플렉시티: {orig_perplexity:.2f}")
    
    # 결과 요약
    print("\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)
    
    print(f"\n추론 속도:")
    print(f"  - 원본: {orig_time:.3f}초")
    print(f"  - 전체압축: {comp_full_time:.3f}초 ({comp_full_time/orig_time:.2f}x)")
    print(f"  - 부분압축: {comp_partial_time:.3f}초 ({comp_partial_time/orig_time:.2f}x)")
    
    print(f"\n퍼플렉시티 (낮을수록 좋음):")
    print(f"  - 원본: {orig_perplexity:.2f}")
    print(f"  - 전체압축: {comp_full_perplexity:.2f} (차이: +{comp_full_perplexity-orig_perplexity:.2f})")
    print(f"  - 부분압축: {comp_partial_perplexity:.2f} (차이: +{comp_partial_perplexity-orig_perplexity:.2f})")
    
    print(f"\n출력 일치율:")
    for name, results in [("전체압축", comp_full_results), ("부분압축", comp_partial_results)]:
        matches = sum(1 for o, c in zip(orig_results, results) if o[1] == c[1])
        print(f"  - {name}: {matches}/{len(prompts)} ({matches/len(prompts)*100:.0f}%)")
    
    # 상세 출력 비교
    print(f"\n상세 출력 비교:")
    for i, prompt in enumerate(prompts):
        print(f"\n프롬프트: '{prompt}'")
        print(f"  원본: {orig_results[i][1]}")
        print(f"  전체: {comp_full_results[i][1]}")
        print(f"  부분: {comp_partial_results[i][1]}")


if __name__ == "__main__":
    main() 