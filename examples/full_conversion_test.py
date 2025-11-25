import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.conversion import convert_to_full_riemannian

def test_full_conversion():
    print("=== Deep Manifold Injection Test ===")
    
    # 1. 모델 로드
    model_id = "gpt2"
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # CUDA 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    
    # 2. 수술 전 테스트
    print("\n--- Before Surgery ---")
    text = "The nature of space and time is"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        orig_out = model.generate(**inputs, max_new_tokens=20, do_sample=True)
    print(f"Output: {tokenizer.decode(orig_out[0])}")

    # 3. 완전 리만 변환 (수술 집도)
    # 곡률(curvature)을 아주 작게(0.001) 설정하여 기존 지식을 보존하면서 기하학적 특성 주입
    print("\n--- Performing Riemannian Surgery ---")
    model = convert_to_full_riemannian(model, curvature=0.001)
    model = model.to(device) # Ensure new layers are on device

    # 4. 수술 후 테스트
    print("\n--- After Surgery ---")
    # 파라미터는 그대로지만, 연산 방식(forward)이 기하학적으로 변경됨
    with torch.no_grad():
        riemann_out = model.generate(**inputs, max_new_tokens=20, do_sample=True)
    print(f"Output: {tokenizer.decode(riemann_out[0])}")
    
    # 5. 레이어 확인
    print("\nChecking internal layer structure (Transformer Block 0 MLP):")
    # GPT2의 경우 h[0].mlp.c_fc 가 Conv1D인데, nn.Linear만 교체하도록 했으므로
    # 확인을 위해 c_proj (Linear인 경우가 많음) 등을 체크하거나
    # Llama 같은 최신 모델 구조에서는 모든게 Linear라 더 확실함.
    # GPT2에서 nn.Linear를 사용하는 부분을 찾아서 출력
    
    # Check if any module is RiemannianLinear
    found = False
    for name, module in model.named_modules():
        if "RiemannianLinear" in str(type(module)):
            print(f"Found converted layer: {name} -> {type(module)}")
            found = True
            break
    
    if not found:
        print("Note: GPT-2 uses Conv1D for many layers, so fewer layers might have been converted than in Llama.")

if __name__ == "__main__":
    test_full_conversion()

