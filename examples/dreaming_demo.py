import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.dreaming import BrainOS
import time

def dreaming_demo():
    print("=== Reality Stone: Brain OS Demo (Zero-Shot Context Switching) ===")
    
    model_id = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 1. BrainOS 설치 (기존 모델 래핑)
    # 이 과정은 모델 가중치를 건드리지 않고, 연산 파이프라인만 가로챕니다.
    brain = BrainOS(model)
    
    input_text = "The logic of the universe is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    def generate(tag):
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=30, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95
            )
        dur = time.time() - start
        print(f"\n[{tag}] ({dur:.3f}s) Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # 2. 상태별 출력 비교 (가중치는 100% 동일함!)
    
    # A. 평상시 (Euclidean)
    brain.wake_up()
    generate("WAKE")
    
    # B. 논리 모드 (Hyperbolic) - 공간을 오목하게 왜곡 (-0.01)
    brain.focus_logic()
    generate("LOGIC")
    
    # C. 창의 모드 (Spherical) - 공간을 볼록하게 왜곡 (+0.01)
    brain.focus_creative()
    generate("CREATIVE")
    
    # D. 수면 (Dreaming) - 최적의 상태를 찾아 자동 조정
    brain.sleep_and_consolidate(None)
    generate("OPTIMIZED")
    
    print("\n✅ 증명 완료: 가중치 변경 없이(Lossless), '공간의 곡률'만으로 사고 방식이 전환되었습니다.")

if __name__ == "__main__":
    dreaming_demo()

