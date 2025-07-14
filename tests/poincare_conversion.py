import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.layers import HyperbolicLinear, project_to_ball
import time

def convert_to_hyperbolic(model: nn.Module, c: float = 1.0):
    """
    모델의 모든 선형 레이어를 HyperbolicLinear로 교체합니다.
    """
    for name, module in model.named_children():
        # 재귀적으로 하위 모듈 탐색
        if len(list(module.children())) > 0:
            convert_to_hyperbolic(module, c=c)
        
        # Conv1D와 Linear 레이어를 HyperbolicLinear로 교체
        if isinstance(module, nn.Linear) or 'Conv1D' in str(type(module)):
            hyperbolic_layer = HyperbolicLinear.from_linear(module, c=c)
            setattr(model, name, hyperbolic_layer)
            print(f"✅ Replaced '{name}' with HyperbolicLinear(c={c})")

def main():
    print("Loading KoGPT-2 model")
    model_name = "skt/kogpt2-base-v2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. 모델의 첫 번째 레이어를 푸앵카레 공으로 투영하는 래퍼 추가
    #    GPT-2의 경우, 임베딩 이후 첫 입력은 유클리드 공간에 있으므로,
    #    첫 번째 하이퍼볼릭 레이어에 들어가기 전에 투영해야 합니다.
    class InputProjector(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, *args, **kwargs):
            # kwargs에서 inputs_embeds 또는 hidden_states를 찾아 투영
            if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
                kwargs['inputs_embeds'] = project_to_ball(kwargs['inputs_embeds'])
            elif 'hidden_states' in kwargs and kwargs['hidden_states'] is not None:
                 kwargs['hidden_states'] = project_to_ball(kwargs['hidden_states'])
            
            return self.model(*args, **kwargs)

    # 3. 레이어 변환
    print("\n🔄 Converting layers to HyperbolicLinear...")
    start_time = time.time()
    convert_to_hyperbolic(model, c=1.0)
    
    # 모델의 transformer를 InputProjector로 감싸기
    model.transformer = InputProjector(model.transformer)
    print("✅ Wrapped transformer with InputProjector.")
    
    conversion_time = time.time() - start_time
    print(f"⏰ Conversion finished in {conversion_time:.2f} seconds.")

    print("\nConverted model architecture:")
    print(model)

    # 4. 변환된 모델로 생성 테스트
    print("\n✍️ Testing generation with the converted model...")
    prompt = "인공지능의 미래는 어떻게 될까?"
    inputs = tokenizer(prompt, return_tensors="pt")

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=60,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("\n✅ Conversion test successful!")
    except Exception as e:
        print(f"❌ Generation failed after conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 