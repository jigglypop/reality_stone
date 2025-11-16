import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from collections import defaultdict

def analyze_model_layers(model_name="skt/kogpt2-base-v2"):
    """모델의 모든 레이어 타입을 분석합니다."""
    print(f"Analyzing model: {model_name}\n")
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 레이어 타입별 카운트
    layer_types = defaultdict(int)
    layer_examples = defaultdict(list)
    
    # 모든 모듈 순회
    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_types[module_type] += 1
        
        # 각 타입별로 최대 3개의 예시만 저장
        if len(layer_examples[module_type]) < 3:
            layer_examples[module_type].append(name)
    
    # 결과 출력
    print("=" * 70)
    print("LAYER TYPE ANALYSIS")
    print("=" * 70)
    
    # 메인 모델 타입 제외하고 출력
    skip_types = ['GPT2LMHeadModel', 'GPT2Model', 'ModuleList']
    
    for layer_type, count in sorted(layer_types.items()):
        if layer_type not in skip_types:
            print(f"\n{layer_type}: {count} instances")
            print(f"  Examples: {', '.join(layer_examples[layer_type][:3])}")
    
    # 변환이 필요한 레이어 타입 식별
    print("\n" + "=" * 70)
    print("LAYERS REQUIRING CONVERSION")
    print("=" * 70)
    
    convertible_layers = []
    for layer_type in layer_types:
        if layer_type in ['Linear', 'Conv1D', 'Embedding', 'LayerNorm']:
            convertible_layers.append(layer_type)
            print(f"- {layer_type}")
    
    return layer_types, layer_examples

if __name__ == "__main__":
    analyze_model_layers() 