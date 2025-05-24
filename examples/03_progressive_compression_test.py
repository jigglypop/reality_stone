"""
점진적 압축으로 에러 발생 지점 찾기
"""

import torch
import torch.nn as nn
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reality Stone 백엔드 로드
import sys
sys.path.insert(0, '.')
import reality_stone

# 압축 클래스 import
from reality_stone_helgason_proper import RealityStoneHelgasonLinear

# 모델 로드
print("모델 로드 중...")
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def test_generation(model, test_name):
    """간단한 생성 테스트"""
    try:
        prompt = "안녕하세요"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 5,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ {test_name}: SUCCESS - {generated}")
        return True
        
    except Exception as e:
        print(f"❌ {test_name}: FAIL - {e}")
        return False

# 테스트 시작
print("\n=== 점진적 압축 테스트 ===")

# 1. 원본 모델 테스트
print("\n1. 원본 모델 테스트")
test_generation(original_model, "원본")

# 2. 첫 번째 레이어 MLP만 압축
print("\n2. 첫 번째 레이어 MLP만 압축")
test_model_1 = copy.deepcopy(original_model)
layer_0 = test_model_1.transformer.h[0]

# MLP c_proj만 압축
compressed_proj = RealityStoneHelgasonLinear(layer_0.mlp.c_proj, 0.5, "mlp_c_proj_0", verbose=False)
layer_0.mlp.c_proj = compressed_proj
test_generation(test_model_1, "Layer 0 MLP c_proj")

# 3. 첫 번째 레이어 MLP 전체 압축
print("\n3. 첫 번째 레이어 MLP 전체 압축")
test_model_2 = copy.deepcopy(original_model)
layer_0 = test_model_2.transformer.h[0]

# MLP 전체 압축
compressed_fc = RealityStoneHelgasonLinear(layer_0.mlp.c_fc, 0.5, "mlp_c_fc_0", verbose=False)
compressed_proj = RealityStoneHelgasonLinear(layer_0.mlp.c_proj, 0.5, "mlp_c_proj_0", verbose=False)
layer_0.mlp.c_fc = compressed_fc
layer_0.mlp.c_proj = compressed_proj
test_generation(test_model_2, "Layer 0 MLP 전체")

# 4. 첫 번째 레이어 Attention도 압축
print("\n4. 첫 번째 레이어 Attention도 압축")
test_model_3 = copy.deepcopy(original_model)
layer_0 = test_model_3.transformer.h[0]

# MLP 압축
compressed_fc = RealityStoneHelgasonLinear(layer_0.mlp.c_fc, 0.5, "mlp_c_fc_0", verbose=False)
compressed_proj = RealityStoneHelgasonLinear(layer_0.mlp.c_proj, 0.5, "mlp_c_proj_0", verbose=False)
layer_0.mlp.c_fc = compressed_fc
layer_0.mlp.c_proj = compressed_proj

# Attention 압축
compressed_attn = RealityStoneHelgasonLinear(layer_0.attn.c_attn, 0.5, "attn_c_attn_0", verbose=False)
compressed_attn_proj = RealityStoneHelgasonLinear(layer_0.attn.c_proj, 0.5, "attn_c_proj_0", verbose=False)
layer_0.attn.c_attn = compressed_attn
layer_0.attn.c_proj = compressed_attn_proj
test_generation(test_model_3, "Layer 0 전체")

# 5. 두 번째 레이어까지 압축
print("\n5. 두 번째 레이어까지 압축")
test_model_4 = copy.deepcopy(original_model)

for layer_idx in range(2):  # Layer 0, 1
    layer = test_model_4.transformer.h[layer_idx]
    
    # MLP 압축
    compressed_fc = RealityStoneHelgasonLinear(layer.mlp.c_fc, 0.5, f"mlp_c_fc_{layer_idx}", verbose=False)
    compressed_proj = RealityStoneHelgasonLinear(layer.mlp.c_proj, 0.5, f"mlp_c_proj_{layer_idx}", verbose=False)
    layer.mlp.c_fc = compressed_fc
    layer.mlp.c_proj = compressed_proj
    
    # Attention 압축
    compressed_attn = RealityStoneHelgasonLinear(layer.attn.c_attn, 0.5, f"attn_c_attn_{layer_idx}", verbose=False)
    compressed_attn_proj = RealityStoneHelgasonLinear(layer.attn.c_proj, 0.5, f"attn_c_proj_{layer_idx}", verbose=False)
    layer.attn.c_attn = compressed_attn
    layer.attn.c_proj = compressed_attn_proj

test_generation(test_model_4, "Layer 0-1 전체")

# 6. 처음 3개 레이어 압축
print("\n6. 처음 3개 레이어 압축")
test_model_5 = copy.deepcopy(original_model)

for layer_idx in range(3):  # Layer 0, 1, 2
    layer = test_model_5.transformer.h[layer_idx]
    
    # MLP 압축
    compressed_fc = RealityStoneHelgasonLinear(layer.mlp.c_fc, 0.5, f"mlp_c_fc_{layer_idx}", verbose=False)
    compressed_proj = RealityStoneHelgasonLinear(layer.mlp.c_proj, 0.5, f"mlp_c_proj_{layer_idx}", verbose=False)
    layer.mlp.c_fc = compressed_fc
    layer.mlp.c_proj = compressed_proj
    
    # Attention 압축
    compressed_attn = RealityStoneHelgasonLinear(layer.attn.c_attn, 0.5, f"attn_c_attn_{layer_idx}", verbose=False)
    compressed_attn_proj = RealityStoneHelgasonLinear(layer.attn.c_proj, 0.5, f"attn_c_proj_{layer_idx}", verbose=False)
    layer.attn.c_attn = compressed_attn
    layer.attn.c_proj = compressed_attn_proj

test_generation(test_model_5, "Layer 0-2 전체")

print("\n=== 점진적 압축 테스트 완료 ===") 