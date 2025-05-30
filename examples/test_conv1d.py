import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# GPT2 모델 로드
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")

# 첫 번째 Conv1D 레이어 확인
conv1d = model.transformer.h[0].attn.c_attn

print(f"Conv1D weight shape: {conv1d.weight.shape}")
print(f"Conv1D bias shape: {conv1d.bias.shape}")
print(f"Conv1D nf: {conv1d.nf}")

# 테스트 입력 생성
batch_size = 1
seq_len = 5
hidden_size = 768

x = torch.randn(batch_size, seq_len, hidden_size)
print(f"\n입력 shape: {x.shape}")

# Conv1D forward 확인
output = conv1d(x)
print(f"출력 shape: {output.shape}")

# 실제 연산 확인
# Conv1D는 실제로 x @ weight.T + bias
weight = conv1d.weight  # [out_features, in_features]
bias = conv1d.bias

# 수동 계산 - Conv1D는 weight가 transpose되어 저장됨
# weight shape: [in_features, out_features] 
# 실제 연산: x @ weight (transpose 없이!)
manual_output = x @ weight + bias
print(f"\n수동 계산 출력 shape: {manual_output.shape}")
print(f"결과 동일: {torch.allclose(output, manual_output)}")

print(f"\nConv1D 정리:")
print(f"- weight 저장: [{weight.shape[0]}, {weight.shape[1]}] = [in_features, out_features]")
print(f"- 실제 weight: [out_features, in_features]의 전치")
print(f"- 입력: [batch, seq_len, in_features]")
print(f"- 출력: [batch, seq_len, out_features]")
print(f"- 연산: input @ weight + bias (transpose 없이!)") 