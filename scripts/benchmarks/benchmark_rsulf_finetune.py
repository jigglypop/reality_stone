import os
import time
import torch
import torch.nn as nn
import psutil
import sys
from pathlib import Path

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from transformers import MistralConfig, MistralForCausalLM
from reality_stone.layers.rsulf_cuda import RSULFLayerCUDA, RSULFLMHeadCUDA

"""
RS-ULF Fine-tuning Benchmark (CUDA)

이 스크립트는 RS-ULF 레이어의 미세조정 성능(속도, 메모리)을 벤치마킹합니다.
"""

def print_memory_usage(step=""):
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024**3
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**3
        max_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{step}] RAM: {ram:.2f}GB, VRAM: {vram:.2f}GB (Max: {max_vram:.2f}GB)")
    else:
        print(f"[{step}] RAM: {ram:.2f}GB")

def run_benchmark():
    print("="*60)
    print("RS-ULF Fine-tuning Benchmark (CUDA)")
    print("="*60)
    
    from reality_stone.utils.misc import get_device
    device = torch.device(get_device())
    print(f"디바이스: {device}")
    if device.type == "cuda":
        print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
    
    print_memory_usage("시작")

    # 1. 모델 초기화 (데모용 소형 Mistral)
    print("\n1. 모델 초기화 중...")
    d_model = 512
    r = 128
    config = MistralConfig(
        vocab_size=2000,
        hidden_size=d_model,
        intermediate_size=d_model * 2,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128
    )
    model = MistralForCausalLM(config).to(device)
    print_memory_usage("모델 로드 완료")

    # 2. RS-ULF CUDA 레이어로 변환
    print("\n2. RS-ULF CUDA 레이어로 변환 중...")
    start_time = time.time()
    rsulf_layers = []
    
    for i, layer in enumerate(model.model.layers):
        # Mistral에서 가중치 추출 시뮬레이션
        # Q, K: (d_model, d_model) -> (d_model, r) 데모용 슬라이싱
        wq = layer.self_attn.q_proj.weight.detach().T[:, :r].contiguous()
        wk = layer.self_attn.k_proj.weight.detach().T[:, :r].contiguous()
        
        # FFN 가중치 시뮬레이션
        w1 = torch.randn(d_model, r, device=device) 
        w2 = torch.randn(r, d_model, device=device)
        
        rs_layer = RSULFLayerCUDA(
            wq=wq, wk=wk, w1=w1, w2=w2,
            d_model=d_model, r=r,
            device=device,
            seq_len=64 # 학습 시퀀스 길이와 일치해야 함
        )
        rsulf_layers.append(rs_layer)
        
    rsulf_head = RSULFLMHeadCUDA(
        rsulf_layers=rsulf_layers,
        hidden_size=d_model,
        vocab_size=config.vocab_size,
        device=device
    )
    print(f"변환 소요 시간: {time.time() - start_time:.4f}초")
    print_memory_usage("RS-ULF 변환 완료")
    
    # 3. 학습 루프 설정
    print("\n3. 학습 루프 시작...")
    optimizer = torch.optim.AdamW(rsulf_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 더미 데이터
    batch_size = 8
    seq_len = 64
    num_steps = 10
    
    print(f"배치 크기: {batch_size}, 시퀀스 길이: {seq_len}, 스텝 수: {num_steps}")
    
    model.eval() # 교사 모델은 평가 모드
    rsulf_head.train()
    
    total_time = 0
    
    print("\nStep | Loss      | Time (ms)")
    print("-" * 30)
    
    for step in range(num_steps):
        step_start = time.time()
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # 교사 모델 Forward (Hidden states 추출)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1] # (B, L, d)
        
        # 학생 모델 (RS-ULF) Forward
        logits = rsulf_head(hidden_states) # (B, L, V)
        
        # Loss 계산
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - step_start
        total_time += step_time
        
        if step % 2 == 0:
            print(f"{step:4d} | {loss.item():.6f} | {step_time*1000:.2f}")
            
    print("-" * 30)
    print(f"평균 스텝 시간: {total_time / num_steps * 1000:.2f}ms")
    print("벤치마크 완료.")
    print("="*60)

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
