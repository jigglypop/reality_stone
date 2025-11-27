#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.models.transformer_converter import (
    load_rsulf_model_checkpoint,
    rsulf_generate,
)

def main():
    print('1. 모델 로딩...')
    tokenizer = AutoTokenizer.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        cache_dir='E:/hf-cache',
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        cache_dir='E:/hf-cache',
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.eval()

    print('2. RS-ULF 체크포인트 로드...')
    rs_model = load_rsulf_model_checkpoint('checkpoints/rsulf_fast')
    print(f'   레이어 수: {len(rs_model.layers)}')
    print(f'   압축률: {rs_model.param_count()["ratio"]:.2f}x')

    print('3. 한글 생성 테스트...')
    prompts = [
        '안녕하세요, 오늘 날씨가',
        '한국의 수도는',
        '인공지능이란',
    ]

    for prompt in prompts:
        print(f'\n입력: {prompt}')
        result = rsulf_generate(
            model,
            rs_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.7,
            top_p=0.9,
            device='cuda',
        )
        print(f'출력: {result}')

if __name__ == "__main__":
    main()

