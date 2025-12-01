#!/usr/bin/env python3
"""
Hierarchical Sentence-Topic LLM 추론 스크립트

이 스크립트는 계층적(Hierarchical) 구조를 가진 RS-ULF 모델을 사용하여
주어진 질문에 대한 답변을 생성합니다. (QA/RAG 태스크용)

주요 기능:
1. Hierarchical 모델 로드 (Checkpoint에서 복원)
2. 문서 검색 (Retrieval) 및 답변 생성
"""

import torch
import argparse
import sys
from pathlib import Path

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    answer_question_with_llm,
)


def load_model(checkpoint_path: str, device: str):
    print(f"모델 로딩 중: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 설정(Config) 복원
    # 체크포인트 포맷: { "config": {...}, "topic_head": ..., "decoder": ... }
    saved_cfg = checkpoint["config"]
    config = HierarchicalLLMConfig(
        vocab_size=saved_cfg["vocab_size"],
        d_model=saved_cfg["d_model"],
        d_head=saved_cfg["d_head"],
        num_topics=saved_cfg["num_topics"],
        num_heads_topic=saved_cfg["num_heads"],
        n_layer_decoder=saved_cfg["n_layer"],
        n_head_decoder=saved_cfg["n_head"],
    )
    
    model = HierarchicalSentenceTopicLLM(config)
    
    # 가중치 로드
    model.topic_head.load_state_dict(checkpoint["topic_head"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="RS-ULF Hierarchical QA Inference")
    parser.add_argument("--model_path", type=str, required=True, help="모델 체크포인트 경로 (.pt)")
    parser.add_argument("--data_path", type=str, required=True, help="참조할 데이터 파일 경로 (.jsonl 등)")
    parser.add_argument("--question", type=str, required=True, help="질문 내용")
    parser.add_argument("--top_k", type=int, default=3, help="참조할 상위 문서 개수")
    
    args = parser.parse_args()
    
    from reality_stone.utils.misc import get_device
    device = get_device()
    
    model = load_model(args.model_path, device)
    
    print(f"\n[질문] {args.question}")
    print("-" * 50)
    
    result = answer_question_with_llm(
        model=model,
        question=args.question,
        data_path=args.data_path,
        max_paragraphs=1000,
        top_k=args.top_k,
        max_new_tokens=64
    )
    
    print("\n[검색 결과 (Evidence)]")
    for i, ans in enumerate(result["retrieval"]["answers"]):
        dist = ans["distance"]
        score = 1.0 / (1.0 + dist)
        sent = ans.get("sentence", "")
        para = ans.get("paragraph", "")
        
        if isinstance(para, dict):
            para_text = str(para.get("paragraph", ""))[:80]
        else:
            para_text = str(para)[:80]
            
        print(f"{i+1}. [유사도: {score:.4f}] {para_text}...")
        if sent:
            print(f"   -> 관련 문장: {sent}")
    
    print("\n[생성된 답변]")
    print(result["answer"])
    print("-" * 50)


if __name__ == "__main__":
    main()
