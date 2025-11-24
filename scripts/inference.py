import torch
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Adjust path to import reality_stone
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    answer_question_with_llm
)

def load_model(checkpoint_path: str, device: str):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct Config
    # The checkpoint format saved in train.py is:
    # { "config": {...}, "topic_head": ..., "decoder": ... }
    
    saved_cfg = checkpoint["config"]
    config = HierarchicalLLMConfig(
        vocab_size=saved_cfg["vocab_size"],
        d_model=saved_cfg["d_model"],
        d_head=saved_cfg["d_head"],
        num_topics=saved_cfg["num_topics"],
        num_heads_topic=saved_cfg["num_heads"],
        n_layer_decoder=saved_cfg["n_layer"],
        n_head_decoder=saved_cfg["n_head"],
        # Use defaults for others or load if saved
    )
    
    model = HierarchicalSentenceTopicLLM(config)
    
    # Load state dicts
    # The model structure in HierarchicalSentenceTopicLLM has .topic_head and .decoder
    # The checkpoint has separate keys "topic_head" and "decoder"
    
    model.topic_head.load_state_dict(checkpoint["topic_head"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model(args.model_path, device)
    
    print(f"\nQuestion: {args.question}")
    print("-" * 50)
    
    result = answer_question_with_llm(
        model=model,
        question=args.question,
        data_path=args.data_path,
        max_paragraphs=1000,
        top_k=args.top_k,
        max_new_tokens=64
    )
    
    print("\n[Retrieval Results (Evidence)]")
    for i, ans in enumerate(result["retrieval"]["answers"]):
        dist = ans['distance']
        # Distance in hyperbolic space can be small; invert/scale for "score" visual
        score = 1.0 / (1.0 + dist)
        print(f"{i+1}. [Score: {score:.4f}] {ans['paragraph'][:100]}...")
        print(f"   -> Sentence: {ans['sentence']}")
    
    print("\n[Generated Answer]")
    print(result["answer"])
    print("-" * 50)

if __name__ == "__main__":
    main()

