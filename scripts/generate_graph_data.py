import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
import argparse
import os

"""
Generate Reverse Attention Graph Data

관세음보살:
이 스크립트는 Teacher Model(예: Llama-3, GPT-2 등)이 텍스트를 생성할 때의
Attention Weight를 역산하여 '논리적 연결 그래프'를 추출합니다.

H×S 엔진(Hyperbolic-Spherical Engine)의 강력한 분별력을 활용하기 위해,
단순한 시간 순서(Previous/Next Token)가 아닌,
실제 모델이 '주목(Attention)'한 의미적 인과관계를 데이터셋에 포함시킵니다.
"""

def extract_attention_graph(
    attentions: Tuple[torch.Tensor], 
    k_neighbors: int = 3,
    threshold: float = 0.1
) -> List[List[int]]:
    """
    Extract topology indices from attention weights.
    
    Args:
        attentions: Tuple of tensors (n_layers, batch, n_heads, seq_len, seq_len)
        k_neighbors: Number of neighbors to keep for each token
        threshold: Minimum attention weight to consider as an edge
        
    Returns:
        topo_idx: List[List[int]] of shape [seq_len, k_neighbors]
    """
    # Use the last layer's average attention across heads
    # attentions[-1] shape: [batch, n_heads, seq_len, seq_len]
    # We assume batch_size=1 for generation
    if attentions is None:
        print("Error: attentions is None")
        return []
    
    print(f"Attentions type: {type(attentions)}, length: {len(attentions)}")
    if len(attentions) > 0:
        print(f"Last layer type: {type(attentions[-1])}")
        if attentions[-1] is None:
            print("Error: Last layer attention is None")
            # Try finding the last non-None layer
            for i in range(len(attentions) - 1, -1, -1):
                if attentions[i] is not None:
                    print(f"Using layer {i} instead")
                    last_layer_attn = attentions[i][0].mean(dim=0).float().cpu().numpy()
                    break
            else:
                 return []
        else:
             last_layer_attn = attentions[-1][0].mean(dim=0).float().cpu().numpy()  # [seq_len, seq_len]
    else:
        return []
    
    seq_len = last_layer_attn.shape[0]
    topo_idx = []
    
    for i in range(seq_len):
        # Get attention scores for current token 'i' attending to previous tokens 'j' (j <= i)
        # In causal models, attn[i, j] is weight of i attending to j.
        scores = last_layer_attn[i, :i+1].copy()
        
        # Self-attention is usually high, we might want to exclude it or keep it.
        # Let's keep it but ensure we find other meaningful connections.
        
        # Get indices of top-k scores
        if len(scores) <= k_neighbors:
            # Not enough history, pad with self
            neighbors = list(range(len(scores)))
            while len(neighbors) < k_neighbors:
                neighbors.append(i)
        else:
            # Argsort returns indices that sort the array. [::-1] to reverse (descending)
            top_indices = np.argsort(scores)[::-1]
            neighbors = []
            for idx in top_indices:
                if len(neighbors) >= k_neighbors:
                    break
                if scores[idx] >= threshold or len(neighbors) < 1: # Ensure at least one (or threshold)
                    neighbors.append(int(idx))
            
            # Fill if below threshold but need k neighbors
            idx_ptr = 0
            while len(neighbors) < k_neighbors:
                if idx_ptr < len(top_indices):
                    candidate = int(top_indices[idx_ptr])
                    if candidate not in neighbors:
                        neighbors.append(candidate)
                    idx_ptr += 1
                else:
                    neighbors.append(i)
        
        topo_idx.append(neighbors[:k_neighbors])
        
    return topo_idx

def generate_and_save(
    model_name: str,
    prompts: List[str],
    output_file: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
    k_neighbors: int = 3
):
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Force eager attention to ensure we get attention weights (SDPA might not return them)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        model.config.output_attentions = True
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating data for {len(prompts)} prompts...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_ids = outputs.sequences[0]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract attentions
            # outputs.attentions is a tuple (one per step) of tuples (one per layer)
            # We need to aggregate them into a full matrix or just take the last step's full view if available.
            # For 'generate', it returns attentions for generated tokens step-by-step.
            # But 'forward' pass on the full sequence gives the full causal mask attention.
            # To get the full graph for the *completed* sentence, we should run one forward pass.
            
            # Re-run forward to get full attention matrix
            with torch.no_grad():
                full_out = model(
                    generated_ids.unsqueeze(0), 
                    output_attentions=True, 
                    return_dict=True
                )
            
            if full_out.attentions is None:
                print("Warning: No attentions returned. Skipping this sample.")
                continue

            # full_out.attentions: tuple of [batch, n_heads, seq_len, seq_len]
            topo_idx = extract_attention_graph(
                full_out.attentions, 
                k_neighbors=k_neighbors
            )
            
            # Token IDs list
            tokens_list = generated_ids.tolist()
            
            # Save entry
            entry = {
                "paragraph": text,
                "tokens": [tokens_list], # Batch size 1 format for Dataset
                "topo_idx": [topo_idx],  # Batch size 1 format
                "sentences": [text],     # Simplified: treat whole generation as one sentence/doc
                "model": model_name,
                "prompt": prompt
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Saved: {prompt[:30]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (e.g., gpt2, meta-llama/Llama-3-8b)")
    parser.add_argument("--output", type=str, default="graph_data.jsonl")
    parser.add_argument("--topics", type=str, nargs="+", default=["Artificial Intelligence", "Riemannian Geometry", "Quantum Physics"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    prompts = [f"Explain {topic} in detail." for topic in args.topics]
    
    generate_and_save(
        model_name=args.model,
        prompts=prompts,
        output_file=args.output,
        device=args.device
    )

