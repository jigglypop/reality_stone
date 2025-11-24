import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional, Union
import os

try:
    from reality_stone.utils.pre_segmenter import PreSegmenter
except ImportError:
    PreSegmenter = None


class SentenceTopicDataset(Dataset):
    """
    Dataset for Hierarchical Sentence-Topic LLM.
    Loads data from a JSON/JSONL file containing paragraphs or structured graph data.
    """
    def __init__(
        self, 
        data_path: str, 
        max_paragraphs: int = 10000,
        max_length: int = 128,
        k_neighbors: int = 3
    ):
        self.data_path = data_path
        self.max_paragraphs = max_paragraphs
        self.max_length = max_length
        self.k_neighbors = k_neighbors
        self.samples: List[Dict[str, Any]] = []
        
        self.segmenter = None
        if PreSegmenter is not None:
            self.segmenter = PreSegmenter(max_length=max_length, k_neighbors=k_neighbors)
        
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} not found.")
            return

        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= self.max_paragraphs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # If data is just a string (paragraph), wrap it
                    if isinstance(data, str):
                        self.samples.append({"paragraph": data})
                    # If data is dict, assume it has "paragraph" or pre-processed fields
                    elif isinstance(data, dict):
                        self.samples.append(data)
                    count += 1
                except json.JSONDecodeError:
                    # Fallback: treat line as raw text paragraph
                    if len(line) > 0:
                        self.samples.append({"paragraph": line})
                        count += 1
        
        print(f"[SentenceTopicDataset] Loaded {len(self.samples)} samples from {self.data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Case 1: Pre-processed data with graph structure (from generate_graph_data.py)
        if "tokens" in sample and "topo_idx" in sample:
            # Convert lists to tensors if they aren't already
            tokens = torch.tensor(sample["tokens"], dtype=torch.long)
            topo_idx = torch.tensor(sample["topo_idx"], dtype=torch.long)
            
            # Optional labels
            topic_labels = None
            if "topic_labels" in sample:
                topic_labels = torch.tensor(sample["topic_labels"], dtype=torch.long)

            # Replacement mask (default to all 1s if not present)
            if "replacement_mask" in sample:
                replacement_mask = torch.tensor(sample["replacement_mask"], dtype=torch.long)
            else:
                replacement_mask = torch.ones_like(tokens)
                
            return {
                "tokens": tokens,
                "topo_idx": topo_idx,
                "replacement_mask": replacement_mask,
                "sentences": sample.get("sentences", []),
                "paragraph": sample.get("paragraph", ""),
                "topic_labels": topic_labels
            }
            
        # Case 2: Raw paragraph text -> Segment on the fly
        elif "paragraph" in sample and self.segmenter is not None:
            text = sample["paragraph"]
            processed = self.segmenter(text)
            
            # PreSegmenter returns dict with tensors on CPU
            # Add missing keys if needed
            processed["paragraph"] = text
            
            # Check for explicit topic labels in raw data
            if "topic_label" in sample:
                # This needs a mapping from label str to int ID. 
                # For now, we skip or implement a simple hash if needed.
                pass
                
            return processed
            
        else:
            # Fallback / Error
            return {
                "tokens": torch.zeros(1, self.max_length, dtype=torch.long),
                "topo_idx": torch.zeros(1, self.k_neighbors, dtype=torch.long),
                "sentences": [],
                "paragraph": ""
            }

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    Pads tokens and topo_idx to the maximum number of sentences in the batch.
    """
    # 1. Find max T (number of sentences)
    max_t = 0
    max_l = 0
    for item in batch:
        if "tokens" in item:
            t, l = item["tokens"].shape
            max_t = max(max_t, t)
            max_l = max(max_l, l)
            
    # 2. Prepare batched tensors
    batch_size = len(batch)
    # tokens: [B, MaxT, L]
    batched_tokens = torch.zeros(batch_size, max_t, max_l, dtype=torch.long)
    
    # topo_idx: [B, MaxT, K]
    # We need to know K from the first non-empty sample
    k = 3
    for item in batch:
        if "topo_idx" in item:
            k = item["topo_idx"].shape[-1]
            break
            
    batched_topo_idx = torch.zeros(batch_size, max_t, k, dtype=torch.long)
    batched_replacement_mask = torch.zeros(batch_size, max_t, max_l, dtype=torch.long)
    
    # topic_labels: [B, MaxT] (Optional)
    has_labels = any("topic_labels" in item and item["topic_labels"] is not None for item in batch)
    batched_topic_labels = torch.full((batch_size, max_t), -1, dtype=torch.long) if has_labels else None
    
    batched_trees = []
    batched_paragraphs = []
    
    for i, item in enumerate(batch):
        tokens = item.get("tokens") # [T, L]
        topo = item.get("topo_idx") # [T, K]
        labels = item.get("topic_labels") # [T] or None
        
        if tokens is not None:
            t, l = tokens.shape
            batched_tokens[i, :t, :l] = tokens
            
        mask = item.get("replacement_mask")
        if mask is not None:
            t_m, l_m = mask.shape
            batched_replacement_mask[i, :t_m, :l_m] = mask
            
        if topo is not None:
            t_topo, k_topo = topo.shape
            # Adjust topology indices if they point to padding? 
            # Actually, topo indices are relative to the sentence list (0..T-1).
            # We just copy them. When using, we must mask out padding sentences.
            batched_topo_idx[i, :t_topo, :k_topo] = topo
            
            # If we pad sentences, we might need to adjust topo to point to self for padded areas?
            # For now, 0-padding is safe if we mask properly in the model.
            
        if has_labels and labels is not None:
            t_labels = labels.shape[0]
            batched_topic_labels[i, :t_labels] = labels
            
        if "tree" in item:
            batched_trees.append(item["tree"])
            
        if "paragraph" in item:
            batched_paragraphs.append(item["paragraph"])
            
    return {
        "tokens": batched_tokens,
        "topo_idx": batched_topo_idx,
        "replacement_mask": batched_replacement_mask,
        "topic_labels": batched_topic_labels,
        "tree": batched_trees if batched_trees else None,
        "paragraphs": batched_paragraphs
    }

