import os
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 1. Configuration
MODEL_NAME = "bert-base-uncased" # 토크나이저만 사용
DATASET_NAME = "clinc_oos"
DATASET_CONFIG = "plus"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3 # 하이퍼볼릭은 학습률 튜닝이 중요함
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[intent_clinc] Device: {DEVICE}")

# 2. Load Data & Tokenizer
print("[intent_clinc] Loading Dataset & Tokenizer...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_labels = ds["train"].features["intent"].num_classes

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["intent"] for item in batch], dtype=torch.long)
    
    enc = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=64, 
        return_tensors="pt"
    )
    return enc, labels

train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ds["validation"], batch_size=BATCH_SIZE * 2, collate_fn=collate_fn)
test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE * 2, collate_fn=collate_fn)
