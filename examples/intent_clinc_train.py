import os
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "clinc_oos"
DATASET_CONFIG = "plus"
MAX_LEN = 32
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
EMBED_DIM = 128
C = 1.0

ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

def build_vocab(texts, min_freq=2, max_size=30000):
    cnt = Counter()
    for t in texts:
        cnt.update(t.strip().lower().split())
    it = sorted([w for w,f in cnt.items() if f>=min_freq], key=lambda w:-cnt[w])[:max_size-2]
    stoi = {"[PAD]":0,"[UNK]":1}
    for w in it:
        stoi[w]=len(stoi)
    return stoi

vocab = build_vocab([x["text"] for x in ds["train"]])
PAD, UNK = 0,1

def encode(text):
    toks = text.strip().lower().split()
    ids = [vocab.get(t,UNK) for t in toks][:MAX_LEN]
    if len(ids)<MAX_LEN:
        ids = ids + [PAD]*(MAX_LEN-len(ids))
    mask = [1 if i!=PAD else 0 for i in ids]
    return torch.tensor(ids,dtype=torch.long), torch.tensor(mask,dtype=torch.long)

def collate(batch):
    ids, masks, labels = [], [], []
    for b in batch:
        i,m = encode(b["text"])
        ids.append(i)
        masks.append(m)
        labels.append(b["intent"])
    return {"input_ids":torch.stack(ids), "attention_mask":torch.stack(masks)}, torch.tensor(labels,dtype=torch.long)

train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(ds["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
num_labels = ds["train"].features["intent"].num_classes

class RiemannIntentModel(nn.Module):
    def __init__(self, vocab_size, dim, num_labels, c):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.classifier = nn.Linear(dim, num_labels)
        self.c = c
    def project_ball(self, x):
        n = torch.norm(x,dim=-1,keepdim=True)
        return torch.tanh(n)*(x/(n+1e-9))*0.99
    def log0(self, y):
        n = torch.norm(y,dim=-1,keepdim=True)
        return torch.atanh(torch.clamp(n,min=0.0,max=0.999999))*(y/(n+1e-9))
    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).float()
        x = (x*m).sum(dim=1)/(m.sum(dim=1)+1e-9)
        u = self.project_ball(x)
        z = self.log0(u)
        return self.classifier(z)

model = RiemannIntentModel(len(vocab), EMBED_DIM, num_labels, C).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

def verify_bindings(sample_x):
    import _rust
    x = sample_x.detach().cpu().numpy().astype(np.float32)
    z = np.zeros_like(x, dtype=np.float32)
    _ = _rust.poincare.poincare_exp_at_cpu(z, x, float(C))
    _ = _rust.poincare.poincare_log_at_cpu(z, x, float(C))
    d = _rust.poincare.poincare_distance_cpu(x, x, float(C))
    assert d.shape[0] == x.shape[0]

with torch.no_grad():
    ids, masks = encode(ds["train"][0]["text"])
    model.eval()
    sample = model.emb(ids.unsqueeze(0).to(DEVICE)).mean(dim=1)
    verify_bindings(sample.squeeze(0))

for epoch in range(1,EPOCHS+1):
    model.train()
    tot, corr, cnt = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"train {epoch}/{EPOCHS}")
    for enc, y in pbar:
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        y = y.to(DEVICE)
        opt.zero_grad()
        logits = model(enc["input_ids"], enc["attention_mask"])
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        tot += loss.item()
        pred = logits.argmax(dim=-1)
        corr += (pred==y).sum().item()
        cnt += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{corr/cnt:.4f}")
    model.eval()
    corr_val, cnt_val = 0, 0
    with torch.no_grad():
        for enc, y in val_loader:
            enc = {k:v.to(DEVICE) for k,v in enc.items()}
            y = y.to(DEVICE)
            logits = model(enc["input_ids"], enc["attention_mask"])
            pred = logits.argmax(dim=-1)
            corr_val += (pred==y).sum().item()
            cnt_val += y.size(0)
    print(f"val_acc={corr_val/cnt_val:.4f}")

model.eval()
corr_test, cnt_test = 0,0
with torch.no_grad():
    for enc, y in test_loader:
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        y = y.to(DEVICE)
        logits = model(enc["input_ids"], enc["attention_mask"])
        pred = logits.argmax(dim=-1)
        corr_test += (pred==y).sum().item()
        cnt_test += y.size(0)
print(f"test_acc={corr_test/cnt_test:.4f}")
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
