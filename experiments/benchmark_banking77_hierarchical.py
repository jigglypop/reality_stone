import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import math
import reality_stone as rs
from reality_stone.layers.lorentz import lorentz_distance
from reality_stone import poincare_distance

# Hierarchical Manifold Diffusion
# Stage 1 (Poincare, 1024) -> Stage 2 (Lorentz, 4096)

def to_lorentz(x, c=1.0):
    sq = (x * x).sum(dim=-1, keepdim=True)
    time_comp = torch.sqrt(1.0/c + sq)
    return torch.cat([time_comp, x], dim=-1)

def project_to_ball(x, c=1.0, eps=1e-5):
    norm = x.norm(dim=-1, keepdim=True)
    max_norm = (1.0 / math.sqrt(c)) - eps
    return torch.where(norm > max_norm, x / norm * max_norm, x)

class BERTEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
            
    def forward(self, texts, device):
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors="pt"
        ).to(device)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :] 

class HierarchicalManifoldModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=77):
        super().__init__()
        
        # --- Stage 1: Mid-Level (Poincare Ball, 1024 dim) ---
        self.dim1 = 1024
        self.c1 = 1.0
        
        self.proj1 = nn.Sequential(
            nn.Linear(input_dim, self.dim1),
            nn.LayerNorm(self.dim1),
            nn.Tanh() # Poincare-friendly
        )
        
        # Stage 1 Diffusion (Recurrent Block)
        self.diff1 = nn.GRUCell(self.dim1, self.dim1)
        
        # --- Stage 2: High-Level (Lorentz Hyperboloid, 4096 dim) ---
        self.dim2 = 4096
        self.c2 = 0.1
        
        self.proj2 = nn.Sequential(
            nn.Linear(self.dim1, self.dim2),
            nn.LayerNorm(self.dim2),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Stage 2 Diffusion (ResNet Block)
        self.diff2 = nn.Sequential(
            nn.Linear(self.dim2, self.dim2),
            nn.GELU(),
            nn.Linear(self.dim2, self.dim2),
            nn.LayerNorm(self.dim2)
        )
        
        # Prototypes living in Stage 2 Manifold
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.dim2) * 0.01)

    def forward(self, x):
        # x: (B, 768)
        
        # === Stage 1: Poincare Dynamics ===
        h1 = self.proj1(x)
        h1 = project_to_ball(h1, self.c1)
        
        # Diffusion in Poincare (3 steps)
        # Using GRU to respect the bounded nature (Tanh gate)
        for _ in range(3):
            h1 = self.diff1(h1, h1) # Self-feedback
            h1 = project_to_ball(h1, self.c1) # Stay in ball
            
        # === Stage 2: Lorentz Expansion ===
        # Lift from Poincare to Hyperspace
        h2 = self.proj2(h1)
        
        # Diffusion in Tangent Space of Lorentz (2 steps)
        for _ in range(2):
            h2 = h2 + self.diff2(h2)
            
        # === Readout ===
        # Lift to Lorentz Manifold
        h_L = to_lorentz(h2, self.c2)
        P_L = to_lorentz(self.prototypes, self.c2)
        
        B = h_L.size(0)
        h_exp = h_L.unsqueeze(1).expand(B, self.prototypes.size(0), -1)
        P_exp = P_L.unsqueeze(0).expand(B, -1, -1)
        
        # RS Lorentz Distance
        dist = lorentz_distance(
            h_exp.reshape(-1, self.dim2 + 1),
            P_exp.reshape(-1, self.dim2 + 1),
            self.c2
        ).reshape(B, -1)
        
        return -dist

def run_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Banking77 Hierarchical Manifold Diffusion ===")
    print(f"Structure: BERT -> Poincare(1024) -> Lorentz(4096)")
    
    dataset = load_dataset("banking77")
    train_data = dataset['train']
    test_data = dataset['test']
    num_classes = 77
    
    encoder = BERTEncoder().to(DEVICE)
    model = HierarchicalManifoldModel(num_classes=num_classes).to(DEVICE)
    
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
        {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
    ])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[5e-5, 5e-4], steps_per_epoch=len(train_data)//32, epochs=10
    )
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 32
    epochs = 10
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        model.train()
        total_loss = 0
        
        indices = np.random.permutation(len(train_data))
        pbar = tqdm(range(0, len(train_data), batch_size), desc=f"Ep {epoch}", leave=False)
        
        for i in pbar:
            batch_idx = indices[i:i+batch_size]
            batch_texts = [train_data[int(j)]['text'] for j in batch_idx]
            batch_labels = torch.tensor([train_data[int(j)]['label'] for j in batch_idx]).to(DEVICE)
            
            optimizer.zero_grad()
            embeddings = encoder(batch_texts, DEVICE)
            logits = model(embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / (len(train_data) / batch_size)
        
        encoder.eval()
        model.eval()
        correct = 0
        total = 0
        eval_indices = range(0, len(test_data), 64)
        
        with torch.no_grad():
            for i in eval_indices:
                batch_idx = range(i, min(i+64, len(test_data)))
                batch_texts = [test_data[j]['text'] for j in batch_idx]
                batch_labels = torch.tensor([test_data[j]['label'] for j in batch_idx]).to(DEVICE)
                
                embeddings = encoder(batch_texts, DEVICE)
                logits = model(embeddings)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)
                
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            
        print(f"Ep {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

if __name__ == "__main__":
    run_experiment()

