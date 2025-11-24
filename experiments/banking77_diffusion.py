import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import math
from reality_stone.layers.lorentz import lorentz_distance

# Reality Stone Philosophy Restored:
# "Riemannian Geometry is the Engine, Diffusion is the Fuel."

def to_lorentz(x, c=1.0):
    """Lift Euclidean vector x to Lorentz Hyperboloid"""
    # x0 = sqrt(1/c + ||x||^2)
    sq = (x * x).sum(dim=-1, keepdim=True)
    time_comp = torch.sqrt(1.0/c + sq)
    return torch.cat([time_comp, x], dim=-1)

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
        cls_emb = outputs.last_hidden_state[:, 0, :] 
        return cls_emb

class RiemannianHyperExpansion(nn.Module):
    def __init__(self, input_dim=768, num_classes=77, hyper_dim=4096, steps=3, c=0.1):
        super().__init__()
        self.steps = steps
        self.hyper_dim = hyper_dim
        self.c = c # Curvature
        
        print(f"Initializing Riemannian Hyper-Expansion: {input_dim} -> {hyper_dim} dim (Curvature {c})")
        
        # 1. Expansion (Euclidean -> High-dim Tangent Space)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hyper_dim),
            nn.LayerNorm(hyper_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 2. Diffusion Block (Tangent Space Dynamics)
        # We process in tangent space for efficiency, but respecting the manifold structure via distance later
        self.diffusion_block = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim),
            nn.GELU(),
            nn.Linear(hyper_dim, hyper_dim),
            nn.LayerNorm(hyper_dim)
        )
        
        # 3. Riemannian Readout
        # Prototypes live in the 4096-dim ambient space (before lifting)
        self.prototypes = nn.Parameter(torch.randn(num_classes, hyper_dim) * 0.01)

    def forward(self, x):
        # 1. Expansion to Hyperspace
        h = self.projector(x)
        
        # 2. Diffusion
        for t in range(self.steps):
            diffused = self.diffusion_block(h)
            h = h + diffused
            
        # 3. Riemannian Distance Calculation
        # Lift both h and prototypes to Lorentz Manifold
        # h: (B, 4096) -> h_L: (B, 4097)
        # P: (77, 4096) -> P_L: (77, 4097)
        
        h_L = to_lorentz(h, self.c)
        P_L = to_lorentz(self.prototypes, self.c)
        
        # Calculate pairwise Lorentz distance
        # We need to expand for broadcasting
        # h_L: (B, 1, 4097)
        # P_L: (1, 77, 4097)
        
        B = h.size(0)
        h_exp = h_L.unsqueeze(1).expand(B, self.prototypes.size(0), -1)
        P_exp = P_L.unsqueeze(0).expand(B, -1, -1)
        
        # Use Reality Stone's Lorentz Distance (Autograd supported now)
        # Flatten batch dims for kernel: (B*77, 4097)
        dist = lorentz_distance(
            h_exp.reshape(-1, self.hyper_dim + 1),
            P_exp.reshape(-1, self.hyper_dim + 1),
            self.c
        ).reshape(B, -1)
        
        return -dist # Negative distance = Logits-like score

def run_banking77_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Banking77 Riemannian Hyper-Expansion (Reality Stone) ===")
    print(f"Device: {DEVICE}")
    
    dataset = load_dataset("banking77")
    train_data = dataset['train']
    test_data = dataset['test']
    num_classes = 77
    
    encoder = BERTEncoder().to(DEVICE)
    model = RiemannianHyperExpansion(
        input_dim=768, 
        num_classes=num_classes, 
        hyper_dim=4096, 
        steps=2,
        c=0.1 # Optimal curvature found in previous experiments
    ).to(DEVICE)
    
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
        {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
    ])
    
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    epochs = 20
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    cycle_epochs = min(10, epochs)
    scheduler_one = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        steps_per_epoch=steps_per_epoch,
        epochs=cycle_epochs
    )
    scheduler_two = None
    best_acc = 0.0
    
    print(f"Start Training...")
    
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
            if epoch <= cycle_epochs:
                scheduler_one.step()
            else:
                if scheduler_two is None:
                    scheduler_two = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=(epochs - cycle_epochs) * steps_per_epoch,
                        eta_min=1e-6
                    )
                scheduler_two.step()
            
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
    run_banking77_experiment()
