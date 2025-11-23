import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import time

class BERTEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        print(f"Loading {model_name} (Full Fine-tuning Mode)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Active use: No freezing!
        # But use gradient checkpointing to save memory if needed
        # self.bert.gradient_checkpointing_enable() 
            
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

class HyperExpansionDiffusionModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=77, hyper_dim=4096, steps=3):
        super().__init__()
        self.steps = steps
        self.hyper_dim = hyper_dim
        
        print(f"Initializing Hyper-Expansion Diffusion: {input_dim} -> {hyper_dim} dim")
        
        # 1. Hyper-Expansion Projector
        # Blast the input into a much larger space where separation is easier
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hyper_dim),
            nn.LayerNorm(hyper_dim),
            nn.GELU(),
            nn.Dropout(0.3) # Strong dropout
        )
        
        # 2. High-Energy Diffusion Matrix
        # Instead of a simple matrix, we use a powerful mixing block
        # recycled over time steps (Recurrent FFN)
        self.diffusion_block = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim),
            nn.GELU(),
            nn.Linear(hyper_dim, hyper_dim),
            nn.LayerNorm(hyper_dim) # Stability is key in high dims
        )
        
        # 3. Readout from Hyperspace
        self.readout = nn.Linear(hyper_dim, num_classes)
        
        # Learnable initial state for prototypes in hyperspace? No, just let them emerge.

    def forward(self, x):
        # x: (B, 768)
        
        # 1. Expansion
        h = self.projector(x) # (B, 4096)
        
        # 2. Diffusion / Reasoning Loop
        # Recirculate energy in hyperspace to refine representation
        for t in range(self.steps):
            # Hard injection + Diffusion
            # h_new = F(h_old) + h_initial (skip connection from expanded input)
            diffused = self.diffusion_block(h)
            h = h + diffused # ResNet style accumulation
            
        # 3. Collapse to Decision
        logits = self.readout(h)
        return logits

def run_banking77_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Banking77 Hyper-Expansion Diffusion (Active Mode) ===")
    print(f"Device: {DEVICE}")
    
    # Data
    dataset = load_dataset("banking77")
    train_data = dataset['train']
    test_data = dataset['test']
    num_classes = 77
    
    # Model
    encoder = BERTEncoder().to(DEVICE)
    model = HyperExpansionDiffusionModel(
        input_dim=768, 
        num_classes=num_classes, 
        hyper_dim=4096, # Aggressive expansion
        steps=2 # Fast, punchy reasoning
    ).to(DEVICE)
    
    # Optimizer: Aggressive Fine-tuning
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': 2e-5, 'weight_decay': 0.01}, # Standard BERT fine-tuning LR
        {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # Higher LR for fresh layers
    ])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[5e-5, 5e-4], 
        epochs=10, 
        steps_per_epoch=len(train_data)//32
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    batch_size = 32
    epochs = 10 # Short and intense training
    
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
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / (len(train_data) / batch_size)
        
        # Eval
        encoder.eval()
        model.eval()
        correct = 0
        total = 0
        
        eval_batch_size = 64
        eval_indices = range(0, len(test_data), eval_batch_size)
        
        with torch.no_grad():
            for i in eval_indices:
                batch_idx = range(i, min(i+eval_batch_size, len(test_data)))
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
