import sys
import os
from pathlib import Path

os.environ["HF_HOME"] = "E:/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/hf-cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "E:/hf-cache/datasets"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from experiments.intent_classification.banking77_dataset import Banking77Dataset
from experiments.intent_classification.models.riemannian_intent_classifier import RiemannianIntentClassifier
from reality_stone.losses import laplacian_same_label, poincare_kinetic_energy, HyperbolicSupConLoss


def collate_banking77(batch, tokenizer, max_length=128): # Increased for roberta-large
    texts = [b["text"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def train_banking77(
    data_root="data/banking77",
    backbone_name="roberta-large", # Changed to roberta-large
    num_epochs=20, # Reduced epochs for fine-tuning
    batch_size=16,
    lr_backbone=1e-5,
    lr_head=1e-3,
    hyp_dim=768,
    num_layers=2,
    num_heads=4,
    curvature=1.0,
    alpha=3.0,
    gamma=0.0,
    lambda_lap=0.0,
    lambda_proto=0.001,
    lambda_kin=0.0,
    lambda_con=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Device: {device}")
    print(f"Backbone: {backbone_name}")
    print(f"Curvature: {curvature}, Alpha: {alpha}, Gamma: {gamma}")
    
    data_root_path = Path(data_root)
    
    train_ds = Banking77Dataset(data_root_path, split="train")
    test_ds = Banking77Dataset(data_root_path, split="test")
    
    num_classes = len(train_ds.label2id)
    print(f"Number of classes: {num_classes}")
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_banking77(b, tokenizer),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_banking77(b, tokenizer),
    )
    
    model = RiemannianIntentClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        hyp_dim=hyp_dim,
        curvature=curvature,
        alpha=alpha,
        gamma=gamma,
        dropout=0.1,
        num_prototypes=4,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Differential Learning Rates
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # Backbone parameters (low lr)
        {'params': [p for n, p in param_optimizer if 'backbone' in n and not any(nd in n for nd in no_decay)], 
         'weight_decay': 0.01, 'lr': lr_backbone},
        {'params': [p for n, p in param_optimizer if 'backbone' in n and any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': lr_backbone},
        # Riemannian Head parameters (high lr)
        {'params': [p for n, p in param_optimizer if 'backbone' not in n and not any(nd in n for nd in no_decay)], 
         'weight_decay': 0.01, 'lr': lr_head},
        {'params': [p for n, p in param_optimizer if 'backbone' not in n and any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': lr_head},
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # Scheduler with warmup (Cosine)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    supcon_loss_fn = HyperbolicSupConLoss(temperature=0.1, curvature=curvature)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits, x_hyp = model(input_ids=input_ids, attention_mask=attention_mask)
            ce_loss = criterion(logits, labels)

            # Regularization
            proto = model.get_prototypes()
            proto_norm_sq = torch.sum(proto ** 2, dim=-1)
            reg_geom = torch.mean(proto_norm_sq)

            # SupCon Loss (replaces Laplacian)
            if lambda_con > 0 and x_hyp.size(0) > 1:
                # Update curvature for loss function if learnable
                supcon_loss_fn.curvature = model.c.item() if hasattr(model.c, 'item') else model.c
                con_loss = supcon_loss_fn(x_hyp, labels)
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = (
                ce_loss
                + lambda_proto * reg_geom
                + lambda_con * con_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "con": f"{con_loss.item():.4f}"
            })
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        
        print(f"Epoch {epoch+1}: Val Acc={acc:.4f}, F1={f1:.4f}")
        
        # Scheduler step handled per iteration
        # scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            out_dir = ROOT_DIR / "models" / "intent_classification"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "best_riemannian_intent_roberta.pth"
            torch.save(model.state_dict(), out_path)
            print(f"Saved best model: {out_path} (Acc={best_acc:.4f})")
    
    print(f"\nBest Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    # Reduce batch size for roberta-large (memory intensive)
    # Use gradient accumulation or smaller batch if OOM
    train_banking77(batch_size=16, num_epochs=20, lr_head=5e-4)
