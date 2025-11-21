import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.bank_prediction.models.bellman_intent_classifier import BellmanIntentClassifier


class BankAccountDataset(Dataset):
    """은행 계좌 데이터셋"""
    
    def __init__(self, data_path, dp_path, label_encoder, max_length=14):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        with open(dp_path, 'r', encoding='utf-8') as f:
            self.dp = np.array(json.load(f))
        
        self.label_encoder = label_encoder
        self.max_length = max_length
        
        self.samples = []
        for code, accounts in self.data.items():
            for account in accounts:
                if len(account) > 14:
                    continue
                self.samples.append((account, code.strip()))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        account, code = self.samples[idx]
        
        dp_feature = self.extract_dp_feature(account)
        account_digits = self.extract_digits(account)
        account_length = len(account)
        label = self.label_encoder[code]
        
        return {
            'dp_feature': torch.from_numpy(dp_feature).float(),
            'account_digits': torch.tensor(account_digits, dtype=torch.long),
            'account_length': torch.tensor([account_length], dtype=torch.long),
            'label': torch.tensor([label], dtype=torch.long)
        }
    
    def extract_dp_feature(self, account):
        """DP 특성 추출"""
        feature_vector = np.zeros(75)
        for i, v in enumerate(account):
            v = int(v)
            feature_vector += self.dp[i][v]
        feature_vector += self.dp[len(account)-1][10]
        return feature_vector
    
    def extract_digits(self, account):
        """계좌번호 숫자 추출"""
        digits = [int(d) + 1 for d in account]
        padded = digits + [0] * (self.max_length - len(digits))
        return padded


class BellmanLagrangianLoss(nn.Module):
    """
    벨만-라그랑지안 통합 손실 함수
    
    L_total = L_classification + λ_bellman * L_bellman + λ_geodesic * L_geodesic
    
    where:
    - L_classification: Cross-entropy loss (분류 손실)
    - L_bellman: Bellman consistency loss (벨만 일관성)
    - L_geodesic: Geodesic regularization (측지선 정규화)
    
    수학적 기반:
    1. 벨만 에러: δ = V(s) - (R + γV(s'))
    2. 라그랑지안: L = T - V (운동 에너지 - 잠재 에너지)
    3. 최소 작용 원리: δ∫L dt = 0
    """
    
    def __init__(
        self,
        lambda_bellman: float = 0.1,
        lambda_geodesic: float = 0.01,
        gamma: float = 0.99
    ):
        super().__init__()
        
        self.lambda_bellman = lambda_bellman
        self.lambda_geodesic = lambda_geodesic
        self.gamma = gamma
        
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        values: torch.Tensor,
        embeddings: torch.Tensor
    ) -> dict:
        """
        Args:
            logits: (B, num_classes) - 분류 로짓
            labels: (B,) - 정답 레이블
            values: (B,) - 가치 함수 값
            embeddings: (B, D) - 하이퍼볼릭 임베딩
            
        Returns:
            losses: dict with total_loss and components
        """
        batch_size = logits.size(0)
        
        loss_cls = self.ce_loss(logits, labels)
        
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        reward = confidence.detach()
        
        target_values = reward + self.gamma * values.detach()
        
        bellman_error = (values - target_values).pow(2).mean()
        
        embedding_velocity = torch.diff(embeddings, dim=0, prepend=embeddings[:1])
        kinetic_energy = embedding_velocity.pow(2).sum(dim=-1).mean()
        
        potential_energy = -values.mean()
        
        action = kinetic_energy - potential_energy
        
        # Normalize geodesic loss to prevent explosion
        geodesic_loss = action.abs().clamp(max=10.0)
        
        total_loss = (
            loss_cls +
            self.lambda_bellman * bellman_error +
            self.lambda_geodesic * geodesic_loss
        )
        
        return {
            'total': total_loss,
            'classification': loss_cls,
            'bellman': bellman_error,
            'geodesic': geodesic_loss,
            'avg_value': values.mean(),
            'avg_confidence': confidence.mean()
        }


def calc_top_k_accuracy(y_true, logits, k):
    """Top-K 정확도 계산"""
    top_k_preds = torch.topk(logits, k, dim=1)[1]
    correct = 0
    for i, label in enumerate(y_true):
        if label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """1 에포크 학습"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'classification': 0.0,
        'bellman': 0.0,
        'geodesic': 0.0
    }
    
    all_labels = []
    all_logits = []
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch in pbar:
        dp_features = batch['dp_feature'].to(device)
        account_digits = batch['account_digits'].to(device)
        account_length = batch['account_length'].to(device)
        labels = batch['label'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        
        logits, values = model(
            dp_features,
            account_digits,
            account_length,
            return_values=True
        )
        
        embeddings = model.euclidean_to_poincare(
            model.to_hyperbolic(
                model.seq_projector(
                    torch.cat([
                        model.dp_to_euclidean(dp_features),
                        (model.digit_embed(account_digits) + 
                         model.pos_embed.unsqueeze(0).expand(dp_features.size(0), -1, -1)
                        ).mean(dim=1)
                    ], dim=-1)
                )
            )
        )
        
        losses = criterion(logits, labels, values, embeddings)
        
        losses['total'].backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        for key in total_losses:
            total_losses[key] += losses[key].item()
        
        all_labels.append(labels.cpu())
        all_logits.append(logits.detach().cpu())
        
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'cls': f"{losses['classification'].item():.4f}",
            'bell': f"{losses['bellman'].item():.4f}",
            'val': f"{losses['avg_value'].item():.2f}"
        })
    
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)
    
    acc = accuracy_score(all_labels, all_logits.argmax(dim=1))
    
    for key in total_losses:
        total_losses[key] /= len(dataloader)
    
    return total_losses, acc


def evaluate(model, dataloader, criterion, device):
    """평가"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'classification': 0.0,
        'bellman': 0.0,
        'geodesic': 0.0
    }
    
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            dp_features = batch['dp_feature'].to(device)
            account_digits = batch['account_digits'].to(device)
            account_length = batch['account_length'].to(device)
            labels = batch['label'].squeeze(1).to(device)
            
            logits, values = model(
                dp_features,
                account_digits,
                account_length,
                return_values=True
            )
            
            embeddings = model.euclidean_to_poincare(
                model.to_hyperbolic(
                    model.seq_projector(
                        torch.cat([
                            model.dp_to_euclidean(dp_features),
                            (model.digit_embed(account_digits) + 
                             model.pos_embed.unsqueeze(0).expand(dp_features.size(0), -1, -1)
                            ).mean(dim=1)
                        ], dim=-1)
                    )
                )
            )
            
            losses = criterion(logits, labels, values, embeddings)
            
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
    
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)
    
    acc = accuracy_score(all_labels, all_logits.argmax(dim=1))
    precision = precision_score(all_labels, all_logits.argmax(dim=1), average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_logits.argmax(dim=1), average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_logits.argmax(dim=1), average='weighted', zero_division=0)
    
    top2_acc = calc_top_k_accuracy(all_labels, all_logits, 2)
    top3_acc = calc_top_k_accuracy(all_labels, all_logits, 3)
    top5_acc = calc_top_k_accuracy(all_labels, all_logits, 5)
    
    for key in total_losses:
        total_losses[key] /= len(dataloader)
    
    return {
        'losses': total_losses,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top2': top2_acc,
        'top3': top3_acc,
        'top5': top5_acc
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path('data')
    
    with open(data_dir / 'code_idx.json', 'r', encoding='utf-8') as f:
        label_encoder = json.load(f)
        label_encoder = {k: int(v) for k, v in label_encoder.items()}
    
    num_classes = len(label_encoder)
    print(f"Number of classes: {num_classes}")
    
    train_dataset = BankAccountDataset(
        data_dir / 'train_data.json',
        data_dir / 'dp_data.json',
        label_encoder
    )
    
    test_dataset = BankAccountDataset(
        data_dir / 'test_data.json',
        data_dir / 'dp_data.json',
        label_encoder
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    model = BellmanIntentClassifier(
        dp_dim=75,
        embed_dim=64,
        hyp_dim=32,
        num_classes=num_classes,
        num_layers=3,
        curvature=1.0,
        gamma=0.99,
        use_spd=False
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = BellmanLagrangianLoss(
        lambda_bellman=0.1,
        lambda_geodesic=0.01,
        gamma=0.99
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    num_epochs = 100
    patience = 10
    best_acc = 0.0
    no_improve_count = 0
    
    print("\n" + "="*60)
    print("Starting Training - Bellman-Lagrangian Intent Classifier")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        
        train_losses, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"Train Loss: {train_losses['total']:.4f} "
              f"(cls: {train_losses['classification']:.4f}, "
              f"bell: {train_losses['bellman']:.4f}, "
              f"geo: {train_losses['geodesic']:.4f})")
        print(f"Train Acc: {train_acc:.4f}")
        
        eval_results = evaluate(model, test_loader, criterion, device)
        
        print(f"Test Loss: {eval_results['losses']['total']:.4f}")
        print(f"Test Acc: {eval_results['accuracy']:.4f}")
        print(f"Precision: {eval_results['precision']:.4f}, "
              f"Recall: {eval_results['recall']:.4f}, "
              f"F1: {eval_results['f1']:.4f}")
        print(f"Top-2: {eval_results['top2']:.4f}, "
              f"Top-3: {eval_results['top3']:.4f}, "
              f"Top-5: {eval_results['top5']:.4f}")
        
        if eval_results['accuracy'] > best_acc:
            best_acc = eval_results['accuracy']
            no_improve_count = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'eval_results': eval_results
            }, 'experiments/bank_prediction/models/best_bellman_model.pth')
            
            print(f"  Best model saved! (Acc: {best_acc:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement (Patience: {no_improve_count}/{patience})")
        
        if no_improve_count >= patience:
            print("\nEarly stopping triggered!")
            break
        
        scheduler.step()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Test Accuracy: {best_acc:.4f} (Epoch {epoch})")
    print(f"Total Training Time: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    
    checkpoint = torch.load('experiments/bank_prediction/models/best_bellman_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("Final Test Results")
    print("="*60)
    print(f"Accuracy:  {final_results['accuracy']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall:    {final_results['recall']:.4f}")
    print(f"F1 Score:  {final_results['f1']:.4f}")
    print(f"Top-2 Acc: {final_results['top2']:.4f}")
    print(f"Top-3 Acc: {final_results['top3']:.4f}")
    print(f"Top-5 Acc: {final_results['top5']:.4f}")


if __name__ == '__main__':
    main()

