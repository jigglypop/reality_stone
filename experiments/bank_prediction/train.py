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

from experiments.bank_prediction.models.hyperbolic_bank_classifier import (
    HyperbolicBankClassifier,
)
from experiments.bank_prediction.models.adaptive_curvature_bank_classifier import (
    AdaptiveCurvatureBankClassifier,
)
from experiments.bank_prediction.models.riemannian_bank_encoder import (
    RiemannianBankEncoder
)
from experiments.bank_prediction.models.product_manifold_bank_classifier import (
    ProductManifoldBankClassifier
)


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
        
        # Convert to tensors once
        # Note: We use clone().detach() or just torch.tensor() for safety
        # Using numpy arrays first is usually faster for instantiation
        return {
            'dp_feature': torch.from_numpy(dp_feature).float(),
            'account_digits': torch.tensor(account_digits, dtype=torch.long),
            'account_length': torch.tensor([account_length], dtype=torch.long),
            'label': torch.tensor([label], dtype=torch.long)
        }
    
    def extract_dp_feature(self, account):
        """DP 특성 추출 (LightGBM과 동일)"""
        feature_vector = np.zeros(75)
        for i, v in enumerate(account):
            v = int(v)
            feature_vector += self.dp[i][v]
        feature_vector += self.dp[len(account)-1][10]
        return feature_vector
    
    def extract_digits(self, account):
        """계좌번호 숫자 추출 (14자리 패딩)"""
        digits = [int(d) + 1 for d in account]  # 1-10 (0은 패딩용)
        padded = digits + [0] * (self.max_length - len(digits))
        return padded


def calc_top_k_accuracy(y_true, logits, k):
    """Top-K 정확도 계산"""
    top_k_preds = torch.topk(logits, k, dim=1)[1]
    correct = 0
    for i, label in enumerate(y_true):
        if label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """1 에포크 학습"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    batch_losses = []
    batch_accs = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        dp_features = batch['dp_feature'].to(device)
        account_digits = batch['account_digits'].to(device)
        account_length = batch['account_length'].squeeze(1).to(device)
        labels = batch['label'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        
        logits = model(dp_features, account_digits, account_length)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_losses.append(loss.item())
        
        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == labels).float().mean().item()
        batch_accs.append(batch_acc)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 매 10 배치마다 중간 결과 출력
        if (batch_idx + 1) % 10 == 0:
            recent_loss = sum(batch_losses[-10:]) / len(batch_losses[-10:])
            recent_acc = sum(batch_accs[-10:]) / len(batch_accs[-10:])
            pbar.set_postfix({
                'loss': f'{recent_loss:.4f}',
                'acc': f'{recent_acc:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, num_classes=54):
    """평가"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            dp_features = batch['dp_feature'].to(device)
            account_digits = batch['account_digits'].to(device)
            account_length = batch['account_length'].squeeze(1).to(device)
            labels = batch['label'].squeeze(1).to(device)
            
            logits = model(dp_features, account_digits, account_length)
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.LongTensor(all_labels)
    
    top_k_accuracies = []
    for k in [2, 3, 4, 5]:
        top_k_acc = calc_top_k_accuracy(all_labels_tensor, all_logits, k)
        top_k_accuracies.append(top_k_acc)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top_2_acc': top_k_accuracies[0],
        'top_3_acc': top_k_accuracies[1],
        'top_4_acc': top_k_accuracies[2],
        'top_5_acc': top_k_accuracies[3]
    }


# 프로젝트 루트 및 데이터 디렉토리 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'


def train(
    model_type='hyperbolic',
    train_data_path=str(DATA_DIR / 'train_data.json'),
    test_data_path=str(DATA_DIR / 'test_data.json'),
    dp_path=str(DATA_DIR / 'dp_data.json'),
    code_idx_path=str(DATA_DIR / 'code_idx.json'),
    num_epochs=100,
    batch_size=512,
    lr=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """학습 메인 함수"""
    
    print(f"Device: {device}")
    print(f"Model Type: {model_type}")
    
    if model_type == 'riemannian' and lr == 0.001:
        lr = 0.003
    
    with open(code_idx_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        label_map = {k: int(v) for k, v in label_map.items()}
    
    num_classes = len(label_map)
    print(f"Number of classes: {num_classes}")
    
    train_dataset = BankAccountDataset(
        train_data_path, dp_path, label_map, max_length=14
    )
    test_dataset = BankAccountDataset(
        test_data_path, dp_path, label_map, max_length=14
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if model_type == 'hyperbolic':
        model = HyperbolicBankClassifier(
            dp_dim=75,
            hyp_dim=32,
            num_classes=num_classes,
            curvature=-1.0,
            num_heads=4
        )
    elif model_type == 'adaptive':
        model = AdaptiveCurvatureBankClassifier(
            dp_dim=75,
            hyp_dim=32,
            num_classes=num_classes,
            num_layers=3,
            c_min=-2.0,
            c_max=-0.1,
            num_heads=4
        )
    elif model_type == 'riemannian':
        model = RiemannianBankEncoder(
            dp_dim=75,
            spd_dim=8,
            hyp_dim=32,
            num_classes=num_classes,
            curvature=-1.0,
            top_k=5
        )
    elif model_type == 'product_manifold':
        model = ProductManifoldBankClassifier(
            dp_dim=75,
            spd_dim=8,
            hyp_dim=32,
            num_classes=num_classes,
            curvature=-1.0,
            top_k=5
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    best_test_acc = 0.0
    best_epoch = 0
    
    # Early Stopping parameters
    patience = 5
    min_delta = 0.001
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Training Start")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
        
        scheduler.step(test_metrics['accuracy'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        print(f"Top-2: {test_metrics['top_2_acc']:.4f}, Top-3: {test_metrics['top_3_acc']:.4f}, Top-4: {test_metrics['top_4_acc']:.4f}, Top-5: {test_metrics['top_5_acc']:.4f}")
        
        if test_metrics['accuracy'] > best_test_acc + min_delta:
            best_test_acc = test_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0  # Reset counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc,
                'test_metrics': test_metrics
            }, f'experiments/bank_prediction/models/best_model_{model_type}.pth')
            print(f"  → Best model saved! (Acc: {best_test_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement (Patience: {patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    print(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f}min)")
    
    checkpoint = torch.load(f'experiments/bank_prediction/models/best_model_{model_type}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    
    print("\n" + "="*60)
    print("Final Test Results")
    print("="*60)
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")
    print(f"Top-2 Acc: {final_metrics['top_2_acc']:.4f}")
    print(f"Top-3 Acc: {final_metrics['top_3_acc']:.4f}")
    print(f"Top-4 Acc: {final_metrics['top_4_acc']:.4f}")
    print(f"Top-5 Acc: {final_metrics['top_5_acc']:.4f}")
    
    return model, final_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='hyperbolic', choices=['hyperbolic', 'adaptive', 'riemannian', 'product_manifold'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    train(
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

