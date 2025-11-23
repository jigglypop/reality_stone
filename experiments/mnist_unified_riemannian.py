#!/usr/bin/env python3
"""
MNIST Classification with Unified Riemannian Layer

4가지 메트릭으로 MNIST 분류 성능 비교:
- Poincare Ball (푸앵카레 볼)
- Lorentz (로렌츠 하이퍼볼로이드)
- Klein (클라인 사영)
- Diagonal (학습 가능한 대각 메트릭)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import reality_stone as rs

# Euclidean baseline
class EuclideanMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    
    def forward(self, x):
        return self.net(x)


class RiemannianLayerFunction(torch.autograd.Function):
    """Autograd-compatible wrapper for Riemannian layer"""
    
    @staticmethod
    def forward(ctx, x, layer):
        # Save for backward
        ctx.layer = layer
        ctx.save_for_backward(x)
        
        # Forward pass
        x_np = x.detach().cpu().numpy().astype(np.float32)
        y_np, _ = layer.forward(x_np)
        y = torch.from_numpy(y_np).to(x.device)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        layer = ctx.layer
        
        # Backward pass
        grad_np = grad_output.detach().cpu().numpy().astype(np.float32)
        x_np = x.detach().cpu().numpy().astype(np.float32)
        grad_input_np = layer.backward(grad_np, x_np)
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        
        return grad_input, None


class RiemannianMNIST(nn.Module):
    """Unified Riemannian Layer를 사용한 MNIST 분류기"""
    
    def __init__(self, metric_type="poincare", hidden_dim=128, curvature=1.0):
        super().__init__()
        self.metric_type = metric_type
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(28*28, hidden_dim)
        self.act1 = nn.ReLU()
        
        # Hidden transformation (학습 능력 추가)
        self.hidden_trans = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        
        # Riemannian layers
        self.layer1 = rs.UnifiedRiemannianLayer(
            metric_type=metric_type,
            curvature=curvature,
            input_dim=hidden_dim,
            enable_bellman=False
        )
        
        self.layer2 = rs.UnifiedRiemannianLayer(
            metric_type=metric_type,
            curvature=curvature,
            input_dim=hidden_dim,
            enable_bellman=False
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Input Projection
        x = x.view(batch_size, -1)
        x = self.input_proj(x)
        x = self.act1(x)
        
        # 2. First Riemannian Block
        # 접공간에서의 선형 변환이라 가정하고 Linear 적용 후 리만 공간(Layer) 통과
        x = self.hidden_trans(x)
        x = self.act2(x)
        
        # Unified Layer (Metric Space Projection / Geometry enforcement)
        # tanh로 값을 제한하여(-1~1) 하이퍼볼릭 공간 내로 매핑 유도
        x = torch.tanh(x) * 0.9 
        x = RiemannianLayerFunction.apply(x, self.layer1)
        
        # 3. Second Block & Output
        # 사실상 마지막 Layer2는 Identity지만, 구조적 대칭성을 위해 유지하거나 생략 가능
        # 여기서는 바로 출력으로 연결
        logits = self.output_proj(x)
        return logits


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def run_experiment(metric_type, hidden_dim=128, epochs=10, batch_size=128, lr=0.001):
    print(f"\n{'='*70}")
    print(f"Training with {metric_type.upper()} metric")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 전체 데이터 사용 (정확도 평가용)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    if metric_type == 'euclidean':
        model = EuclideanMLP(hidden_dim=hidden_dim).to(device)
    else:
        curvature = 1.0 if metric_type != 'diagonal' else 0.0
        model = RiemannianMNIST(
            metric_type=metric_type,
            hidden_dim=hidden_dim,
            curvature=curvature
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    best_acc = 0
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'time': []
    }
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['time'].append(epoch_time)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    print(f"Average Time per Epoch: {np.mean(results['time']):.2f}s")
    
    return results, best_acc


def main():
    print("="*70)
    print("MNIST Classification with Unified Riemannian Layers")
    print("="*70)
    
    # Configuration (baseline 수준 정확도를 목표로)
    hidden_dim = 128
    epochs = 10
    batch_size = 128
    lr = 0.001
    
    print(f"\nConfiguration:")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    # Metrics to test (리만 메트릭 우선, 유클리드는 마지막에)
    metrics = ['poincare', 'lorentz', 'klein', 'diagonal', 'euclidean']
    
    all_results = {}
    best_accuracies = {}
    
    for metric in metrics:
        results, best_acc = run_experiment(
            metric_type=metric,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        all_results[metric] = results
        best_accuracies[metric] = best_acc
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Metric':<15} | {'Best Acc':<10} | {'Avg Time':<10}")
    print("-"*70)
    
    for metric in metrics:
        acc = best_accuracies[metric]
        avg_time = np.mean(all_results[metric]['time'])
        print(f"{metric.upper():<15} | {acc:>9.2f}% | {avg_time:>9.2f}s")
    
    print("="*70)
    
    # Find best
    best_metric = max(best_accuracies, key=best_accuracies.get)
    print(f"\n🏆 Best Metric: {best_metric.upper()} ({best_accuracies[best_metric]:.2f}%)")
    
    # Save results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        for metric in metrics:
            axes[0].plot(all_results[metric]['test_acc'], 
                        label=f'{metric.upper()}', marker='o', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Test Accuracy by Metric')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        for metric in metrics:
            axes[1].plot(all_results[metric]['test_loss'], 
                        label=f'{metric.upper()}', marker='o', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('Test Loss by Metric')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mnist_unified_riemannian_results.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved plot to 'mnist_unified_riemannian_results.png'")
    except ImportError:
        print("\n(matplotlib not available, skipping plot)")


if __name__ == "__main__":
    main()

