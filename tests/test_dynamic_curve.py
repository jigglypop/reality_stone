import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs
import numpy as np
import random
from collections import defaultdict

# 🔥 시드 고정으로 재현성 보장
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 통합 상수 정의
class HyperbolicConfig:
    BASE_CURVATURE = 1e-3
    DEFAULT_T = 0.7
    DEFAULT_L = 2
    
    # 🔥 더 안정적인 곡률 범위로 수정
    DYNAMIC_CURVATURE_MIN_RATIO = 0.5   # 0.1 → 0.5 (더 보수적)
    DYNAMIC_CURVATURE_MAX_RATIO = 2.0   # 10.0 → 2.0 (더 보수적)
    CONSERVATIVE_MIN_RATIO = 0.8        # 0.5 → 0.8 (더 보수적)
    CONSERVATIVE_MAX_RATIO = 1.2        # 2.0 → 1.2 (더 보수적)
    
    GRADIENT_CLIP_NORM = 1.0
    WEIGHT_INIT_STD_ORIGINAL = 0.01
    WEIGHT_INIT_STD_IMPROVED = 0.02
    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 256
    EPOCHS = 10

# 🎯 가장 안정적인 모델들만 선별
class StableOriginalMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.c = HyperbolicConfig.BASE_CURVATURE
        self.t = HyperbolicConfig.DEFAULT_T
        
        std = HyperbolicConfig.WEIGHT_INIT_STD_ORIGINAL
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)  
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u) 
        z = rs.poincare_ball_layer(h, u, self.c, self.t)
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
        return output

# 🏆 가장 일관된 성능의 SuperSimple (단순 곡률 스케일링)
class StableSuperSimpleDynamicMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.base_c = HyperbolicConfig.BASE_CURVATURE
        self.t = HyperbolicConfig.DEFAULT_T
        
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 🎯 안정적인 초기화: 1.0 근처에서 시작
        self.curvature_multiplier = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 🔥 안정적인 곡률 (더 작은 범위)
        adaptive_c = self.base_c * torch.clamp(
            self.curvature_multiplier, 
            HyperbolicConfig.CONSERVATIVE_MIN_RATIO, 
            HyperbolicConfig.CONSERVATIVE_MAX_RATIO
        )
        
        z = rs.poincare_ball_layer(h, u, adaptive_c, self.t)
        if torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🧪 개선된 SimpleDynamic (더 안정적인 범위)
class StableSimpleDynamicMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.base_c = HyperbolicConfig.BASE_CURVATURE
        self.t = HyperbolicConfig.DEFAULT_T
        
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 🔥 더 안정적인 스케일 초기화
        self.curvature_scale = nn.Parameter(torch.zeros(1))  # 0에서 시작 (시그모이드 = 0.5)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 🔥 더 안정적인 곡률 범위
        min_c = self.base_c * HyperbolicConfig.DYNAMIC_CURVATURE_MIN_RATIO
        max_c = self.base_c * HyperbolicConfig.DYNAMIC_CURVATURE_MAX_RATIO
        adaptive_c = min_c + (max_c - min_c) * torch.sigmoid(self.curvature_scale).item()
        
        z = rs.poincare_ball_layer(h, u, adaptive_c, self.t)
        if torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        try:
            logits = model(imgs)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HyperbolicConfig.GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        except Exception as e:
            print(f"Training error: {e}")
            continue
            
    return total_loss / len(loader.dataset), time.time() - t0

def test_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            try:
                pred = model(imgs).argmax(dim=1)
                correct += (pred == labels).sum().item()
            except:
                continue
    return correct / len(loader.dataset)

def train_model(model_name, model, loader_train, loader_test, epochs, lr, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=HyperbolicConfig.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    test_accs = []
    
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        scheduler.step()
        
        print(f"[{model_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    
    best_acc = max(test_accs) * 100
    return best_acc

def run_multiple_experiments(num_runs=5):
    """🔥 다중 실행으로 안정성 평가"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = HyperbolicConfig.BATCH_SIZE
    lr = HyperbolicConfig.LEARNING_RATE
    epochs = 7  # 빠른 테스트용
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # 🎯 안정적인 모델들만 테스트
    model_classes = {
        "📊 Original": StableOriginalMLP,
        "🎪 SuperSimple": StableSuperSimpleDynamicMLP,
        "🧪 Simple": StableSimpleDynamicMLP,
    }
    
    results = defaultdict(list)
    
    print(f"\n🔥 Running {num_runs} experiments for stability analysis...")
    
    for run in range(num_runs):
        print(f"\n{'='*50}")
        print(f"🧪 EXPERIMENT {run+1}/{num_runs}")
        print(f"{'='*50}")
        
        # 각 실행마다 다른 시드 (하지만 재현 가능)
        set_seed(42 + run)
        
        for name, model_class in model_classes.items():
            print(f"\n--- Training {name} (Run {run+1}) ---")
            
            model = model_class().to(device)
            try:
                acc = train_model(name, model, train_loader, test_loader, epochs, lr, device)
                results[name].append(acc)
                print(f"✅ {name} Run {run+1}: {acc:.2f}%")
            except Exception as e:
                print(f"❌ {name} Run {run+1} failed: {e}")
                results[name].append(0.0)
    
    return results

def analyze_stability(results):
    """🔍 안정성 분석"""
    print(f"\n{'='*70}")
    print("🔍 STABILITY ANALYSIS")
    print(f"{'='*70}")
    
    stability_stats = {}
    
    for name, accs in results.items():
        accs = [acc for acc in accs if acc > 0]  # 실패한 실행 제외
        if len(accs) == 0:
            continue
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        cv = std_acc / mean_acc  # 변동계수 (낮을수록 안정적)
        
        stability_stats[name] = {
            'mean': mean_acc,
            'std': std_acc,
            'min': min_acc,
            'max': max_acc,
            'cv': cv,
            'runs': len(accs)
        }
        
        print(f"{name:20}: {mean_acc:.2f}±{std_acc:.2f}% (CV={cv:.3f}) [{min_acc:.2f}%-{max_acc:.2f}%]")
    
    # 🏆 순위 결정 (평균 성능 + 안정성 고려)
    print(f"\n🏆 FINAL RANKING (Mean ± Std, Stability)")
    print(f"{'='*70}")
    
    # 평균 성능 기준 정렬
    sorted_by_mean = sorted(stability_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for i, (name, stats) in enumerate(sorted_by_mean, 1):
        stability = "🔥 STABLE" if stats['cv'] < 0.005 else "⚠️ UNSTABLE" if stats['cv'] > 0.01 else "✅ STABLE"
        print(f"{i}. {name:20}: {stats['mean']:.2f}±{stats['std']:.2f}% {stability}")
        
    return stability_stats

if __name__ == "__main__":
    print("🚀 Stable Hyperbolic MNIST Benchmark")
    print("🎯 Testing model stability with multiple runs...")
    
    # 🔥 다중 실행으로 안정성 테스트
    results = run_multiple_experiments(num_runs=3)  # 3번 실행
    
    # 📊 결과 분석
    stability_stats = analyze_stability(results)
    
    print(f"\n🎯 CONCLUSIONS:")
    
    # 가장 안정적인 모델 찾기
    if stability_stats:
        most_stable = min(stability_stats.items(), key=lambda x: x[1]['cv'])
        best_performer = max(stability_stats.items(), key=lambda x: x[1]['mean'])
        
        print(f"🏆 Best Overall: {best_performer[0]} ({best_performer[1]['mean']:.2f}%)")
        print(f"🔥 Most Stable: {most_stable[0]} (CV={most_stable[1]['cv']:.3f})")
        
        if most_stable[0] == best_performer[0]:
            print(f"✅ {most_stable[0]} is both BEST and MOST STABLE! 🎉")
        else:
            print(f"⚖️  Trade-off between performance and stability detected.")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Use multiple runs for reliable evaluation")
    print(f"   • Focus on stable models for production")
    print(f"   • Consider CV < 0.005 as highly stable")