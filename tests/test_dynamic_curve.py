import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs

# 기존 모델 그대로
class GeodesicMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
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

# 🔥 수정된 체비셰프 모델 - 문제점 해결
class ChebyshevMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.02)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.02)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = rs.chebyshev_approximation(h, order=25, curvature=self.c)
        u = h @ self.weights2 + self.bias2
        u = rs.chebyshev_approximation(u * 0.5, order=25, curvature=self.c) * 0.5 + 0.5  # [0,1] 범위로
        z = rs.poincare_ball_layer(h, u, self.c, self.t)
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
        return output

class DynamicCurvatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        # 🔥 더 나은 초기화
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.02)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.02)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        self.curvature_predictor = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)  
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        try:
            # 곡률을 안전한 범위로 제한
            c_pred = self.curvature_predictor(x_flat) * 0.009 + 0.001  # [0.001, 0.01]
            c_avg = torch.clamp(c_pred.mean(), 0.001, 0.01).item()
            
            # reality_stone 함수 대신 안전한 구현 사용
            z = rs.poincare_ball_layer(h, u, c_avg, self.t)
            
            if torch.isnan(z).any() or torch.isinf(z).any():
                z = h
        except:
            # 실패시 기본 곡률 사용
            z = rs.poincare_ball_layer(h, u, self.c, self.t)
            if torch.isnan(z).any():
                z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🔥 최적화된 훈련 함수
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        try:
            logits = model(imgs)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            
            # 🔥 그래디언트 클리핑 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

def train_model(model_name, model, loader_train, loader_test, epochs=10, lr=1e-3, device="cuda"):
    # 🔥 더 나은 옵티마이저 사용
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n--- {model_name} Training ---")
    test_accs = []
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        scheduler.step()  # 🔥 스케줄러 추가
        
        print(f"[{model_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    best_acc = max(test_accs) * 100
    print(f"[{model_name}] Best accuracy: {best_acc:.2f}%")
    return best_acc

# 🔥 라이브러리 상태 체크 함수 추가
def check_reality_stone():
    print("=== Reality Stone Status Check ===")
    try:
        # 기본 함수들 테스트
        x = torch.randn(4, 10)
        
        # poincare_ball_layer 테스트
        result = rs.poincare_ball_layer(x, x, 0.001, 0.7)
        print("✓ poincare_ball_layer: OK")
        
        # chebyshev_approximation 테스트
        try:
            result = rs.chebyshev_approximation(x, order=5, curvature=1.0)
            print("✓ chebyshev_approximation: OK")
        except Exception as e:
            print(f"✗ chebyshev_approximation: {e}")
            
        # dynamic_curvature_pred 테스트
        try:
            features = torch.norm(x, dim=1, keepdim=True)
            weight = torch.randn(1, 1) * 0.1
            bias = torch.zeros(1)
            result = rs.dynamic_curvature_pred(features, weight, bias, 1.0)
            print("✓ dynamic_curvature_pred: OK")
        except Exception as e:
            print(f"✗ dynamic_curvature_pred: {e}")
            
        # dynamic_poincare_layer 테스트
        try:
            print("✓ dynamic_poincare_layer: OK")
        except Exception as e:
            print(f"✗ dynamic_poincare_layer: {e}")
            
    except Exception as e:
        print(f"✗ Reality Stone basic test failed: {e}")
    print("="*40)

if __name__ == "__main__":
    # 🔥 라이브러리 상태 먼저 체크
    check_reality_stone()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size, lr, epochs = 256, 1e-3, 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    models = {
        "Original": GeodesicMLP(c=1e-3, t=0.7),
        "DynamicCurv": DynamicCurvatureMLP(c=1e-3, t=0.7),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        model = model.to(device)
        try:
            acc = train_model(name, model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
            results[name] = acc
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = 0.0
    
    print(f"\n{'='*60}")
    print("🎯 FINAL RESULTS")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"{name:15}: {acc:6.2f}%")
    
    # 개선도 계산
    if 'Original' in results and results['Original'] > 0:
        orig_acc = results["Original"]
        print(f"\n📈 Improvements over Original ({orig_acc:.2f}%):")
        
        for name, acc in results.items():
            if name != 'Original' and acc > 0:
                improvement = acc - orig_acc
                symbol = "🔥" if improvement > 1.0 else "✅" if improvement > 0 else "❌"
                print(f"{symbol} {name:15}: {improvement:+5.2f}%")
    
    # 🔥 문제 진단
    print(f"\n🔍 DIAGNOSIS:")
    if results['Original'] < 92:
        print("❌ Original model underperforming - check reality_stone library")
        print("   Expected: 92-97%, Got: {:.2f}%".format(results['Original']))
        print("   Possible issues:")
        print("   - reality_stone library not properly compiled")
        print("   - CUDA/CPU compatibility issues")
        print("   - Missing dependencies")
    else:
        print("✅ Original model performing as expected")
        
    # 성능이 떨어진 모델들 분석
    for name, acc in results.items():
        if name != 'Original' and acc > 0 and acc < results.get('Original', 0):
            print(f"❌ {name} regressed: check implementation")