import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs

# 🔥 통합 상수 정의 - 하드코딩 제거
class HyperbolicConfig:
    # 기본 하이퍼볼릭 파라미터
    BASE_CURVATURE = 1e-3
    DEFAULT_T = 0.7
    DEFAULT_L = 2
    
    # 동적 곡률 범위 (원본 기준 상대적)
    DYNAMIC_CURVATURE_MIN_RATIO = 0.01   # BASE_CURVATURE * 0.1
    DYNAMIC_CURVATURE_MAX_RATIO = 10000.0  # BASE_CURVATURE * 10.0
    CONSERVATIVE_MIN_RATIO = 0.5        # 보수적 버전
    CONSERVATIVE_MAX_RATIO = 2.0
    
    # 수치 안정성
    GRADIENT_CLIP_NORM = 1.0
    SAFE_CLAMP_MIN = 0.01
    SAFE_CLAMP_MAX = 100.0
    NAN_FALLBACK_ENABLED = True
    
    # 체비셰프 파라미터
    CHEBYSHEV_ORDER = 25
    CHEBYSHEV_SCALE = 0.5
    CHEBYSHEV_OFFSET = 0.5
    
    # 모델 초기화
    WEIGHT_INIT_STD_ORIGINAL = 0.01
    WEIGHT_INIT_STD_IMPROVED = 0.02
    
    # 학습 파라미터
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 256
    EPOCHS = 10

# 기존 원본 모델
class GeodesicMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, 
                 c=HyperbolicConfig.BASE_CURVATURE, 
                 L=HyperbolicConfig.DEFAULT_L, 
                 t=HyperbolicConfig.DEFAULT_T):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        
        # 원본과 동일한 초기화
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
        
        if HyperbolicConfig.NAN_FALLBACK_ENABLED and torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🔥 개선된 체비셰프 모델
class ChebyshevMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, 
                 c=HyperbolicConfig.BASE_CURVATURE, 
                 L=HyperbolicConfig.DEFAULT_L, 
                 t=HyperbolicConfig.DEFAULT_T):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        
        # 개선된 초기화
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        
        # 체비셰프 근사 적용
        h = rs.chebyshev_approximation(h, order=HyperbolicConfig.CHEBYSHEV_ORDER, curvature=self.c)
        
        u = h @ self.weights2 + self.bias2
        u = rs.chebyshev_approximation(
            u * HyperbolicConfig.CHEBYSHEV_SCALE, 
            order=HyperbolicConfig.CHEBYSHEV_ORDER, 
            curvature=self.c
        ) * HyperbolicConfig.CHEBYSHEV_SCALE + HyperbolicConfig.CHEBYSHEV_OFFSET
        
        z = rs.poincare_ball_layer(h, u, self.c, self.t)
        
        if HyperbolicConfig.NAN_FALLBACK_ENABLED and torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🚀 수정된 동적 곡률 모델 - 하이퍼볼릭 인식 버전
class ImprovedDynamicCurvatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, 
                 c=HyperbolicConfig.BASE_CURVATURE, 
                 L=HyperbolicConfig.DEFAULT_L, 
                 t=HyperbolicConfig.DEFAULT_T):
        super().__init__()
        self.base_c = c
        self.L = L
        self.t = t
        
        # 개선된 초기화
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 🔥 하이퍼볼릭 인식 곡률 예측기 (유클리디안 선형 레이어 제거)
        self.curvature_features = nn.Parameter(torch.randn(16) * 0.01)
        self.curvature_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 곡률 범위 (상수 사용)
        self.min_curvature = self.base_c * HyperbolicConfig.DYNAMIC_CURVATURE_MIN_RATIO
        self.max_curvature = self.base_c * HyperbolicConfig.DYNAMIC_CURVATURE_MAX_RATIO

    def predict_adaptive_curvature(self, x_flat):
        """하이퍼볼릭 기하학을 고려한 곡률 예측"""
        batch_size = x_flat.size(0)
        
        # 1. 입력의 기하학적 특성 (하이퍼볼릭 노름)
        norms = torch.norm(x_flat, dim=1, keepdim=True)  # [B, 1]
        
        # 2. 학습 가능한 특징과의 상호작용 (차원 수정)
        # 입력을 16차원으로 축소하여 feature와 매칭
        x_reduced = torch.mm(x_flat, torch.randn(x_flat.size(1), 16, device=x_flat.device))  # [B, 16]
        feature_interaction = torch.sum(x_reduced * self.curvature_features.unsqueeze(0), dim=1, keepdim=True)  # [B, 1]
        
        # 3. 안전한 곡률 범위로 스케일링
        curvature_adjustment = torch.sigmoid(feature_interaction + self.curvature_scale)  # [B, 1]
        curvatures = self.min_curvature + (self.max_curvature - self.min_curvature) * curvature_adjustment  # [B, 1]
        
        return curvatures.squeeze(-1)  # [B]

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)  
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        try:
            # 🔥 개선된 동적 곡률 예측
            curvatures = self.predict_adaptive_curvature(x_flat)
            
            # 배치별 개별 처리 (평균화 제거)
            z_list = []
            for i in range(h.size(0)):
                h_i = h[i:i+1]
                u_i = u[i:i+1]
                c_i = torch.clamp(curvatures[i], self.min_curvature, self.max_curvature).item()
                
                z_i = rs.poincare_ball_layer(h_i, u_i, c_i, self.t)
                if torch.isnan(z_i).any():
                    z_i = h_i
                z_list.append(z_i)
            
            z = torch.cat(z_list, dim=0)
            
        except Exception as e:
            print(f"🔧 Improved dynamic curvature failed: {e}")
            z = rs.poincare_ball_layer(h, u, self.base_c, self.t)
            if HyperbolicConfig.NAN_FALLBACK_ENABLED and torch.isnan(z).any():
                z = h
        
        output = z @ self.out_weights + self.out_bias
        return output

# 🎯 보수적 동적 곡률 모델 - 곡률 범위만 축소
class ConservativeDynamicCurvatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, 
                 c=HyperbolicConfig.BASE_CURVATURE, 
                 L=HyperbolicConfig.DEFAULT_L, 
                 t=HyperbolicConfig.DEFAULT_T):
        super().__init__()
        self.base_c = c
        self.L = L
        self.t = t
        
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 보수적 곡률 예측기 (원래 방식 유지하되 범위만 축소)
        self.curvature_predictor = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 보수적 곡률 범위
        self.min_curvature = self.base_c * HyperbolicConfig.CONSERVATIVE_MIN_RATIO
        self.max_curvature = self.base_c * HyperbolicConfig.CONSERVATIVE_MAX_RATIO

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)  
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        try:
            # 🔥 보수적 곡률 예측 (원본 근처 범위)
            c_pred = self.curvature_predictor(x_flat)
            c_range = self.min_curvature + (self.max_curvature - self.min_curvature) * c_pred
            c_avg = torch.clamp(c_range.mean(), self.min_curvature, self.max_curvature).item()
            
            z = rs.poincare_ball_layer(h, u, c_avg, self.t)
            if torch.isnan(z).any():
                z = h
                
        except Exception as e:
            print(f"🔧 Conservative curvature failed: {e}")
            z = rs.poincare_ball_layer(h, u, self.base_c, self.t)
            if HyperbolicConfig.NAN_FALLBACK_ENABLED and torch.isnan(z).any():
                z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🧪 단순 동적 곡률 모델 - 학습 가능한 스케일링
class SimpleDynamicCurvatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, 
                 c=HyperbolicConfig.BASE_CURVATURE, 
                 L=HyperbolicConfig.DEFAULT_L, 
                 t=HyperbolicConfig.DEFAULT_T):
        super().__init__()
        self.base_c = c
        self.L = L  
        self.t = t
        
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 🎯 가장 간단한 접근: 학습 가능한 곡률 스케일
        self.curvature_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 🔥 단순한 적응적 곡률
        adaptive_c = self.base_c * (HyperbolicConfig.CONSERVATIVE_MIN_RATIO + 
                                   (HyperbolicConfig.CONSERVATIVE_MAX_RATIO - HyperbolicConfig.CONSERVATIVE_MIN_RATIO) * 
                                   torch.sigmoid(self.curvature_scale).item())
        
        z = rs.poincare_ball_layer(h, u, adaptive_c, self.t)
        if HyperbolicConfig.NAN_FALLBACK_ENABLED and torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🎪 초단순 동적 곡률 모델 - 글로벌 스케일링만
class SuperSimpleDynamicCurvatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, 
                 c=HyperbolicConfig.BASE_CURVATURE, 
                 L=HyperbolicConfig.DEFAULT_L, 
                 t=HyperbolicConfig.DEFAULT_T):
        super().__init__()
        self.base_c = c
        self.L = L  
        self.t = t
        
        std = HyperbolicConfig.WEIGHT_INIT_STD_IMPROVED
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * std)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * std)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * std)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 🎯 초단순: 하나의 곡률 배율만 학습
        self.curvature_multiplier = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = x_flat @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 🔥 가장 단순한 적응적 곡률: base_c * learnable_multiplier
        adaptive_c = self.base_c * torch.clamp(self.curvature_multiplier, 
                                             HyperbolicConfig.CONSERVATIVE_MIN_RATIO, 
                                             HyperbolicConfig.CONSERVATIVE_MAX_RATIO)
        
        z = rs.poincare_ball_layer(h, u, adaptive_c, self.t)
        if HyperbolicConfig.NAN_FALLBACK_ENABLED and torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 🔥 최적화된 훈련 함수
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    
    # 🚀 성능 최적화
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        try:
            # Mixed precision training (CUDA only)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = nn.functional.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HyperbolicConfig.GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
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
    print(f"\n--- {model_name} Training ---")
    test_accs = []
    
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        scheduler.step()
        
        print(f"[{model_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    
    best_acc = max(test_accs) * 100
    print(f"[{model_name}] Best accuracy: {best_acc:.2f}%")
    return best_acc

def check_reality_stone():
    print("=== Reality Stone Status Check (Improved) ===")
    try:
        x = torch.randn(4, 10)
        result = rs.poincare_ball_layer(x, x, HyperbolicConfig.BASE_CURVATURE, HyperbolicConfig.DEFAULT_T)
        print("✓ poincare_ball_layer: OK")
        
        try:
            result = rs.chebyshev_approximation(x, order=HyperbolicConfig.CHEBYSHEV_ORDER, curvature=1.0)
            print("✓ chebyshev_approximation: OK")
        except Exception as e:
            print(f"✗ chebyshev_approximation: {e}")
            
        # 동적 곡률 테스트 (적절한 범위)
        try:
            features = torch.norm(x, dim=1, keepdim=True)
            weight = torch.randn(1, 1) * 0.1
            bias = torch.zeros(1)
            result = rs.predict_dynamic_curvature(
                features, weight, bias, 
                base_curvature=HyperbolicConfig.BASE_CURVATURE,    
                min_curvature=HyperbolicConfig.BASE_CURVATURE * HyperbolicConfig.DYNAMIC_CURVATURE_MIN_RATIO,    
                max_curvature=HyperbolicConfig.BASE_CURVATURE * HyperbolicConfig.DYNAMIC_CURVATURE_MAX_RATIO    
            )
            print(f"✓ dynamic_curvature_pred: OK, range=[{result.min():.2e}, {result.max():.2e}]")
        except Exception as e:
            print(f"✗ dynamic_curvature_pred: {e}")
            
        # 경계 페널티 테스트
        try:
            penalty = rs.boundary_penalty(x, curvature=HyperbolicConfig.BASE_CURVATURE, epsilon=0.001)
            print(f"✓ boundary_penalty: OK, value={penalty.item():.6f}")
        except Exception as e:
            print(f"✗ boundary_penalty: {e}")
            
    except Exception as e:
        print(f"✗ Reality Stone basic test failed: {e}")
    print("="*60)

if __name__ == "__main__":
    # 라이브러리 상태 체크
    check_reality_stone()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 상수 사용
    batch_size = HyperbolicConfig.BATCH_SIZE
    lr = HyperbolicConfig.LEARNING_RATE
    epochs = HyperbolicConfig.EPOCHS
    
    # 🚀 빠른 테스트 모드 (필요시 활성화)
    QUICK_TEST = True  # 빠른 테스트용
    if QUICK_TEST:
        epochs = 10
        print("🚀 Quick Test Mode: 10 epochs only")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    
    # 🚀 성능 최적화된 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # 🔥 다양한 모델 비교 (다이나믹 모델 우선 테스트)
    models = {
        "🎪 SuperSimpleDynamic": SuperSimpleDynamicCurvatureMLP(),
        "🧪 SimpleDynamic": SimpleDynamicCurvatureMLP(),
        "🛡️ ConservativeDynamic": ConservativeDynamicCurvatureMLP(),
        "📊 Original": GeodesicMLP(),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        model = model.to(device)
        try:
            acc = train_model(name, model, train_loader, test_loader, epochs, lr, device)
            results[name] = acc
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = 0.0
    
    # 🎯 결과 분석
    print(f"\n{'='*60}")
    print("🎯 COMPREHENSIVE RESULTS")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_results:
        if acc > 95:
            status = "🔥 EXCELLENT"
        elif acc > 90:
            status = "✅ GOOD"
        elif acc > 80:
            status = "⚠️  FAIR"
        else:
            status = "❌ POOR"
        print(f"{name:25}: {acc:6.2f}% {status}")
    
    # 개선도 분석
    if '📊 Original' in results and results['📊 Original'] > 0:
        orig_acc = results["📊 Original"]
        print(f"\n📈 Improvements over Original ({orig_acc:.2f}%):")
        
        for name, acc in results.items():
            if name != '📊 Original' and acc > 0:
                improvement = acc - orig_acc
                symbol = "🔥" if improvement > 1.0 else "✅" if improvement > 0 else "❌"
                print(f"{symbol} {name:25}: {improvement:+5.2f}%")
    
    print(f"\n🔍 DIAGNOSIS:")
    best_name, best_acc = max(results.items(), key=lambda x: x[1])
    print(f"🏆 Best Model: {best_name} ({best_acc:.2f}%)")
    
    if best_acc > 95:
        print("✅ Excellent performance achieved!")
    elif best_acc > 90:
        print("✅ Good performance - room for improvement")
    else:
        print("⚠️  Performance below expectations - check implementation")
        
    print(f"\n🔧 Configuration Used:")
    print(f"   Base Curvature: {HyperbolicConfig.BASE_CURVATURE}")
    print(f"   Dynamic Range: {HyperbolicConfig.DYNAMIC_CURVATURE_MIN_RATIO}x - {HyperbolicConfig.DYNAMIC_CURVATURE_MAX_RATIO}x")
    print(f"   Conservative Range: {HyperbolicConfig.CONSERVATIVE_MIN_RATIO}x - {HyperbolicConfig.CONSERVATIVE_MAX_RATIO}x")
    print(f"   Learning Rate: {HyperbolicConfig.LEARNING_RATE}")
    print(f"   Batch Size: {HyperbolicConfig.BATCH_SIZE}")