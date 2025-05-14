import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs

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

class LorentzMLP(nn.Module):
    """
    로렌츠 모델을 사용한 다층 퍼셉트론
    
    로렌츠 모델은 (n+1)차원 민코프스키 공간에서 초곡면 상의 점들을 사용합니다.
    특징:
    - 수치적으로 안정적
    - 곡률 계산이 정확함
    - 차원이 n+1로 증가함 (시간 성분 추가)
    """
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        
        # 입력 차원 -> 은닉 차원 투영
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        
        # 로렌츠 공간 내부의 변환
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        
        # 출력 차원으로 투영 (로렌츠 모델은 차원이 하나 더 큼)
        self.out_weights = nn.Parameter(torch.randn(hid+1, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # 입력 형태 변환
        x = x.view(x.size(0), -1)
        
        # 첫 번째 레이어 (일반 선형 변환)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        
        # 두 번째 레이어 (다시 일반 선형 변환)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 로렌츠 공간으로 변환 (푸앵카레 → 로렌츠 좌표계 변환)
        lorentz_h = rs.poincare_to_lorentz(h, self.c)
        lorentz_u = rs.poincare_to_lorentz(u, self.c)
        
        # 로렌츠 측지선 연산
        lorentz_z = rs.lorentz_layer(lorentz_h, lorentz_u, self.c, self.t)
        
        # NaN 처리
        if torch.isnan(lorentz_z).any():
            lorentz_z = lorentz_h
        
        # 출력 레이어
        output = lorentz_z @ self.out_weights + self.out_bias
        return output

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset), time.time() - t0

def test_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(loader.dataset)

# 모델 훈련 함수
def train_model(model_name, model, loader_train, loader_test, epochs=10, lr=1e-3, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # t값을 모델 이름에 추가
    display_name = model_name
    if hasattr(model, 't'):
        display_name = f"{model_name} (t={model.t})"
    
    print(f"\n--- {display_name} Training ---")
    test_accs = []  # 테스트 정확도 기록
    
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        print(f"[{display_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    
    # 최종 성능 출력
    best_acc = max(test_accs) * 100
    print(f"[{display_name}] Best accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, lr, epochs = 256, 1e-3, 10
    
    # 데이터 로딩
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        # 2. 로렌츠 MLP 테스트
    print("\n===== 로렌츠 모델 테스트 =====")
    lorentz_t_values = [0.5, 0.7, 1.0]  # 동일한 t값으로 테스트
    lorentz_results = {}
    for t in lorentz_t_values:
        model = LorentzMLP(c=1e-3, t=t).to(device)
        acc = train_model(f"LorentzMLP", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
        lorentz_results[t] = acc
    # 1. 먼저 GeodesicMLP (푸앵카레 볼) 테스트
    print("\n===== 푸앵카레 볼 모델 테스트 =====")
    poincare_t_values = [0.5, 0.7, 1.0]  # 간소화된 t값 리스트
    geodesic_results = {}
    for t in poincare_t_values:
        model = GeodesicMLP(c=1e-3, L=2, t=t).to(device)
        acc = train_model(f"GeodesicMLP", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
        geodesic_results[t] = acc
    # 결과 요약
    print("\n=== 결과 요약 ===")
    
    print("\n푸앵카레 볼 모델 정확도 (t값에 따른 비교):")
    for t, acc in sorted(geodesic_results.items()):
        print(f"GeodesicMLP (t = {t}): {acc:.2f}%")
    best_poincare_t = max(geodesic_results.items(), key=lambda x: x[1])[0]
    best_poincare_acc = geodesic_results[best_poincare_t]
    
    print("\n로렌츠 모델 정확도 (t값에 따른 비교):")
    for t, acc in sorted(lorentz_results.items()):
        print(f"LorentzMLP (t = {t}): {acc:.2f}%")
    best_lorentz_t = max(lorentz_results.items(), key=lambda x: x[1])[0]
    best_lorentz_acc = lorentz_results[best_lorentz_t]
    
    # 모델 간 비교
    print("\n모델 간 최고 성능 비교:")
    print(f"최고 푸앵카레 볼 모델: t = {best_poincare_t}, 정확도 = {best_poincare_acc:.2f}%")
    print(f"최고 로렌츠 모델: t = {best_lorentz_t}, 정확도 = {best_lorentz_acc:.2f}%")
    
    # 최고 성능 모델 선정
    if best_poincare_acc > best_lorentz_acc:
        print(f"\n최고 성능 모델: 푸앵카레 볼 (t = {best_poincare_t}, 정확도 = {best_poincare_acc:.2f}%)")
    else:
        print(f"\n최고 성능 모델: 로렌츠 (t = {best_lorentz_t}, 정확도 = {best_lorentz_acc:.2f}%)")