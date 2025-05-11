import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs

#
# 1) Geodesic-Butterfly MLP (새로운 geodesic_layer 사용)
#
class GeodesicMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7):
        super().__init__()
        self.c = c
        self.L = L  # 호환성 유지
        self.t = t
        # 학습 가능한 파라미터 직접 정의
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        # 두 번째 변환 경로용 파라미터
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        # 출력용 파라미터
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)  
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u) 
        z = rs.geodesic_layer(h, u, self.c, self.t)
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
        return output

# class GeodesicMLP(nn.Module):
#     def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hid)
#         self.fc2 = nn.Linear(hid, out_dim)
#         self.c = c
#         self.L = L
#         self.t = t
#         
#         # butterfly params 계산 (GeodesicButterflyLayer와 동일)
#         log2_h = int(torch.log2(torch.tensor(float(hid))).item())
#         total_params = sum((hid // (2 * (1 << (l % log2_h)))) * 2 for l in range(L))
#         self.params = nn.Parameter(torch.randn(total_params) * 1e-3)
# 
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         h = torch.relu(self.fc1(x))
#         
#         # 1) butterfly transform 적용하여 u 생성
#         u = rs.reality_stone(h, self.params, self.c, self.L)
#         
#         # 2) geodesic_layer 적용
#         z = rs.geodesic_layer(h, u, self.c, self.t)
#         
#         if torch.isnan(z).any():
#             z = torch.relu(h)
#             
#         return self.fc2(z)

#
# 2) Hyper-Butterfly MLP
#
class HyperMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=1):
        super().__init__()
        self.c = c
        self.L = L
        self.fc1 = nn.Linear(in_dim, hid)
        # butterfly params 개수 계산
        log2_h = int(torch.log2(torch.tensor(float(hid))).item())
        total_params = sum((hid // (2 * (1 << (l % log2_h)))) * 2 for l in range(L))
        print(f"[Hyper] hid={hid}, total_params={total_params}")
        self.params = nn.Parameter(torch.randn(total_params) * 1e-3)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        u = rs.reality_stone(h, self.params, self.c, self.L)
        if torch.isnan(u).any():
            u = torch.relu(h)
        return self.fc2(u)

#
# 3) Euclid MLP
#
class EuclidMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


# 학습/평가 함수
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
    
    # t값을 모델 이름에 추가 (GeodesicMLP인 경우)
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

    # 다양한 t값으로 GeodesicMLP 테스트
    t_values = [0.5, 0.7, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    geodesic_results = {}
    
    for t in t_values:
        model = GeodesicMLP(c=1e-3, L=2, t=t).to(device)
        acc = train_model(f"GeodesicMLP", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
        geodesic_results[t] = acc
    
    # HyperMLP 테스트
    hy = HyperMLP(c=1e-3, L=1).to(device)
    hyper_acc = train_model("HyperMLP", hy, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    
    # EuclidMLP 테스트
    eu = EuclidMLP().to(device)
    euclid_acc = train_model("EuclidMLP", eu, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    
    # 결과 요약
    print("\n=== 결과 요약 ===")
    print(f"HyperMLP 최고 정확도: {hyper_acc:.2f}%")
    print(f"EuclidMLP 최고 정확도: {euclid_acc:.2f}%")
    print("\nGeodesicMLP 정확도 (t값에 따른 비교):")
    for t, acc in sorted(geodesic_results.items()):
        print(f"t = {t}: {acc:.2f}%")

    # 최적의 t값 찾기
    best_t = max(geodesic_results.items(), key=lambda x: x[1])[0]
    print(f"\n최적의 t값: {best_t} (정확도: {geodesic_results[best_t]:.2f}%)")
    
    # 모델 간 비교
    best_geodesic_acc = max(geodesic_results.values())
    print("\n모델 성능 순위:")
    models = [("HyperMLP", hyper_acc), ("최적 GeodesicMLP", best_geodesic_acc), ("EuclidMLP", euclid_acc)]
    for i, (name, acc) in enumerate(sorted(models, key=lambda x: x[1], reverse=True), 1):
        print(f"{i}위: {name} - {acc:.2f}%")