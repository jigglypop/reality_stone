import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as rs
from reality_stone.models import LorentzMLP, KleinMLP, HybridHyperbolicMLP

def train_epoch(model, loader, optimizer, device):
    """한 에포크 훈련"""
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
    """테스트 세트 평가"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(loader.dataset)

def train_model(model_name, model, loader_train, loader_test, epochs=10, lr=1e-3, device="cuda"):
    """모델 훈련"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # t값을 모델 이름에 추가 (t 파라미터가 있는 경우)
    display_name = model_name
    if hasattr(model, 't'):
        display_name = f"{model_name} (t={model.t})"
    
    print(f"\n--- {display_name} 훈련 ---")
    test_accs = []  # 테스트 정확도 기록
    
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        print(f"[{display_name}] 에포크 {ep}/{epochs} 손실={loss:.4f} 시간={t:.2f}초 정확도={acc*100:.2f}%")
    
    # 최종 성능 출력
    best_acc = max(test_accs) * 100
    print(f"[{display_name}] 최고 정확도: {best_acc:.2f}%")
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
    
    # # 기존 GeodesicMLP (푸앵카레 볼)
    # geodesic_results = {}
    # model = rs.GeodesicMLP(c=1e-3, L=2, t=0.7).to(device)
    # acc = train_model("GeodesicMLP", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    # geodesic_results['poincare'] = acc
    
    # 로렌츠 모델
    lorentz_results = {}
    t_values = [0.5, 0.7, 1.0]
    # 클라인 모델
    klein_results = {}
    for t in t_values:
        model = KleinMLP(c=1e-3, t=t).to(device)
        acc = train_model(f"KleinMLP", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
        klein_results[t] = acc

    for t in t_values:
        model = LorentzMLP(c=1e-3, t=t).to(device)
        acc = train_model(f"LorentzMLP", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
        lorentz_results[t] = acc
    

    
    # 하이브리드 모델
    hybrid_results = {}
    for mode in ['poincare', 'lorentz', 'klein', 'auto']:
        model = HybridHyperbolicMLP(c=1e-3, t=0.7, mode=mode).to(device)
        acc = train_model(f"HybridMLP ({mode})", model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
        hybrid_results[mode] = acc
    
    # 결과 요약
    print("\n=== 결과 요약 ===")
    
    print("\n로렌츠 모델 정확도 (t값에 따른 비교):")
    for t, acc in sorted(lorentz_results.items()):
        print(f"LorentzMLP (t = {t}): {acc:.2f}%")
    
    print("\n클라인 모델 정확도 (t값에 따른 비교):")
    for t, acc in sorted(klein_results.items()):
        print(f"KleinMLP (t = {t}): {acc:.2f}%")
    
    print("\n하이브리드 모델 정확도 (모드에 따른 비교):")
    for mode, acc in hybrid_results.items():
        print(f"HybridMLP ({mode}): {acc:.2f}%")
    
    best_lorentz = max(lorentz_results.values())
    best_klein = max(klein_results.values())
    best_hybrid = max(hybrid_results.values())
    
    results = {
        "로렌츠": best_lorentz,
        "클라인": best_klein,
        "하이브리드": best_hybrid
    }
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n최적의 모델: {best_model[0]} (정확도: {best_model[1]:.2f}%)")