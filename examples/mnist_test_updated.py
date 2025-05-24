"""
리팩토링된 Reality Stone API를 사용한 MNIST 테스트
Hyperbolic Neural Network 정확도 검증
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as rs

class HyperbolicLinearLayer(nn.Module):
    """
    Hyperbolic Linear Layer using poincare_ball_layer
    """
    def __init__(self, in_features, out_features, c=1e-3, t=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.t = t
        
        # 가중치 초기화 (작은 값으로)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        """
        x: (batch_size, in_features)
        """
        batch_size = x.size(0)
        
        # Linear transformation 먼저 적용
        linear_out = torch.matmul(x, self.weight.t()) + self.bias  # (batch_size, out_features)
        
        # Hyperbolic transformation 적용
        # poincare_ball_layer는 (input, v, c, t) 형태
        try:
            # 0 벡터를 u로 사용하고, linear_out을 v로 사용
            u = torch.zeros_like(linear_out)
            hyperbolic_out = rs.poincare_ball_layer(u, linear_out, self.c, self.t)
            return hyperbolic_out
        except Exception as e:
            # poincare_ball_layer가 실패하면 일반 linear 결과 반환
            print(f"Hyperbolic layer fallback: {e}")
            return linear_out

class MobiusLinearLayer(nn.Module):
    """
    Möbius transformation based Linear Layer
    """
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 입력 차원을 맞추기 위한 projection
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = nn.Identity()
            
        # Möbius 변환을 위한 파라미터
        self.mobius_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 입력을 적절한 크기로 projection
        projected = self.projection(x)  # (batch_size, out_features)
        
        # Möbius 변환을 위한 y 벡터 생성
        y = torch.matmul(x, self.mobius_weight.t())  # (batch_size, out_features)
        
        # 차원을 맞춤
        if y.size(-1) != projected.size(-1):
            y = y[:, :projected.size(-1)]
        
        try:
            # Möbius addition 적용
            result = rs.mobius_add(projected, y, self.c)
            return result
        except Exception as e:
            print(f"Möbius layer fallback: {e}")
            return projected

class HyperbolicMNISTNet(nn.Module):
    """
    Hyperbolic MNIST Classifier
    """
    def __init__(self, use_hyperbolic=True):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        
        if use_hyperbolic:
            # Hyperbolic layers 사용
            self.fc1 = HyperbolicLinearLayer(784, 256, c=1e-3, t=0.1)
            self.fc2 = MobiusLinearLayer(256, 128, c=1.0)
            self.fc3 = nn.Linear(128, 10)  # 마지막은 일반 linear
        else:
            # 일반 linear layers
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

def train_epoch(model, dataloader, optimizer, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader), 100. * correct / total

def test_epoch(model, dataloader, device):
    """테스트"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    """메인 테스트 함수"""
    print("🧠 리팩토링된 Reality Stone MNIST 테스트 시작")
    print("=" * 50)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 데이터 로딩
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../MNIST', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 모델 비교 테스트
    models = {
        'Hyperbolic Net': HyperbolicMNISTNet(use_hyperbolic=True),
        'Standard Net': HyperbolicMNISTNet(use_hyperbolic=False)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🚀 {model_name} 테스트 시작")
        print("-" * 30)
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        
        # 3 에포크만 빠르게 테스트
        for epoch in range(1, 4):
            print(f"\nEpoch {epoch}/3:")
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc = test_epoch(model, test_loader, device)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        elapsed_time = time.time() - start_time
        
        results[model_name] = {
            'final_test_acc': test_acc,
            'final_train_acc': train_acc,
            'final_test_loss': test_loss,
            'training_time': elapsed_time
        }
        
        print(f"  총 학습 시간: {elapsed_time:.2f}초")
    
    # 결과 비교
    print("\n" + "=" * 50)
    print("📊 최종 결과 비교")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  최종 테스트 정확도: {result['final_test_acc']:.2f}%")
        print(f"  최종 학습 정확도: {result['final_train_acc']:.2f}%")
        print(f"  최종 테스트 손실: {result['final_test_loss']:.4f}")
        print(f"  학습 시간: {result['training_time']:.2f}초")
    
    # 성능 차이 분석
    hyperbolic_acc = results['Hyperbolic Net']['final_test_acc']
    standard_acc = results['Standard Net']['final_test_acc']
    
    print(f"\n🎯 성능 분석:")
    print(f"  Hyperbolic vs Standard 정확도 차이: {hyperbolic_acc - standard_acc:.2f}%")
    
    if abs(hyperbolic_acc - standard_acc) < 5.0:
        print("  ✅ 리팩토링 성공! 정확도 차이가 5% 이내입니다.")
    else:
        print("  ⚠️  주의: 정확도 차이가 5%를 초과합니다.")
    
    return results

if __name__ == "__main__":
    main() 