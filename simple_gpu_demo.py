import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

# Enhanced debugging at start
print("=" * 50)
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("GPU memory:")
    print(torch.cuda.memory_summary())
print("=" * 50)

# Add debugging print at the start of the file
print("Script execution started")

class HyperbolicNetwork(nn.Module):
    """
    하이퍼볼릭 공간에서 동작하는 간단한 네트워크
    입력을 하이퍼볼릭 공간으로 매핑하고, 하이퍼볼릭 거리 기반 손실 함수 사용
    """
    def __init__(self, input_dim, hidden_dim, output_dim, curvature=0.1):
        super(HyperbolicNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        
        # 유클리드 공간에서의 선형 변환
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        
        # 초기화
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
    
    def to_poincare_ball(self, x, c=None):
        """유클리드 벡터를 포인카레 볼로 투영"""
        if c is None:
            c = self.curvature
            
        # 벡터 정규화 (하이퍼볼릭 공간의 경계 안으로)
        max_norm = (1.0 - 1e-5) / np.sqrt(c)
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-10)
        
        # tanh를 사용하여 모든 점이 경계 내부에 있도록 매핑
        coef = torch.tanh(x_norm) / x_norm * max_norm
        return x * coef
    
    def poincare_distance(self, x, y, c=None):
        """포인카레 볼 모델에서의 거리 계산"""
        if c is None:
            c = self.curvature
            
        # 안정성을 위한 클램핑
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
        eps = 1e-10
        
        # 각 점의 노름 계산
        x_norm_sq = torch.sum(x * x, dim=1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=1, keepdim=True)
        
        # 두 점 사이의 유클리드 거리 제곱
        xy_dist_sq = torch.sum((x - y) * (x - y), dim=1, keepdim=True)
        
        # 분자, 분모 계산
        num = 2 * xy_dist_sq
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq) + eps
        
        # 거리 계산 공식
        return torch.acosh(1 + c * num / denom + eps) / torch.sqrt(c)
    
    def forward(self, x):
        # 유클리드 공간에서의 변환
        h1 = torch.tanh(self.linear1(x))
        h2 = torch.tanh(self.linear2(h1))
        y_euclidean = self.linear3(h2)
        
        # 하이퍼볼릭 공간으로 매핑
        y_hyperbolic = self.to_poincare_ball(y_euclidean)
        
        return y_hyperbolic

def generate_spiral_data(n_samples=1000, n_classes=2, noise=0.1):
    """
    나선형 데이터 생성 - 비선형 분리가 필요한 예제 데이터
    하이퍼볼릭 공간에서의 장점을 보여주기 위함
    """
    X = np.zeros((n_samples*n_classes, 2))
    y = np.zeros(n_samples*n_classes, dtype=int)
    
    for j in range(n_classes):
        ix = range(n_samples*j, n_samples*(j+1))
        r = np.linspace(0.0, 1, n_samples)  # 반지름
        t = np.linspace(j*4, (j+1)*4, n_samples) + np.random.randn(n_samples)*noise  # 각도
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    
    return X, y

def train_model(model, X_train, y_train, device, epochs=100, batch_size=32, lr=0.001):
    """모델 학습"""
    # 데이터를 텐서로 변환
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 손실 함수 (CrossEntropy는 값을 확률로 변환하므로 여기서 직접 구현)
    def hyperbolic_loss(outputs, targets):
        # 클래스 중심점 (각 클래스의 원형)
        n_classes = len(torch.unique(targets))
        centers = []
        for i in range(n_classes):
            # 각 클래스별 중심점 생성 (하이퍼볼릭 공간에서)
            angle = (2 * np.pi * i) / n_classes
            radius = 0.5 * model.curvature**0.5  # 적절한 반지름
            center = torch.tensor([[np.sin(angle) * radius, np.cos(angle) * radius]], 
                                  device=device, dtype=outputs.dtype)
            centers.append(center)
        
        class_centers = torch.cat(centers, dim=0)
        
        # 각 샘플을 각 클래스 중심점과의 거리 계산
        loss = 0
        for i in range(len(outputs)):
            # 현재 샘플
            out = outputs[i:i+1]
            target = targets[i]
            
            # 정답 클래스 중심점과의 거리
            pos_dist = model.poincare_distance(out, class_centers[target:target+1])
            
            # 다른 클래스 중심점과의 거리
            neg_indices = [j for j in range(n_classes) if j != target]
            neg_centers = class_centers[neg_indices]
            neg_dists = model.poincare_distance(out.repeat(len(neg_indices), 1), neg_centers)
            
            # 손실 계산 (정답 클래스와는 가깝게, 다른 클래스와는 멀게)
            hinge_loss = torch.clamp(1.0 - neg_dists + pos_dist, min=0.0)
            loss += hinge_loss.sum()
        
        return loss / len(outputs)
    
    # 학습 이력
    losses = []
    times = []
    
    # 학습 루프
    for epoch in range(epochs):
        start_time = time.time()
        
        # 배치 학습
        permutation = torch.randperm(X_tensor.size(0))
        total_loss = 0
        
        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]
            
            # 순전파
            outputs = model(batch_x)
            loss = hyperbolic_loss(outputs, batch_y)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 시간 측정
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        
        # 손실 기록
        avg_loss = total_loss / (X_tensor.size(0) // batch_size)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.4f}s")
    
    return losses, times

def visualize_results(model, X, y, device, title=""):
    """결과 시각화"""
    model.eval()
    
    # 시각화를 위한 그리드 생성
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 그리드 포인트 예측
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.zeros(grid.shape[0])
    
    # 배치 단위로 예측 (메모리 효율성)
    batch_size = 1000
    for i in range(0, grid.shape[0], batch_size):
        batch = torch.FloatTensor(grid[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch)
            
            # 각 클래스 중심점 (모델에 정의된 대로)
            n_classes = len(np.unique(y))
            centers = []
            for j in range(n_classes):
                angle = (2 * np.pi * j) / n_classes
                radius = 0.5 * model.curvature**0.5
                center = torch.tensor([[np.sin(angle) * radius, np.cos(angle) * radius]], 
                                     device=device, dtype=outputs.dtype)
                centers.append(center)
            
            class_centers = torch.cat(centers, dim=0)
            
            # 각 클래스 중심점과의 거리 계산
            for j in range(len(batch)):
                out = outputs[j:j+1]
                dists = []
                
                for k in range(n_classes):
                    dist = model.poincare_distance(out, class_centers[k:k+1])
                    dists.append(dist.item())
                
                # 가장 가까운 중심점의 클래스 할당
                Z[i+j] = np.argmin(dists)
    
    # 분류 결과 시각화
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    
    # 실제 데이터 포인트
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.savefig('hyperbolic_classification.png')
    
    # 모델 임베딩 시각화
    plt.figure(figsize=(6, 6))
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    
    plt.scatter(outputs[:, 0], outputs[:, 1], c=y, cmap=plt.cm.Spectral)
    
    # 포인카레 디스크 경계 그리기
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1.0
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    plt.plot(x_circle, y_circle, 'k-', alpha=0.3)
    
    # 클래스 중심점 표시
    n_classes = len(np.unique(y))
    for i in range(n_classes):
        angle = (2 * np.pi * i) / n_classes
        r = 0.5 * model.curvature**0.5
        plt.plot(np.sin(angle) * r, np.cos(angle) * r, 'ko', markersize=10)
    
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title("Hyperbolic Embeddings")
    plt.savefig('hyperbolic_embeddings.png')

def main():
    print("===== 하이퍼볼릭 네트워크 GPU 데모 =====")
    
    # CUDA 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU instead.")
    
    try:
        # 데이터 생성 (나선형 구분 문제)
        print("Generating spiral data...")
        X, y = generate_spiral_data(n_samples=1000, n_classes=3, noise=0.1)
        print(f"데이터 형태: {X.shape}, 레이블: {len(np.unique(y))} 클래스")
        
        # 모델 초기화
        print("Initializing model...")
        model = HyperbolicNetwork(
            input_dim=2, 
            hidden_dim=32,
            output_dim=2,  # 2D 하이퍼볼릭 공간에 투영
            curvature=0.1
        ).to(device)
        
        print(f"모델 초기화 완료, 파라미터 수: {sum(p.numel() for p in model.parameters())}")
        
        # 모델 학습
        print("\n학습 시작...")
        losses, times = train_model(
            model, 
            X, 
            y, 
            device,
            epochs=100,
            batch_size=64,
            lr=0.005
        )
        
        print(f"학습 완료! 평균 에폭 시간: {np.mean(times):.4f}초")
        
        # 결과 시각화
        print("결과 시각화 중...")
        visualize_results(model, X, y, device, title="Hyperbolic Network Classification")
        print("시각화 완료! 'hyperbolic_classification.png'와 'hyperbolic_embeddings.png' 파일을 확인하세요.")
        
        # 손실 그래프
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig('hyperbolic_loss.png')
        print("손실 그래프를 'hyperbolic_loss.png' 파일로 저장했습니다.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting main function")
    main()
    print("Script completed") 