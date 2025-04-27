import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
import matplotlib.pyplot as plt

# 파이썬 구현만 사용하도록 수정
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 순수 Python 구현을 위한 모듈
import torch
from torch.autograd import Function

# 순수 Python 폴백 구현 - riemannian_manifold에서 가져온 코드
def py_poincare_exp_map(x, v, c=1.0):
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / (1.0 - c * x_norm_squared)
    
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=1e-8)
    
    second_term = torch.tanh(torch.sqrt(torch.tensor(c)) * lambda_x * v_norm / 2.0) / (torch.sqrt(torch.tensor(c)) * v_norm) * v
    
    numerator = (1.0 - c * x_norm_squared) * second_term
    denominator = 1.0 - 2.0 * c * torch.sum(x * second_term, dim=-1, keepdim=True) + c * c * x_norm_squared * torch.sum(second_term * second_term, dim=-1, keepdim=True)
    
    return x + numerator / denominator

def py_poincare_log_map(x, y, c=1.0):
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / (1.0 - c * x_norm_squared)
    
    diff = y - x
    diff_norm_squared = torch.sum(diff * diff, dim=-1, keepdim=True)
    y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
    
    transport_vector = (-x * y_norm_squared + y * (1.0 + c * x_norm_squared) - 2 * c * torch.sum(x * y, dim=-1, keepdim=True) * x) / (1.0 - c * x_norm_squared)
    transport_norm = torch.norm(transport_vector, p=2, dim=-1, keepdim=True)
    
    numerator = 2 * torch.sqrt(torch.tensor(c)) * torch.atanh(torch.sqrt(torch.tensor(c)) * transport_norm)
    denominator = torch.sqrt(torch.tensor(c)) * lambda_x * transport_norm
    
    return numerator / denominator * transport_vector

def py_poincare_distance(x, y, c=1.0):
    norm_x = torch.sum(x * x, dim=-1, keepdim=True)
    norm_y = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
    
    numerator = 2 * torch.sqrt(torch.tensor(c)) * torch.norm(x - y, p=2, dim=-1, keepdim=True)
    denominator = torch.sqrt((1 - c * norm_x) * (1 - c * norm_y)) + torch.sqrt(c) * xy_inner
    
    return 2 * torch.atanh(numerator / denominator) / torch.sqrt(torch.tensor(c))

def py_butterfly_factor(input_tensor, params, layer):
    n = input_tensor.size(0)
    block_size = 1 << layer
    num_blocks = n // block_size
    
    result = input_tensor.clone()
    
    param_idx = 0
    total_params = params.size(0)
    
    for b in range(num_blocks):
        for i in range(0, block_size, 2):
            if b * block_size + i + 1 >= n: 
                break
            if param_idx + 1 >= total_params: 
                break
            
            idx = b * block_size + i
            a = params[param_idx].item()
            b_val = params[param_idx + 1].item()
            param_idx += 2
            
            temp1 = a * input_tensor[idx] + b_val * input_tensor[idx + 1]
            temp2 = -b_val * input_tensor[idx] + a * input_tensor[idx + 1]
            
            result[idx] = temp1
            result[idx + 1] = temp2
    
    return result

# Hyper-Butterfly 레이어 클래스 재구현
class HyperButterflyLayer(nn.Module):
    def __init__(self, dim, num_layers, curvature=1.0):
        super(HyperButterflyLayer, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.curvature = curvature
        
        # 파라미터 초기화
        params = []
        for l in range(num_layers):
            block_size = 1 << l
            num_blocks = dim // block_size
            for b in range(num_blocks):
                a = nn.Parameter(torch.ones(1) * 0.9)  # a 초기값
                b = nn.Parameter(torch.zeros(1))       # b 초기값
                params.extend([a, b])
        
        self.params = nn.ParameterList(params)
    
    def forward(self, x):
        u = py_poincare_log_map(torch.zeros_like(x), x, self.curvature)
        
        for l in range(self.num_layers):
            block_size = 1 << l
            num_blocks = self.dim // block_size
            params_needed = num_blocks * 2
            
            # 실제 사용 가능한 파라미터 추출
            layer_params = torch.cat([p for p in self.params[l*2*num_blocks:(l+1)*2*num_blocks]])
            u = py_butterfly_factor(u, layer_params, l)
        
        return py_poincare_exp_map(torch.zeros_like(u), u, self.curvature)

# MNIST 데이터셋 로드
def load_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 일반 MLP 모델
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Hyper-Butterfly 기반 모델
class HyperButterflyMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(HyperButterflyMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 입력 크기를 2의 거듭제곱으로 조정
        log2_dim = int(np.ceil(np.log2(hidden_dim)))
        self.butterfly_dim = 2 ** log2_dim
        
        # 입력층
        self.fc1 = nn.Linear(input_dim, self.butterfly_dim)
        
        # Hyper-Butterfly 레이어
        # 레이어 수 = log2(dimension)
        self.hyper_butterfly = HyperButterflyLayer(
            dim=self.butterfly_dim,
            num_layers=log2_dim,
            curvature=0.5
        )
        
        # 출력층
        self.fc2 = nn.Linear(self.butterfly_dim, output_dim)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        
        # 배치 처리를 위해 각 샘플을 개별적으로 처리
        outputs = []
        for i in range(x.size(0)):
            out = self.hyper_butterfly(x[i])
            outputs.append(out)
        
        x = torch.stack(outputs)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 모델 학습 함수
def train(model, train_loader, optimizer, epoch, device, log_interval=100):
    model.train()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_time = time.time() - start_time
    return train_time

# 모델 테스트 함수
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_time = time.time() - start_time
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_time, accuracy

def run_experiment(model_name, epochs=5, batch_size=128):
    # 데이터 로드
    train_loader, test_loader = load_mnist(batch_size)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # 모델 초기화
    if model_name == 'mlp':
        model = MLP().to(device)
    elif model_name == 'hyper_butterfly':
        model = HyperButterflyMLP().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습 및 평가
    train_times = []
    test_times = []
    accuracies = []
    
    for epoch in range(1, epochs + 1):
        train_time = train(model, train_loader, optimizer, epoch, device)
        test_time, accuracy = test(model, test_loader, device)
        
        train_times.append(train_time)
        test_times.append(test_time)
        accuracies.append(accuracy)
    
    # 결과 출력
    print(f"Model: {model_name}")
    print(f"Average training time per epoch: {np.mean(train_times):.2f} seconds")
    print(f"Average inference time on test set: {np.mean(test_times):.2f} seconds")
    print(f"Final accuracy: {accuracies[-1]:.2f}%")
    
    return {
        'model': model_name,
        'train_times': train_times,
        'test_times': test_times,
        'accuracies': accuracies
    }

# 메인 함수
if __name__ == "__main__":
    # 모델 비교 실험
    print("Running MLP experiment...")
    mlp_results = run_experiment('mlp', epochs=5)
    
    print("\nRunning Hyper-Butterfly experiment...")
    hb_results = run_experiment('hyper_butterfly', epochs=5)
    
    # 결과 시각화
    epochs = range(1, len(mlp_results['accuracies']) + 1)
    
    # 정확도 비교
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlp_results['accuracies'], 'bo-', label='MLP')
    plt.plot(epochs, hb_results['accuracies'], 'ro-', label='Hyper-Butterfly')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 학습 시간 비교
    plt.subplot(1, 2, 2)
    plt.bar(['MLP', 'Hyper-Butterfly'], 
            [np.mean(mlp_results['train_times']), np.mean(hb_results['train_times'])],
            color=['blue', 'red'])
    plt.title('Average Training Time per Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mnist_comparison.png')
    plt.show()
    
    print("Experiment completed. Results saved to mnist_comparison.png") 