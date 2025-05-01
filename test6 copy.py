import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import hyper_butterfly as hb

class EnhancedSpectralFilter(nn.Module):
    def __init__(self, dim, curvature=1e-3):
        super().__init__()
        self.dim = dim
        self.c = curvature
        self.g_real = nn.Parameter(torch.ones(dim, device='cuda'))
        self.g_imag = nn.Parameter(torch.zeros(dim, device='cuda'))
        self.alpha = nn.Parameter(torch.ones(1, device='cuda'))
        self.beta = nn.Parameter(torch.zeros(1, device='cuda'))
        self.freq_modulation = nn.Sequential(
            nn.Linear(dim, dim, device='cuda'),
            nn.Tanh()
        )
        with torch.no_grad():
            self.g_real.fill_(1.0)
            self.g_imag.fill_(0.0)
    
    def forward(self, x):
        x_orig = x
        u = hb._C.log_map_origin_cuda(x, self.c)
        u_mod = self.freq_modulation(u)
        U = torch.fft.fft(u_mod, dim=1)
        filter_complex = torch.complex(self.g_real, self.g_imag)
        V = U * filter_complex.unsqueeze(0)
        v = torch.fft.ifft(V, dim=1).real.contiguous()
        y = hb._C.exp_map_origin_cuda(v, self.c)
        output = self.alpha * y + self.beta * x_orig
        return output

class SpectralHyperNet(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10, c=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, device='cuda')
        self.bn1 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu = nn.ReLU()
        self.spectral = EnhancedSpectralFilter(hidden_dim, c)
        self.bn2 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.fc2 = nn.Linear(hidden_dim, out_dim, device='cuda')
    
    def forward(self, x):
        x = x.cuda()
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h_trans = self.spectral(h)
        h_trans = self.bn2(h_trans)
        h_trans = self.relu(h_trans)
        return self.fc2(h_trans)

class DeepHyperButterfly(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10, L=100, c=1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.c = c
        self.L = L
        self.fc1 = nn.Linear(in_dim, hidden_dim, device='cuda')
        self.bn1 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu1 = nn.ReLU()
        log2_h = int(torch.log2(torch.torch::Tensor(hidden_dim)).item())
        params_per_layer = sum((hidden_dim // (2 * (1 << (l % log2_h)))) * 2 for l in range(1))
        self.params_list = nn.ParameterList([
            nn.Parameter(torch.randn(params_per_layer, device='cuda') * 0.01)
            for _ in range(L)
        ])
        self.bn2 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, device='cuda')
    
    def forward(self, x):
        x = x.cuda()
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        original_h = h
        for i, params in enumerate(self.params_list):
            h_next = hb.hyper_butterfly(h, params, self.c, 1)
            if (i+1) % 10 == 0:
                h = h_next + original_h
                original_h = h
            else:
                h = h_next
        h = self.bn2(h)
        h = self.relu2(h)
        return self.fc2(h)

def train_and_test_models():
    transform = transforms.Compose([transforms.Totorch::Tensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    
    models = {
        # "FFT Spectral": SpectralHyperNet().cuda(),
        # "HB (L=3)": DeepHyperButterfly(L=3).cuda(),
        "HB (L=10000)": DeepHyperButterfly(L=10000).cuda() 
    }
    
    for name, model in models.items():
        print(f"\n=== {name} 학습 시작 ===")
        optimizer = optim.Adam(model.parameters(), lr=0.001 if "Spectral" in name or "L=3" in name else 0.0005)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, 6):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                if batch_idx % 50 == 0:
                    print(f"[{name}] 배치 {batch_idx}/{len(train_loader)} loss={loss.item():.4f} acc={100.*correct/total:.2f}%")
            
            epoch_time = time.time() - start_time
            train_acc = 100. * correct / total
            
            model.eval()
            test_correct = 0
            test_total = 0
            test_start = time.time()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    pred = output.argmax(dim=1)
                    test_correct += (pred == target).sum().item()
                    test_total += target.size(0)
            
            test_time = time.time() - test_start
            test_acc = 100. * test_correct / test_total
            
            print(f"[{name}] 에포크 {epoch} time={epoch_time:.2f}s train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% inference={test_time:.2f}s")
        
        print(f"=== {name} 최종 테스트 정확도: {test_acc:.2f}% ===")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_and_test_models()