"""
MNIST Graph Diffusion using Reality Stone (Riemannian Lagrangian Diffusion)

이 실험은 "그래프 위에서 확산시키는" Reality Stone의 메인 아이디어를
Rust+CUDA 리만 디퓨전 엔진을 활용해 구현한 버전입니다.

- 노드: 2048개의 히든 노드 + 10개의 클래스 노드
- 그래프 에지: 학습 가능한 인접 행렬 W ∈ R^{(2048+10)×(2048+10)}
- 동역학:
    h_{t+1} = RiemannianDiffusionStep( h_t, tanh(h_t @ W) )
- 구현:
    - 상태 업데이트는 Reality Stone의 Rust+CUDA 커널이 담당 (PyRiemannianDiffusion)
    - 그래프 구조 W만 PyTorch/Adam으로 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import reality_stone as rs


# Device 설정
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("CUDA Not Available, using CPU")


class RiemannianDiffusionStep(torch.autograd.Function):
    """
    Reality Stone의 리만 디퓨전 엔진을 감싸는 PyTorch Autograd Function.

    Forward:
        - CUDA 커널이 수행하는 연산 (각 성분 i):
              v_i      = (1 - alpha) * (flow_i - h_i)
              h_next_i = h_i + v_i * dt
          (이후 컴포넌트-wise 클리핑으로 Poincaré Ball 제약을 근사).

    Backward:
        - 위 식의 선형 부분에 대한 정확한 야코비안 사용 (클리핑은 무시한 근사):
              h_next = a * h + b * flow
              a = 1 - (1 - alpha) * dt
              b = (1 - alpha) * dt
          ⇒ ∂h_next/∂h   = a
             ∂h_next/∂flow = b
    """

    @staticmethod
    def forward(ctx, h, flow, diffusion_engine, alpha, dt):
        # 연속 메모리 보장
        h = h.contiguous()
        flow = flow.contiguous()

        # 출력 텐서 (GPU 상에 미리 할당)
        h_next = torch.empty_like(h)

        batch_size, dim = h.shape

        # CUDA 사용 가능 시: Rust+CUDA 커널 직접 호출
        if h.is_cuda and getattr(rs, "_has_cuda", False):
            diffusion_engine.step_cuda(
                h.data_ptr(),
                flow.data_ptr(),
                h_next.data_ptr(),
                batch_size,
                dim,
            )
        else:
            # CPU Fallback (Numpy ↔ Rust)
            h_np = h.detach().cpu().numpy().astype("float32")
            flow_np = flow.detach().cpu().numpy().astype("float32")
            h_next_np = diffusion_engine.step_cpu(h_np, flow_np)
            h_next = torch.from_numpy(h_next_np).to(h.device)

        ctx.save_for_backward(h, flow)
        ctx.alpha = float(alpha)
        ctx.dt = float(dt)
        return h_next

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        dt = ctx.dt
        # h_next = (1 - (1-alpha) * dt) * h + ((1-alpha) * dt) * flow
        a = 1.0 - (1.0 - alpha) * dt  # d h_next / d h
        b = (1.0 - alpha) * dt        # d h_next / d flow

        grad_h = grad_output * a
        grad_flow = grad_output * b

        # diffusion_engine, alpha, dt 는 학습 대상이 아니므로 None
        return grad_h, grad_flow, None, None, None


class FixedRandomEncoder(nn.Module):
    """
    고정 랜덤 인코더 (V1 스타일).
    28×28 이미지를 2048차원 히든 노드로 사상.
    """

    def __init__(self, in_dim: int = 784, out_dim: int = 2048):
        super().__init__()
        weight = torch.empty(out_dim, in_dim)
        nn.init.orthogonal_(weight)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = x.view(bsz, -1)
        x = x @ self.weight.t()
        x = torch.relu(x)
        return x


class GraphDiffusionRiemann(nn.Module):
    """
    그래프 기반 리만 라그랑지안 디퓨전 모델.

    - 노드:
        hidden_dim 개의 히든 노드 + num_classes 개의 클래스 노드
    - 파라미터:
        단 하나의 그래프 가중치 행렬 W ∈ R^{(H+C)×(H+C)}
    - 동역학:
        state_{t+1} = RiemannianDiffusionStep( state_t, tanh(state_t @ W), engine )
    - 분류:
        마지막 C개의 클래스 노드 활성값을 logits로 사용.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_classes: int = 10,
        steps: int = 5,
        alpha: float = 0.9,
        dt: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.total_dim = hidden_dim + num_classes
        self.steps = steps
        self.alpha = alpha
        self.dt = dt

        # 고정 인코더
        self.encoder = FixedRandomEncoder(784, hidden_dim)

        # 그래프 가중치 행렬 W (인접 행렬 해석)
        w_init = torch.eye(self.total_dim) * 0.8 + torch.randn(
            self.total_dim, self.total_dim
        ) * 0.01
        self.W = nn.Parameter(w_init)

        # Reality Stone 리만 디퓨전 엔진 (Rust+CUDA)
        print(
            f"Initializing Reality Stone Riemannian Engine "
            f"(Dim={self.total_dim}, Alpha={alpha}, dt={dt})"
        )
        self.rs_engine = rs.PyRiemannianDiffusion(self.total_dim, alpha, dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)

        # 1) 입력을 히든 노드 상태로 인코딩 (인코더는 고정)
        with torch.no_grad():
            h_enc = self.encoder(x)  # [B, H]

        # 2) 전체 그래프 상태 벡터 구성: [hidden | class]
        state = torch.zeros(
            bsz, self.total_dim, device=x.device, dtype=h_enc.dtype
        )
        state[:, : self.hidden_dim] = h_enc

        # 3) 그래프 위 리만 디퓨전
        for _ in range(self.steps):
            flow = torch.tanh(state @ self.W)  # [B, total_dim]
            state = RiemannianDiffusionStep.apply(
                state, flow, self.rs_engine, self.alpha, self.dt
            )

        # 4) 마지막 C개 노드가 클래스 노드 → logits
        logits = state[:, self.hidden_dim :]
        return logits


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def run_graph_riemann_mnist():
    print("\n=== MNIST Graph Riemannian Diffusion (Reality Stone) ===")
    print("Nodes: 2048 hidden + 10 class")
    print("Dynamics: state_{t+1} = RS_RiemannianDiffusion(state_t, tanh(state_t @ W))\n")

    # 데이터셋
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 모델 / 손실 / 옵티마이저
    model = GraphDiffusionRiemann(
        hidden_dim=2048,
        num_classes=10,
        steps=5,
        alpha=0.9,
        dt=0.1,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            epoch_loss += loss.item() * bsz
            total += bsz

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(1, total)

        acc = evaluate(model, test_loader)
        best_acc = max(best_acc, acc)

        print(
            f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {elapsed:.2f}s"
        )

    print("\n=== Final Results (Graph Riemannian Diffusion) ===")
    print(f"Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    run_graph_riemann_mnist()


