import math
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

import reality_stone as rs
from reality_stone import poincare_distance


# Reality Stone Philosophy:
# "Riemannian Geometry is the Engine, Diffusion is the Fuel."


class RiemannianDiffusionStep(torch.autograd.Function):
    """
    Reality Stone의 리만 디퓨전 엔진을 감싸는 PyTorch Autograd Function.

    Forward:
        CUDA 커널에서 수행하는 연산 (각 성분 i):
            v_i      = (1 - alpha) * (flow_i - h_i)
            h_next_i = h_i + v_i * dt
        (이후 컴포넌트-wise 클리핑으로 Poincaré Ball 제약을 근사).

    Backward:
        위 식의 선형 부분에 대한 정확한 야코비안 (클리핑은 무시한 근사):
            h_next = a * h + b * flow
            a = 1 - (1 - alpha) * dt
            b = (1 - alpha) * dt
        ⇒ ∂h_next/∂h    = a
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

        # CUDA 사용 가능 시: Rust+CUDA 커널 직접 호출 (Zero-Copy)
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


def project_to_poincare(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Poincaré Ball 투영.
    norm(x) < 1/sqrt(c) 를 보장하도록 반지름을 클리핑.
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    max_norm = 1.0 / math.sqrt(c) - eps
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


class BERTEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb


class PoincareHyperExpansion(nn.Module):
    """
    Banking77용 Poincaré 기반 Riemannian Hyper-Expansion.

    - 1단계: BERT CLS → 4096차원 고차원 투영 (Euclidean / tangent)
    - 2단계: Reality Stone Riemannian Diffusion (Rust+CUDA)
    - 3단계: Poincaré Ball로 투영 후, Reality Stone poincare_distance로 프로토타입과 거리 계산
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 77,
        hyper_dim: int = 4096,
        steps: int = 3,
        c: float = 0.1,
    ):
        super().__init__()
        self.steps = steps
        self.hyper_dim = hyper_dim
        self.c = c  # 곡률
        self.alpha = 0.9
        self.dt = 0.1

        print(
            f"Initializing Poincaré Hyper-Expansion: {input_dim} -> {hyper_dim} dim (Curvature {c})"
        )

        # 1. Expansion (Euclidean → High-dim)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hyper_dim),
            nn.LayerNorm(hyper_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # 2. Diffusion Block: flow(h) 생성 네트워크
        self.diffusion_block = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim),
            nn.GELU(),
            nn.Linear(hyper_dim, hyper_dim),
            nn.LayerNorm(hyper_dim),
        )

        # 3. Poincaré 프로토타입 (Ball 좌표계 상에서의 클래스 센터)
        self.prototypes = nn.Parameter(torch.randn(num_classes, hyper_dim) * 0.01)

        # 4. Reality Stone Riemannian Diffusion Engine (Rust+CUDA)
        print(
            f"Initializing Reality Stone Riemannian Engine "
            f"(Dim={hyper_dim}, Alpha={self.alpha}, dt={self.dt})"
        )
        self.rs_engine = rs.PyRiemannianDiffusion(hyper_dim, self.alpha, self.dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Expansion to Hyperspace (Euclidean)
        h = self.projector(x)  # [B, D]

        # 2. Riemannian Diffusion (Reality Stone Rust+CUDA)
        for _ in range(self.steps):
            flow = self.diffusion_block(h)
            h = RiemannianDiffusionStep.apply(
                h,
                flow,
                self.rs_engine,
                self.alpha,
                self.dt,
            )

        # 3. Poincaré Ball로 투영 (상태 및 프로토타입)
        h_p = project_to_poincare(h, c=self.c)
        proto_p = project_to_poincare(self.prototypes, c=self.c)

        # 4. Reality Stone의 Poincaré 거리 사용
        B = h_p.size(0)
        num_classes = proto_p.size(0)

        h_exp = h_p.unsqueeze(1).expand(B, num_classes, -1).contiguous()
        p_exp = proto_p.unsqueeze(0).expand(B, num_classes, -1).contiguous()

        dist = poincare_distance(
            h_exp.reshape(-1, self.hyper_dim),
            p_exp.reshape(-1, self.hyper_dim),
            c=self.c,
        ).reshape(B, num_classes)

        # Negative distance = logits
        return -dist


def run_banking77_poincare_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Banking77 Poincaré Riemannian Diffusion (Reality Stone) ===")
    print(f"Device: {DEVICE}")

    dataset = load_dataset("banking77")
    train_data = dataset["train"]
    test_data = dataset["test"]
    num_classes = 77

    encoder = BERTEncoder().to(DEVICE)
    model = PoincareHyperExpansion(
        input_dim=768,
        num_classes=num_classes,
        hyper_dim=4096,
        steps=2,
        c=0.1,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": 2e-5, "weight_decay": 0.01},
            {"params": model.parameters(), "lr": 1e-4, "weight_decay": 0.01},
        ]
    )

    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    epochs = 20
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    cycle_epochs = min(10, epochs)
    scheduler_one = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        steps_per_epoch=steps_per_epoch,
        epochs=cycle_epochs,
    )
    scheduler_two = None
    best_acc = 0.0

    print("Start Training...")

    for epoch in range(1, epochs + 1):
        encoder.train()
        model.train()
        total_loss = 0.0

        indices = np.random.permutation(len(train_data))

        pbar = tqdm(
            range(0, len(train_data), batch_size),
            desc=f"Ep {epoch}",
            leave=False,
        )
        for i in pbar:
            batch_idx = indices[i : i + batch_size]
            batch_texts = [train_data[int(j)]["text"] for j in batch_idx]
            batch_labels = torch.tensor(
                [train_data[int(j)]["label"] for j in batch_idx]
            ).to(DEVICE)

            optimizer.zero_grad()
            embeddings = encoder(batch_texts, DEVICE)
            logits = model(embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if epoch <= cycle_epochs:
                scheduler_one.step()
            else:
                if scheduler_two is None:
                    scheduler_two = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=(epochs - cycle_epochs) * steps_per_epoch,
                        eta_min=1e-6,
                    )
                scheduler_two.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / (len(train_data) / batch_size)

        encoder.eval()
        model.eval()
        correct = 0
        total = 0
        eval_indices = range(0, len(test_data), 64)

        with torch.no_grad():
            for i in eval_indices:
                batch_idx = range(i, min(i + 64, len(test_data)))
                batch_texts = [test_data[j]["text"] for j in batch_idx]
                batch_labels = torch.tensor(
                    [test_data[j]["label"] for j in batch_idx]
                ).to(DEVICE)

                embeddings = encoder(batch_texts, DEVICE)
                logits = model(embeddings)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc

        print(f"Ep {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")


if __name__ == "__main__":
    run_banking77_poincare_experiment()


