"""
Banking77 의도 분류를 위한 리만 하이퍼볼릭 확산 실험 (Lorentz + Reality Stone).

구성 개요
---------
- 인코더: BERT CLS 임베딩 (768차원)
- 하이퍼 확장: 768 → 4096 차원으로 투영 (유클리드 / 접공간 표현)
- 리만 디퓨전: Reality Stone의 Rust+CUDA 엔진으로 T-step 확산
- 리드아웃: Lorentz 모델 위로 올린 후, Reality Stone `lorentz_distance` 로 프로토타입까지의 거리 계산

철학
-----
- "Riemannian Geometry is the Engine, Diffusion is the Fuel."
  (기하학은 엔진, 확산은 연료)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import math
import reality_stone as rs
from reality_stone.layers.lorentz import lorentz_distance


class RiemannianDiffusionStep(torch.autograd.Function):
    """
    Reality Stone 리만 디퓨전 엔진(PyRiemannianDiffusion)을 감싸는 Autograd Function.

    Forward (CUDA 커널 수준 수식)
    ------------------------------
    각 성분 i 에 대해,
        v_i      = (1 - alpha) * (flow_i - h_i)
        h_next_i = h_i + v_i * dt
    이후 커널 내부에서 컴포넌트별 클리핑으로 Poincaré Ball 제약을 근사(retraction).

    Backward (정확한 선형 야코비안)
    --------------------------------
    위 식을 정리하면,
        h_next = a * h + b * flow
        a = 1 - (1 - alpha) * dt
        b = (1 - alpha) * dt
    이므로,
        ∂h_next/∂h    = a
        ∂h_next/∂flow = b
    를 그대로 사용한다. (클리핑 항은 근사적으로 무시)
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


def to_lorentz(x, c: float = 1.0) -> torch.Tensor:
    """
    유클리드 벡터 x 를 Lorentz 하이퍼볼로이드 모델 좌표로 올린다.

    수식
    ----
    - 입력: x ∈ R^d
    - 출력: x_L ∈ R^{d+1}
        x0 = sqrt(1/c + ||x||^2)
        xL = [x0, x]
    """
    sq = (x * x).sum(dim=-1, keepdim=True)
    time_comp = torch.sqrt(1.0 / c + sq)
    return torch.cat([time_comp, x], dim=-1)

class BERTEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
            
    def forward(self, texts, device):
        """
        텍스트 리스트 → BERT CLS 임베딩 (배치, 768차원).
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors="pt"
        ).to(device)
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :] 
        return cls_emb

class RiemannianHyperExpansion(nn.Module):
    def __init__(self, input_dim=768, num_classes=77, hyper_dim=4096, steps=3, c=0.1):
        super().__init__()
        # 하이퍼파라미터
        self.steps = steps
        self.hyper_dim = hyper_dim
        self.c = c  # 곡률 (쌍곡 공간 negative curvature)
        self.alpha = 0.9
        self.dt = 0.1
        
        print(f"Initializing Riemannian Hyper-Expansion: {input_dim} -> {hyper_dim} dim (Curvature {c})")
        
        # 1. 고차원 확장 (Euclidean → High-dim "tangent-like" space)
        #    BERT CLS (768) → 4096 차원 표현
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hyper_dim),
            nn.LayerNorm(hyper_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 2. Diffusion Block
        #    - h 로부터 "흐름(flow)" 벡터를 생성하는 네트워크
        #    - 이 flow 가 리만 디퓨전 커널에 들어가는 목표 방향 / 속도 역할
        self.diffusion_block = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim),
            nn.GELU(),
            nn.Linear(hyper_dim, hyper_dim),
            nn.LayerNorm(hyper_dim),
        )
        
        # 3. Riemannian Readout
        #    - 4096차원 상의 클래스 프로토타입 (쌍곡 공간상의 클래스 중심 역할)
        self.prototypes = nn.Parameter(torch.randn(num_classes, hyper_dim) * 0.01)

        # 4. Reality Stone 리만 디퓨전 엔진 (Rust+CUDA)
        #    - PyRiemannianDiffusion(dim, alpha, dt)
        #    - 내부에서 CUDA 커널을 호출해 지오데식 근사 업데이트 수행
        print(
            f"Initializing Reality Stone Riemannian Engine "
            f"(Dim={hyper_dim}, Alpha={self.alpha}, dt={self.dt})"
        )
        self.rs_engine = rs.PyRiemannianDiffusion(hyper_dim, self.alpha, self.dt)

    def forward(self, x):
        # 1. BERT 임베딩을 고차원 공간으로 확장
        h = self.projector(x)
        
        # 2. Reality Stone 기반 리만 디퓨전 (T 스텝)
        for t in range(self.steps):
            # flow: 현재 상태 h 에서 생성된 "속도/흐름" 벡터
            flow = self.diffusion_block(h)
            # 리만 디퓨전 스텝 (Rust+CUDA 백엔드에서 동역학 수행)
            h = RiemannianDiffusionStep.apply(
                h,
                flow,
                self.rs_engine,
                self.alpha,
                self.dt,
            )
            
        # 3. Lorentz 모델로 리프트 후, 리만 거리 계산
        #    h: (B, 4096)  → h_L: (B, 4097)
        #    P: (77, 4096) → P_L: (77, 4097)
        h_L = to_lorentz(h, self.c)
        P_L = to_lorentz(self.prototypes, self.c)
        
        # 배치-프로토타입 쌍별 Lorentz 거리 계산을 위한 브로드캐스트 준비
        #   h_L: (B,   1, 4097)
        #   P_L: (1,  77, 4097)
        B = h.size(0)
        h_exp = h_L.unsqueeze(1).expand(B, self.prototypes.size(0), -1)
        P_exp = P_L.unsqueeze(0).expand(B, -1, -1)
        
        # Reality Stone의 Lorentz 거리 함수 (CUDA + Autograd 지원)
        # 커널은 (B*77, 4097) 형태의 2D 텐서를 기대하므로 평탄화 후 다시 reshape
        dist = lorentz_distance(
            h_exp.reshape(-1, self.hyper_dim + 1),
            P_exp.reshape(-1, self.hyper_dim + 1),
            self.c
        ).reshape(B, -1)
        
        # 음의 거리 = 로짓 (가까울수록 점수가 높아지도록)
        return -dist

def run_banking77_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Banking77 Riemannian Hyper-Expansion (Reality Stone) ===")
    print(f"Device: {DEVICE}")
    
    dataset = load_dataset("banking77")
    train_data = dataset['train']
    test_data = dataset['test']
    num_classes = 77
    
    encoder = BERTEncoder().to(DEVICE)
    model = RiemannianHyperExpansion(
        input_dim=768, 
        num_classes=num_classes, 
        hyper_dim=4096, 
        steps=2,
        c=0.1 # Optimal curvature found in previous experiments
    ).to(DEVICE)
    
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
        {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
    ])
    
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    epochs = 20
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    cycle_epochs = min(10, epochs)
    scheduler_one = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        steps_per_epoch=steps_per_epoch,
        epochs=cycle_epochs
    )
    scheduler_two = None
    best_acc = 0.0
    
    print(f"Start Training...")
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        model.train()
        total_loss = 0
        
        indices = np.random.permutation(len(train_data))
        
        pbar = tqdm(range(0, len(train_data), batch_size), desc=f"Ep {epoch}", leave=False)
        for i in pbar:
            batch_idx = indices[i:i+batch_size]
            batch_texts = [train_data[int(j)]['text'] for j in batch_idx]
            batch_labels = torch.tensor([train_data[int(j)]['label'] for j in batch_idx]).to(DEVICE)
            
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
                        eta_min=1e-6
                    )
                scheduler_two.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / (len(train_data) / batch_size)
        
        encoder.eval()
        model.eval()
        correct = 0
        total = 0
        eval_indices = range(0, len(test_data), 64)
        
        with torch.no_grad():
            for i in eval_indices:
                batch_idx = range(i, min(i+64, len(test_data)))
                batch_texts = [test_data[j]['text'] for j in batch_idx]
                batch_labels = torch.tensor([test_data[j]['label'] for j in batch_idx]).to(DEVICE)
                
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
    run_banking77_experiment()
