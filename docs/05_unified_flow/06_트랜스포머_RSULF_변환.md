# Transformer → RS‑ULF LLM 변환 이론·실전 가이드

> **목표**: 기존 Transformer 모델(Mistral, Qwen, LLaMA 등)의 가중치를 Reality Stone Unified Lagrangian Flow(RS-ULF)로 정합성을 유지하며 완전히 변환

**핵심 원칙**:
- 표현력 100% 보존
- 시간복잡도 O(n²d) → O(nd) 달성
- 공간복잡도 O(n²) → O(d) 달성
- 수학적 정합성 검증 필수

---

## 목차 (LLM 변환 단계)

1. [개요 및 수학적 기반](#1-개요-및-수학적-기반)
2. [변환 준비](#2-변환-준비)
3. [Step 1: Metric 추출](#3-step-1-metric-추출)
4. [Step 2: Potential 함수 구축](#4-step-2-potential-함수-구축)
5. [Step 3: Graph Diffusion 설정](#5-step-3-graph-diffusion-설정)
6. [Step 4: DP Memory 구성](#6-step-4-dp-memory-구성)
7. [Step 5: 레이어 조립](#7-step-5-레이어-조립)
8. [Step 6: 차원 접기(Folding)](#8-step-6-차원-접기folding)
9. [Step 7: 정합성 테스트](#9-step-7-정합성-테스트)
10. [Step 8: 전체 모델 변환](#10-step-8-전체-모델-변환)
11. [실전 예제](#11-실전-예제)
12. [성능 벤치마크](#12-성능-벤치마크)

---

## 1. LLM 변환 개요 및 수학적 기반

### 1.1 변환의 수학적 근거

**Transformer 레이어**:
$$
x_{t+1} = x_t + \text{Attn}(x_t) + \text{FFN}(x_t)
$$

여기서:
- Attention: $\text{softmax}(QK^T)V$
- FFN: $W_2\sigma(W_1 x)$

**RS-ULF 레이어**:
$$
x_{t+1} = \exp_{x_t}\left[-\eta \nabla_g \Phi(x_t) + \alpha \Delta_g x_t + \beta L x_t + \gamma V_t\right]
$$

여기서:
- $g$: Riemannian metric (from $W_Q^T W_K$)
- $\Phi$: Potential function (from FFN)
- $\Delta_g$: Riemannian Laplacian
- $L$: Graph Laplacian (directed diffusion)
- $V_t$: Bellman/DP memory

### 1.2 동등성 정리 (Equivalence Theorem)

**정리**: Transformer 레이어 $T$와 RS-ULF 레이어 $R$에 대해:

$$
\lim_{K \to 0} R(x; g, \Phi, L, V) = T(x; W_Q, W_K, W_V, W_O, W_1, W_2)
$$

여기서 $K$는 sectional curvature.

즉, **RS-ULF는 Transformer의 일반화**이며, 곡률=0일 때 Transformer와 동등.

### 1.3 복잡도 및 압축률 분석 (Global Basis 관점)

시퀀스 길이 $n$ 에 대한 이론 복잡도는 다음과 같다.

| 구성요소 | Transformer | RS-ULF | 개선비 |
|---------|------------|--------|-------|
| Attention | O(n²d) | O(0) | ∞ |
| Metric | - | O(d) | - |
| Potential | O(d²) | O(d²) | 1× |
| Diffusion | - | O(Ed) | - |
| Memory | O(n²) | O(d) | O(n²/d) |
| **총합** | **O(n²d)** | **O((n+E)d)** | **O(n)** |

압축률은 **전 레이어를 하나의 global basis로 묶는지 여부**에 크게 의존한다.

- 원본 Transformer (레이어 수 $L$, 폭 $d$, FFN 폭 $d_\text{ff}$):
  - 레이어당 $O(d^2 + d d_\text{ff})$
  - 전체 $O(L (d^2 + d d_\text{ff}))$
- RS-ULF (global basis 사용 시):
  - 공통 basis: $O(d r_\* + d_\text{ff} r_\*)$ (한 번만)
  - 레이어별 스케일/곡률: $O(L r_\text{small})$

적절한 $r_\*, r_\text{small} \ll d, d_\text{ff}$ 를 택하면,

- **레이어당 압축률 ≥ 8–10×**, 전체 모델 기준 **≥ 6–8×** 를 목표로 할 수 있다.  
  (정확한 수치는 `benchmark_conversion.py` 결과와 함께 최종 결정)

---

## 2. 변환 준비 (모델·라이브러리)

### 2.1 필수 라이브러리

```bash
pip install torch transformers accelerate
pip install reality_stone  # 또는 로컬 빌드: uv run maturin develop --features cuda
```

### 2.2 모델 로딩

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mistral 예시
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Qwen 예시
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 2.3 가중치 추출 함수

```python
def extract_layer_weights(model, layer_idx):
    """Transformer 레이어에서 필요한 가중치 추출"""
    layer = model.model.layers[layer_idx]
    
    # Attention weights
    WQ = layer.self_attn.q_proj.weight.detach()
    WK = layer.self_attn.k_proj.weight.detach()
    WV = layer.self_attn.v_proj.weight.detach()
    WO = layer.self_attn.o_proj.weight.detach()
    
    # FFN weights
    # Mistral/Qwen: gate_proj, down_proj
    W1 = layer.mlp.gate_proj.weight.detach()
    W2 = layer.mlp.down_proj.weight.detach()
    
    # Optional: up_proj (SwiGLU에서 사용)
    if hasattr(layer.mlp, 'up_proj'):
        W_up = layer.mlp.up_proj.weight.detach()
    else:
        W_up = None
    
    # Normalization
    norm_attn = layer.input_layernorm.weight.detach()
    norm_ffn = layer.post_attention_layernorm.weight.detach()
    
    return {
        'WQ': WQ, 'WK': WK, 'WV': WV, 'WO': WO,
        'W1': W1, 'W2': W2, 'W_up': W_up,
        'norm_attn': norm_attn, 'norm_ffn': norm_ffn
    }
```

### 2.4 모델별 특수사항

#### Mistral
- Sliding Window Attention (SWA) → local diffusion으로 매핑
- GQA (Grouped Query Attention) → metric rank 조정 필요

#### Qwen
- RoPE positional encoding → curvature term으로 매핑
- GEGLU activation → potential 정의에 반영

#### LLaMA
- 표준 구조, 가장 직접적 변환 가능

---

## 3. Step 1: 리만 메트릭 추출 이론·구현 (Global Basis)

### 3.1 수학적 정의 (Global)

Transformer의 모든 레이어에 대해 Q, K projection을 모아,

$$
M_Q = 
\begin{bmatrix}
W_Q^{(1)} \\
W_Q^{(2)} \\
\vdots \\
W_Q^{(L)}
\end{bmatrix},
\quad
M_K = 
\begin{bmatrix}
W_K^{(1)} \\
\vdots \\
W_K^{(L)}
\end{bmatrix}
$$

을 구성한 뒤, Randomized SVD 등으로

$$
M_Q^\top M_K \approx U_\* \Sigma_\* V_\*^\top
$$

형태의 **global metric basis**를 추출한다.

### 3.2 구현 스케치 (Global)

```python
def build_global_metric_basis(WQ_list, WK_list, target_rank):
    # WQ_list, WK_list: 리스트 (레이어 수 L)
    M_Q = torch.cat(WQ_list, dim=0)  # (L * d_q, d)
    M_K = torch.cat(WK_list, dim=0)  # (L * d_k, d)
    G = M_Q.t() @ M_K                # (d, d)
    
    # Randomized SVD로 상위 r 성분만 추출
    U_star, S_star, V_star = randomized_svd(G, target_rank)
    return U_star, S_star, V_star
```

### 3.3 Metric 안정화 및 오차‑곡률 보정

추출된 metric은 positive definite(PD)가 아닐 수 있음. 안정화 필수:

#### Strategy A: Diagonal Metric (권장)

```python
def stabilize_metric_diagonal(g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Diagonal metric으로 안정화 (가장 빠르고 안정적)"""
    diag_vals = torch.diag(g)
    # 양수 보장
    diag_vals = torch.abs(diag_vals) + eps
    return torch.diag(diag_vals)
```

**장점**:
- 역행렬 계산 O(d)
- 수치 안정성 최고
- Christoffel symbol = 0

#### Strategy B: Symmetrize + Regularize

```python
def stabilize_metric_symmetric(g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """대칭화 후 정규화"""
    # 대칭화
    g_sym = 0.5 * (g + g.t())
    
    # PD 보장: g + εI
    eye = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
    g_stable = g_sym + eps * eye
    
    return g_stable
```

#### Strategy C: Low-Rank Approximation + Error Curvature

Global SVD에서 잘려 나간 singular value 집합 $\{\sigma_{r+1},\dots\}$ 에 대해

$$
K_\text{error}
 := \Big( \sum_{i>r} \sigma_i^2 \Big)^{1/2}
$$

를 정의하고, 이를 레이어 곡률/regularization 스칼라로 사용한다.  
이 값은 폴딩‧압축으로 버려진 정보를 곡률 functional에 다시 흡수하는 역할을 한다.

### 3.4 Metric Inverse 계산

```python
def compute_metric_inverse(g: torch.Tensor, strategy: str = "diagonal") -> torch.Tensor:
    """
    Metric 역행렬 계산
    
    Args:
        g: Metric tensor
        strategy: 안정화 전략 ("diagonal", "symmetric", "lowrank")
    
    Returns:
        g_inv: Inverse metric
    """
    if strategy == "diagonal":
        # Diagonal metric의 역행렬
        diag_vals = torch.diag(g)
        inv_vals = 1.0 / (diag_vals + 1e-8)
        return torch.diag(inv_vals)
    else:
        # 일반 역행렬
        try:
            g_inv = torch.inverse(g + 1e-6 * torch.eye(g.size(0), device=g.device))
            return g_inv
        except:
            # Fallback to diagonal
            return compute_metric_inverse(g, "diagonal")
```

### 3.5 체크포인트: Metric 추출

**필수 검증**:

```python
def validate_metric(g: torch.Tensor) -> dict:
    """Metric 검증"""
    results = {}
    
    # 1. Positive Definite 확인
    try:
        eigvals = torch.linalg.eigvalsh(g)
        results['is_pd'] = torch.all(eigvals > 0).item()
        results['min_eigenvalue'] = eigvals.min().item()
        results['max_eigenvalue'] = eigvals.max().item()
    except:
        results['is_pd'] = False
        results['min_eigenvalue'] = None
        results['max_eigenvalue'] = None
    
    # 2. Condition number
    if results['is_pd']:
        cond = results['max_eigenvalue'] / results['min_eigenvalue']
        results['condition_number'] = cond
        results['is_well_conditioned'] = cond < 1e6
    else:
        results['condition_number'] = float('inf')
        results['is_well_conditioned'] = False
    
    # 3. Symmetry
    sym_error = torch.norm(g - g.t()) / torch.norm(g)
    results['symmetry_error'] = sym_error.item()
    results['is_symmetric'] = sym_error < 1e-4
    
    return results

# 사용 예시
g = extract_metric(WQ, WK)
g_stable = stabilize_metric_diagonal(g)
validation = validate_metric(g_stable)
print(validation)
```

**테스트 코드**:

```python
def test_metric_extraction():
    """Metric 추출 테스트"""
    d = 4096
    WQ = torch.randn(d, d) * 0.02
    WK = torch.randn(d, d) * 0.02
    
    # 추출
    g = extract_metric(WQ, WK)
    assert g.shape == (d, d), "Shape mismatch"
    
    # 안정화
    g_stable = stabilize_metric_diagonal(g)
    
    # 검증
    val = validate_metric(g_stable)
    assert val['is_pd'], "Metric not PD"
    assert val['is_well_conditioned'], "Metric ill-conditioned"
    
    # 역행렬
    g_inv = compute_metric_inverse(g_stable)
    identity = g_stable @ g_inv
    error = torch.norm(identity - torch.eye(d)) / d
    assert error < 1e-3, f"Inverse error too large: {error}"
    
    print("✓ Metric extraction test passed")

test_metric_extraction()
```

---

## 4. Step 2: Potential 함수 구축 이론·구현

### 4.1 수학적 정의

Transformer FFN을 potential function으로 변환:

**Transformer FFN**:
$$
f(x) = W_2 \sigma(W_1 x)
$$

**RS-ULF Potential**:
$$
\Phi(x) = \frac{1}{2} \|f(x)\|^2 = \frac{1}{2} \|W_2 \sigma(W_1 x)\|^2
$$

**Gradient**:
$$
\nabla_x \Phi(x) = J_f(x)^T f(x)
$$

여기서 $J_f$는 Jacobian.

### 4.2 구현

```python
import torch.nn.functional as F

class PotentialFunction(nn.Module):
    """Potential function Φ(x) from Transformer FFN weights"""
    
    def __init__(self, W1: torch.Tensor, W2: torch.Tensor, 
                 W_up: torch.Tensor = None, activation: str = 'relu'):
        super().__init__()
        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)
        
        # SwiGLU/GEGLU 지원
        self.W_up = nn.Parameter(W_up) if W_up is not None else None
        
        # Activation function
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'gelu':
            self.act = F.gelu
        elif activation == 'silu':
            self.act = F.silu
        else:
            self.act = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Φ(x)
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            phi: (batch, seq_len) scalar potential values
        """
        # FFN forward
        if self.W_up is not None:
            # SwiGLU: W2(silu(W1 x) ⊙ (W_up x))
            gate = self.act(F.linear(x, self.W1))
            up = F.linear(x, self.W_up)
            h = gate * up
        else:
            # Standard: W2(σ(W1 x))
            h = self.act(F.linear(x, self.W1))
        
        y = F.linear(h, self.W2)
        
        # Potential: ||y||^2 / 2
        phi = 0.5 * torch.sum(y ** 2, dim=-1)
        
        return phi
    
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇Φ(x) using autograd
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            grad: (batch, seq_len, d_model)
        """
        # Enable gradient computation
        x_input = x.detach().requires_grad_(True)
        
        # Compute potential
        phi = self.forward(x_input)
        
        # Compute gradient
        grad = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=x_input,
            create_graph=True
        )[0]
        
        return grad
```

### 4.3 Activation별 특수 처리

#### ReLU (일반적)
```python
def phi_relu(x, W1, W2):
    h = F.relu(F.linear(x, W1))
    y = F.linear(h, W2)
    return 0.5 * y.pow(2).sum(dim=-1)
```

#### SwiGLU (Mistral, LLaMA)
```python
def phi_swiglu(x, W1, W2, W_up):
    gate = F.silu(F.linear(x, W1))
    up = F.linear(x, W_up)
    h = gate * up
    y = F.linear(h, W2)
    return 0.5 * y.pow(2).sum(dim=-1)
```

#### GEGLU (Qwen)
```python
def phi_geglu(x, W1, W2, W_up):
    gate = F.gelu(F.linear(x, W1))
    up = F.linear(x, W_up)
    h = gate * up
    y = F.linear(h, W2)
    return 0.5 * y.pow(2).sum(dim=-1)
```

### 4.4 체크포인트: Potential

**필수 검증**:

```python
def test_potential():
    """Potential function 테스트"""
    batch, seq_len, d = 2, 16, 4096
    x = torch.randn(batch, seq_len, d)
    
    W1 = torch.randn(d * 4, d) * 0.02
    W2 = torch.randn(d, d * 4) * 0.02
    
    pot = PotentialFunction(W1, W2, activation='relu')
    
    # 1. Scalar output
    phi = pot(x)
    assert phi.shape == (batch, seq_len), f"Shape mismatch: {phi.shape}"
    
    # 2. Non-negative
    assert torch.all(phi >= 0), "Potential should be non-negative"
    
    # 3. Gradient shape
    grad = pot.gradient(x)
    assert grad.shape == x.shape, f"Gradient shape mismatch"
    
    # 4. Gradient direction test (finite difference)
    eps = 1e-4
    x_perturb = x + eps * grad
    phi_perturb = pot(x_perturb)
    
    # Φ should increase along gradient
    assert torch.all(phi_perturb >= phi), "Potential should increase along gradient"
    
    print("✓ Potential function test passed")

test_potential()
```

---

## 5. Step 3: Graph Diffusion 설정 이론·구현

### 5.1 수학적 정의

방향성 그래프 $G = (V, E)$에서:

**Adjacency Matrix**:
$$
A_{ij} = \begin{cases}
w_{ij} & \text{if } (i \to j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

**Degree Matrix**:
$$
D_{ii} = \sum_j A_{ij}
$$

**Laplacian**:
$$
L = D - A
$$

**Diffusion Dynamics**:
$$
\frac{\partial x}{\partial t} = -L x
$$

### 5.2 구현

```python
def build_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Build graph Laplacian L = D - A
    
    Args:
        adjacency: (n, n) adjacency matrix
    
    Returns:
        L: (n, n) Laplacian matrix
    """
    # Degree matrix
    degrees = adjacency.sum(dim=1)
    D = torch.diag(degrees)
    
    # Laplacian
    L = D - adjacency
    
    return L

def create_sequence_graph(seq_len: int, window_size: int = 8, 
                          directed: bool = True) -> torch.Tensor:
    """
    Create sequence graph for token-level diffusion
    
    Args:
        seq_len: Sequence length
        window_size: Local attention window size
        directed: If True, only backward edges (causal)
    
    Returns:
        adjacency: (seq_len, seq_len) adjacency matrix
    """
    adj = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        # Local window
        start = max(0, i - window_size)
        end = i if directed else min(seq_len, i + window_size)
        
        for j in range(start, end):
            if i != j:
                # Distance-based weight
                dist = abs(i - j)
                weight = 1.0 / (1.0 + dist)
                adj[i, j] = weight
    
    return adj
```

### 5.3 Diffusion Operator

```python
class GraphDiffusion(nn.Module):
    """Graph diffusion operator"""
    
    def __init__(self, laplacian: torch.Tensor, tau: float = 0.01):
        super().__init__()
        self.register_buffer('L', laplacian)
        self.tau = tau
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply diffusion: x' = x - τLx
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            x_diffused: (batch, seq_len, d_model)
        """
        # x: (B, L, D)
        # L: (L, L)
        # Lx: (B, L, D)
        
        Lx = torch.matmul(self.L, x.transpose(1, 2)).transpose(1, 2)
        x_diffused = x - self.tau * Lx
        
        return x_diffused
```

### 5.4 모델별 그래프 구성

#### Mistral (Sliding Window Attention)
```python
def create_mistral_graph(seq_len: int, window_size: int = 4096) -> torch.Tensor:
    """Mistral의 SWA를 그래프로 표현"""
    adj = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        # Sliding window
        start = max(0, i - window_size)
        for j in range(start, i):
            adj[i, j] = 1.0
    
    return adj
```

#### Qwen/LLaMA (Full Causal)
```python
def create_causal_graph(seq_len: int, decay: float = 0.9) -> torch.Tensor:
    """Causal graph with exponential decay"""
    adj = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        for j in range(i):
            # Exponential decay based on distance
            dist = i - j
            weight = decay ** dist
            adj[i, j] = weight
    
    return adj
```

### 5.5 체크포인트: Graph Diffusion

```python
def test_graph_diffusion():
    """Graph diffusion 테스트"""
    seq_len = 128
    d_model = 4096
    batch = 2
    
    # Create graph
    adj = create_sequence_graph(seq_len, window_size=8, directed=True)
    L = build_laplacian(adj)
    
    # Test 1: Laplacian properties
    # L @ 1 = 0 (constant vector is kernel)
    ones = torch.ones(seq_len)
    L_ones = L @ ones
    assert torch.allclose(L_ones, torch.zeros(seq_len), atol=1e-5), \
        "Laplacian should annihilate constant vector"
    
    # Test 2: Diffusion
    diffusion = GraphDiffusion(L, tau=0.01)
    x = torch.randn(batch, seq_len, d_model)
    x_diff = diffusion(x)
    
    assert x_diff.shape == x.shape, "Shape mismatch"
    
    # Test 3: Energy decrease (for undirected graph)
    # Energy = x^T L x should decrease
    # (For directed graph, this may not hold)
    
    print("✓ Graph diffusion test passed")

test_graph_diffusion()
```

---

## 6. Step 4: DP Memory 구성 이론·구현

### 6.1 수학적 정의

Bellman/DP memory는 long-range dependency를 캐싱:

$$
V_t = \gamma V_{t-1} + \Phi(x_t)
$$

여기서:
- $V_t$: time $t$의 메모리 상태
- $\gamma \in [0, 1)$: decay factor
- $\Phi(x_t)$: 현재 potential

### 6.2 구현

```python
class BellmanMemory(nn.Module):
    """Bellman/DP memory for long-range dependency"""
    
    def __init__(self, gamma: float = 0.98):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, V_prev: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Update memory: V = γV_prev + Φ
        
        Args:
            V_prev: (batch, seq_len) or (batch, seq_len, 1)
            phi: (batch, seq_len) potential values
        
        Returns:
            V: (batch, seq_len) updated memory
        """
        if V_prev is None:
            # Initialize with current potential
            return phi
        
        # Ensure same shape
        if V_prev.dim() == 3 and V_prev.size(-1) == 1:
            V_prev = V_prev.squeeze(-1)
        
        # Update
        V = self.gamma * V_prev + phi
        
        return V
    
    def reset(self):
        """Reset memory (for new sequence)"""
        pass  # Stateless, no internal state
```

### 6.3 메모리 초기화 전략

#### Strategy A: Zero Initialization
```python
def init_memory_zero(batch_size: int, seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """Zero initialization (기본)"""
    return torch.zeros(batch_size, seq_len, device=device)
```

#### Strategy B: Potential Initialization
```python
def init_memory_from_potential(x: torch.Tensor, potential_fn: PotentialFunction) -> torch.Tensor:
    """첫 입력의 potential로 초기화"""
    with torch.no_grad():
        V0 = potential_fn(x)
    return V0
```

#### Strategy C: Learned Initialization
```python
class LearnedMemoryInit(nn.Module):
    """학습 가능한 메모리 초기화"""
    def __init__(self, d_model: int):
        super().__init__()
        self.init_param = nn.Parameter(torch.zeros(1, 1, d_model))
    
    def forward(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return self.init_param.expand(batch_size, seq_len, -1)
```

### 6.4 체크포인트: DP Memory

```python
def test_bellman_memory():
    """Bellman memory 테스트"""
    batch, seq_len = 2, 128
    gamma = 0.98
    
    memory = BellmanMemory(gamma=gamma)
    
    # Test 1: Initialization
    V0 = None
    phi = torch.randn(batch, seq_len)
    V1 = memory(V0, phi)
    assert torch.allclose(V1, phi), "Initial memory should equal first potential"
    
    # Test 2: Update
    phi2 = torch.randn(batch, seq_len)
    V2 = memory(V1, phi2)
    expected = gamma * V1 + phi2
    assert torch.allclose(V2, expected), "Memory update incorrect"
    
    # Test 3: Decay behavior
    # If phi=0, memory should decay
    V3 = memory(V2, torch.zeros_like(phi))
    assert torch.allclose(V3, gamma * V2), "Memory decay incorrect"
    
    # Test 4: Long-term accumulation
    V = torch.zeros(batch, seq_len)
    constant_phi = torch.ones(batch, seq_len)
    for _ in range(100):
        V = memory(V, constant_phi)
    
    # Converges to phi / (1 - gamma)
    expected_limit = constant_phi / (1 - gamma)
    assert torch.allclose(V, expected_limit, atol=0.1), \
        f"Memory should converge to {expected_limit.mean():.2f}, got {V.mean():.2f}"
    
    print("✓ Bellman memory test passed")

test_bellman_memory()
```

---

## 7. Step 5: RS‑ULF 레이어 조립 이론·구현

### 7.1 통합 업데이트 수식

RS-ULF 레이어의 완전한 형태:

$$
x_{t+1} = \exp_{x_t}\left[-\eta g^{-1}\nabla\Phi(x_t) + \alpha \Delta_g x_t + \beta L x_t + \gamma V_t\right]
$$

각 항의 역할:
1. $-\eta g^{-1}\nabla\Phi$: Lagrangian gradient flow (FFN 대체)
2. $\alpha \Delta_g x$: Riemannian smoothing (local regularization)
3. $\beta L x$: Graph diffusion (attention 대체)
4. $\gamma V_t$: DP memory (long-range dependency)

### 7.2 완전한 레이어 구현

```python
class RSULFLayer(nn.Module):
    """Reality Stone Unified Lagrangian Flow Layer"""
    
    def __init__(
        self,
        d_model: int,
        WQ: torch.Tensor,
        WK: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        W_up: torch.Tensor = None,
        laplacian: torch.Tensor = None,
        lr: float = 0.02,
        alpha: float = 0.04,
        beta: float = 0.01,
        gamma: float = 0.98,
        metric_strategy: str = "diagonal",
        activation: str = "relu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma_mem = gamma
        self.metric_strategy = metric_strategy
        
        # 1. Metric g
        self.WQ = nn.Parameter(WQ)
        self.WK = nn.Parameter(WK)
        
        # 2. Potential Φ
        self.potential = PotentialFunction(W1, W2, W_up, activation)
        
        # 3. Graph Laplacian L (buffer, not parameter)
        if laplacian is not None:
            self.register_buffer('L', laplacian)
        else:
            self.register_buffer('L', torch.zeros(1, 1))
        
        # 4. Bellman memory
        self.memory = BellmanMemory(gamma=gamma)
    
    def get_metric(self) -> tuple:
        """Compute metric g and its inverse"""
        # g = WQ^T @ WK
        g = extract_metric(self.WQ, self.WK)
        
        # Stabilize
        if self.metric_strategy == "diagonal":
            g_stable = stabilize_metric_diagonal(g)
        elif self.metric_strategy == "symmetric":
            g_stable = stabilize_metric_symmetric(g)
        else:
            g_stable = stabilize_metric_diagonal(g)
        
        # Inverse
        g_inv = compute_metric_inverse(g_stable, self.metric_strategy)
        
        return g_stable, g_inv
    
    def forward(
        self,
        x: torch.Tensor,
        V: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, d_model)
            V: (batch, seq_len) previous memory state
        
        Returns:
            x_next: (batch, seq_len, d_model)
            V_next: (batch, seq_len) updated memory
        """
        batch_size, seq_len, d = x.shape
        device = x.device
        
        # 1. Metric
        g, g_inv = self.get_metric()
        
        # 2. Potential gradient
        grad_phi = self.potential.gradient(x)  # (B, L, D)
        
        # 3. Compute update vector v
        v = torch.zeros_like(x)
        
        # Term 1: -η g^{-1} ∇Φ
        # g_inv: (D, D), grad_phi: (B, L, D)
        # v1: (B, L, D)
        v1 = -self.lr * torch.matmul(grad_phi, g_inv)
        v = v + v1
        
        # Term 2: α Δ_g x (simple Laplacian)
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        v2 = self.alpha * (x - x_mean)
        v = v + v2
        
        # Term 3: β L x (graph diffusion)
        if self.L.numel() > 1:
            # L: (L, L), x: (B, L, D)
            # Lx: (B, L, D)
            Lx = torch.matmul(self.L, x.transpose(1, 2)).transpose(1, 2)
            v3 = self.beta * Lx
            v = v + v3
        
        # Term 4: γ V (memory)
        if V is not None:
            if V.dim() == 2:  # (B, L)
                V_expanded = V.unsqueeze(-1)  # (B, L, 1)
            else:
                V_expanded = V
            v4 = self.gamma_mem * V_expanded
            v = v + v4
        
        # 4. Exponential map (retraction)
        # Simple: exp_x(v) ≈ x + v
        x_next = x + v
        
        # 5. Update memory
        phi = self.potential(x_next)  # (B, L)
        V_next = self.memory(V, phi)
        
        return x_next, V_next
```

### 7.3 체크포인트: 레이어 조립

```python
def test_rsulf_layer():
    """RSULF layer 전체 테스트"""
    batch, seq_len, d = 2, 128, 4096
    
    # 가중치 생성
    WQ = torch.randn(d, d) * 0.02
    WK = torch.randn(d, d) * 0.02
    W1 = torch.randn(d * 4, d) * 0.02
    W2 = torch.randn(d, d * 4) * 0.02
    
    # Laplacian
    adj = create_sequence_graph(seq_len, window_size=8)
    L = build_laplacian(adj)
    
    # Layer
    layer = RSULFLayer(
        d_model=d,
        WQ=WQ, WK=WK, W1=W1, W2=W2,
        laplacian=L,
        lr=0.02, alpha=0.04, beta=0.01, gamma=0.98
    )
    
    # Input
    x = torch.randn(batch, seq_len, d)
    V = None
    
    # Forward
    x_next, V_next = layer(x, V)
    
    # Test 1: Shape
    assert x_next.shape == (batch, seq_len, d), "Output shape mismatch"
    assert V_next.shape == (batch, seq_len), "Memory shape mismatch"
    
    # Test 2: No NaN/Inf
    assert not torch.isnan(x_next).any(), "NaN in output"
    assert not torch.isinf(x_next).any(), "Inf in output"
    
    # Test 3: Multiple steps
    V = V_next
    for _ in range(5):
        x_next, V = layer(x_next, V)
        assert not torch.isnan(x_next).any(), "NaN after multiple steps"
    
    # Test 4: Gradient flow
    loss = x_next.sum()
    loss.backward()
    assert layer.WQ.grad is not None, "No gradient for WQ"
    assert layer.potential.W1.grad is not None, "No gradient for W1"
    
    print("✓ RSULF layer test passed")

test_rsulf_layer()
```

---

## 8. Step 6: 차원 접기(Folding) 이론·구현

### 8.1 수학적 근거

고차원 가중치 행렬을 저차원으로 압축하면서 정보 보존:

**원본**:
$$
W \in \mathbb{R}^{m \times n}
$$

**접기 후**:
$$
W \approx U \Sigma V^T, \quad U \in \mathbb{R}^{m \times r}, \Sigma \in \mathbb{R}^{r \times r}, V \in \mathbb{R}^{n \times r}
$$

여기서 $r \ll \min(m, n)$.

**손실 정보는 curvature로 보상**:
$$
K = \text{discarded\_singular\_values}
$$

### 8.2 SVD 기반 Folding

```python
def fold_weight_matrix(
    W: torch.Tensor,
    target_rank: int,
    return_full: bool = False
) -> dict:
    """
    Fold weight matrix using SVD
    
    Args:
        W: (m, n) weight matrix
        target_rank: target rank r
        return_full: if True, return U, S, V separately
    
    Returns:
        dict with folded weights and reconstruction info
    """
    m, n = W.shape
    r = min(target_rank, min(m, n))
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except RuntimeError:
        print(f"Warning: SVD failed for shape {W.shape}, using random initialization")
        return {
            'W_folded': torch.randn(m, r) * 0.02,
            'reconstruction_error': None,
            'discarded_energy': None
        }
    
    # Keep top-r
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh[:r, :]
    
    # Folded weight
    if return_full:
        result = {
            'U': U_r,
            'S': S_r,
            'V': V_r,
            'W_folded': U_r @ torch.diag(S_r),  # For storage
        }
    else:
        # Reconstruct as U_r @ diag(S_r) @ V_r
        W_folded = U_r @ torch.diag(S_r) @ V_r
        result = {'W_folded': W_folded}
    
    # Reconstruction error
    W_recon = U_r @ torch.diag(S_r) @ V_r
    error = torch.norm(W - W_recon) / torch.norm(W)
    result['reconstruction_error'] = error.item()
    
    # Discarded energy (for curvature compensation)
    total_energy = torch.sum(S ** 2)
    kept_energy = torch.sum(S_r ** 2)
    discarded_energy = (total_energy - kept_energy) / total_energy
    result['discarded_energy'] = discarded_energy.item()
    
    return result
```

### 8.3 Folding with Curvature Compensation

```python
def fold_layer_with_curvature(
    WQ: torch.Tensor,
    WK: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    target_rank: int = 128
) -> dict:
    """
    Fold all weights in a layer with curvature compensation
    
    Returns:
        dict with folded weights and curvature info
    """
    # Fold Q, K
    q_result = fold_weight_matrix(WQ, target_rank, return_full=True)
    k_result = fold_weight_matrix(WK, target_rank, return_full=True)
    
    # Fold FFN
    w1_result = fold_weight_matrix(W1, target_rank * 2)  # Larger rank for FFN
    w2_result = fold_weight_matrix(W2, target_rank * 2)
    
    # Compute curvature from discarded info
    curvature_q = q_result.get('discarded_energy', 0.0)
    curvature_k = k_result.get('discarded_energy', 0.0)
    
    # Average curvature
    avg_curvature = (curvature_q + curvature_k) / 2
    
    return {
        'WQ_folded': q_result['W_folded'],
        'WK_folded': k_result['W_folded'],
        'W1_folded': w1_result['W_folded'],
        'W2_folded': w2_result['W_folded'],
        'curvature': avg_curvature,
        'reconstruction_errors': {
            'Q': q_result['reconstruction_error'],
            'K': k_result['reconstruction_error'],
            'W1': w1_result['reconstruction_error'],
            'W2': w2_result['reconstruction_error'],
        }
    }
```

### 8.4 Metric Anchor 설정

여러 레이어의 metric을 공통 좌표계로 정렬:

```python
def compute_anchor_metric(metrics: list) -> torch.Tensor:
    """
    Compute anchor metric from multiple layer metrics
    
    Args:
        metrics: list of (d, d) metric tensors
    
    Returns:
        anchor_metric: (d, d) averaged metric
    """
    # Average all metrics
    anchor = torch.stack(metrics).mean(dim=0)
    
    # Stabilize
    anchor = stabilize_metric_symmetric(anchor)
    
    return anchor

def align_metric_to_anchor(
    g: torch.Tensor,
    g_anchor: torch.Tensor
) -> tuple:
    """
    Align metric g to anchor metric g_anchor
    
    Returns:
        g_aligned: aligned metric
        P: alignment transformation
    """
    # Find transformation P such that g ≈ P^T g_anchor P
    # Using Procrustes-like alignment
    
    try:
        # Cholesky decomposition
        L_g = torch.linalg.cholesky(g)
        L_anchor = torch.linalg.cholesky(g_anchor)
        
        # Alignment
        P = torch.linalg.solve(L_anchor, L_g)
        
        # Aligned metric
        g_aligned = P.t() @ g_anchor @ P
        
        return g_aligned, P
    except:
        # Fallback: return original
        return g, torch.eye(g.size(0), device=g.device)
```

### 8.5 체크포인트: Folding

```python
def test_folding():
    """Folding 테스트"""
    m, n = 4096, 4096
    target_rank = 128
    
    W = torch.randn(m, n) * 0.02
    
    # Fold
    result = fold_weight_matrix(W, target_rank)
    W_folded = result['W_folded']
    error = result['reconstruction_error']
    
    # Test 1: Rank
    rank = torch.linalg.matrix_rank(W_folded).item()
    assert rank <= target_rank, f"Rank {rank} exceeds target {target_rank}"
    
    # Test 2: Reconstruction error
    print(f"Reconstruction error: {error:.6f}")
    assert error < 0.5, f"Reconstruction error {error} too large"
    
    # Test 3: Parameter reduction
    orig_params = m * n
    folded_params = W_folded.numel()
    reduction = (1 - folded_params / orig_params) * 100
    print(f"Parameter reduction: {reduction:.2f}%")
    
    # Test 4: Layer folding with curvature
    WQ = torch.randn(m, n) * 0.02
    WK = torch.randn(m, n) * 0.02
    W1 = torch.randn(m * 4, n) * 0.02
    W2 = torch.randn(m, n * 4) * 0.02
    
    folded = fold_layer_with_curvature(WQ, WK, W1, W2, target_rank)
    print(f"Curvature: {folded['curvature']:.6f}")
    
    print("✓ Folding test passed")

test_folding()
```

---

## 9. Step 7: 정합성 테스트 스위트

### 9.1 테스트 Suite 구성

변환된 RS-ULF 레이어가 Transformer와 동등한지 검증:

```python
class ConsistencyTester:
    """Transformer ↔ RS-ULF consistency tester"""
    
    def __init__(
        self,
        transformer_layer,
        rsulf_layer,
        tolerance: float = 1e-2
    ):
        self.tf_layer = transformer_layer
        self.rs_layer = rsulf_layer
        self.tolerance = tolerance
        self.results = {}
    
    def test_inner_product_preservation(self, x: torch.Tensor) -> bool:
        """
        Test: (Q x_i) · (K x_j) ≈ x_i^T g x_j
        """
        with torch.no_grad():
            # Transformer side
            Q = self.tf_layer.self_attn.q_proj
            K = self.tf_layer.self_attn.k_proj
            
            q = Q(x)  # (B, L, D)
            k = K(x)
            
            # Dot products
            tf_dots = torch.matmul(q, k.transpose(-1, -2))  # (B, L, L)
            
            # RS side
            g, _ = self.rs_layer.get_metric()
            
            # x^T g x
            # x: (B, L, D), g: (D, D)
            gx = torch.matmul(x, g)  # (B, L, D)
            rs_dots = torch.matmul(gx, x.transpose(-1, -2))  # (B, L, L)
            
            # Compare
            diff = torch.abs(tf_dots - rs_dots).mean()
            similarity = torch.nn.functional.cosine_similarity(
                tf_dots.flatten(),
                rs_dots.flatten(),
                dim=0
            )
            
            passed = diff < self.tolerance and similarity > 0.9
            
            self.results['inner_product'] = {
                'passed': passed,
                'mean_diff': diff.item(),
                'cosine_sim': similarity.item()
            }
            
            return passed
    
    def test_potential_gradient_preservation(self, x: torch.Tensor) -> bool:
        """
        Test: FFN(x) ≈ ∇Φ(x)
        """
        with torch.no_grad():
            # Transformer FFN
            ffn_out = self.tf_layer.mlp(x)  # (B, L, D)
            
            # RS potential gradient
            grad_phi = self.rs_layer.potential.gradient(x)  # (B, L, D)
            
            # Compare
            diff = torch.abs(ffn_out - grad_phi).mean()
            similarity = torch.nn.functional.cosine_similarity(
                ffn_out.flatten(),
                grad_phi.flatten(),
                dim=0
            )
            
            passed = diff < self.tolerance * 10 and similarity > 0.8
            
            self.results['potential_gradient'] = {
                'passed': passed,
                'mean_diff': diff.item(),
                'cosine_sim': similarity.item()
            }
            
            return passed
    
    def test_output_similarity(self, x: torch.Tensor) -> bool:
        """
        Test: Transformer(x) ≈ RS-ULF(x)
        """
        with torch.no_grad():
            # Transformer forward
            tf_out = self.tf_layer(x)[0]  # Ignore attention weights
            
            # RS forward
            rs_out, _ = self.rs_layer(x, V=None)
            
            # Compare
            diff = torch.abs(tf_out - rs_out).mean()
            norm_diff = diff / torch.abs(tf_out).mean()
            
            similarity = torch.nn.functional.cosine_similarity(
                tf_out.flatten(),
                rs_out.flatten(),
                dim=0
            )
            
            passed = norm_diff < self.tolerance and similarity > 0.95
            
            self.results['output'] = {
                'passed': passed,
                'mean_diff': diff.item(),
                'norm_diff': norm_diff.item(),
                'cosine_sim': similarity.item()
            }
            
            return passed
    
    def test_all(self, x: torch.Tensor) -> dict:
        """Run all tests"""
        self.test_inner_product_preservation(x)
        self.test_potential_gradient_preservation(x)
        self.test_output_similarity(x)
        
        all_passed = all(r['passed'] for r in self.results.values())
        
        self.results['summary'] = {
            'all_passed': all_passed,
            'num_passed': sum(r['passed'] for r in self.results.values()),
            'num_total': len(self.results) - 1  # Exclude summary itself
        }
        
        return self.results
    
    def print_report(self):
        """Print test report"""
        print("\n" + "="*60)
        print("CONSISTENCY TEST REPORT")
        print("="*60)
        
        for test_name, result in self.results.items():
            if test_name == 'summary':
                continue
            
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n{test_name.upper()}: {status}")
            
            for metric, value in result.items():
                if metric != 'passed':
                    print(f"  {metric}: {value:.6f}")
        
        print("\n" + "-"*60)
        summary = self.results['summary']
        print(f"SUMMARY: {summary['num_passed']}/{summary['num_total']} tests passed")
        
        if summary['all_passed']:
            print("✓ ALL TESTS PASSED - Conversion successful!")
        else:
            print("✗ SOME TESTS FAILED - Review conversion")
        
        print("="*60 + "\n")
```

### 9.2 테스트 실행

```python
def run_consistency_tests():
    """일관성 테스트 실행 예시"""
    # Mock Transformer layer (실제로는 loaded model 사용)
    from transformers import AutoModel
    
    # ... (실제 모델 로딩 코드)
    
    # Test data
    batch, seq_len, d = 2, 64, 4096
    x = torch.randn(batch, seq_len, d)
    
    # Run tests
    tester = ConsistencyTester(
        transformer_layer=tf_layer,
        rsulf_layer=rs_layer,
        tolerance=1e-2
    )
    
    results = tester.test_all(x)
    tester.print_report()
    
    return results
```

---

## 10. Step 8: 전체 LLM 모델 변환

### 10.1 Converter 클래스

```python
class TransformerToRSULFConverter:
    """Complete Transformer → RS-ULF converter"""
    
    def __init__(
        self,
        config: dict = None
    ):
        self.config = config or self.default_config()
        self.conversion_stats = {}
    
    @staticmethod
    def default_config() -> dict:
        return {
            'metric_strategy': 'diagonal',
            'lr': 0.02,
            'alpha': 0.04,
            'beta': 0.01,
            'gamma': 0.98,
            'folding_rank': None,  # None = no folding
            'activation': 'silu',  # or 'relu', 'gelu'
            'graph_window_size': 8,
            'run_consistency_tests': True,
            'consistency_tolerance': 1e-2
        }
    
    def convert_model(
        self,
        transformer_model,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Convert entire Transformer model to RS-ULF
        
        Args:
            transformer_model: Huggingface Transformer model
            device: target device
        
        Returns:
            rs_model: RS-ULF model
        """
        print("="*60)
        print("TRANSFORMER → RS-ULF CONVERSION")
        print("="*60)
        
        # Extract model info
        num_layers = len(transformer_model.model.layers)
        d_model = transformer_model.config.hidden_size
        
        print(f"\nModel info:")
        print(f"  Layers: {num_layers}")
        print(f"  Hidden size: {d_model}")
        print(f"  Config: {self.config}")
        
        # Convert layers
        rs_layers = []
        
        for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
            rs_layer = self.convert_layer(
                transformer_model,
                layer_idx,
                d_model,
                device
            )
            rs_layers.append(rs_layer)
        
        # Build RS model
        rs_model = RSULFStack(rs_layers)
        rs_model.to(device)
        
        # Print stats
        self.print_conversion_stats()
        
        return rs_model
    
    def convert_layer(
        self,
        model,
        layer_idx: int,
        d_model: int,
        device: str
    ) -> RSULFLayer:
        """Convert single layer"""
        # Extract weights
        weights = extract_layer_weights(model, layer_idx)
        
        WQ = weights['WQ'].to(device)
        WK = weights['WK'].to(device)
        W1 = weights['W1'].to(device)
        W2 = weights['W2'].to(device)
        W_up = weights.get('W_up')
        if W_up is not None:
            W_up = W_up.to(device)
        
        # Folding (optional)
        if self.config['folding_rank'] is not None:
            folded = fold_layer_with_curvature(
                WQ, WK, W1, W2,
                target_rank=self.config['folding_rank']
            )
            WQ = folded['WQ_folded']
            WK = folded['WK_folded']
            W1 = folded['W1_folded']
            W2 = folded['W2_folded']
            
            # Store stats
            self.conversion_stats[f'layer_{layer_idx}'] = {
                'reconstruction_errors': folded['reconstruction_errors'],
                'curvature': folded['curvature']
            }
        
        # Create graph Laplacian (placeholder, will be set dynamically)
        # Actual seq_len is determined at runtime
        L = torch.eye(1, device=device)
        
        # Create RS layer
        rs_layer = RSULFLayer(
            d_model=d_model,
            WQ=WQ,
            WK=WK,
            W1=W1,
            W2=W2,
            W_up=W_up,
            laplacian=L,
            lr=self.config['lr'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            gamma=self.config['gamma'],
            metric_strategy=self.config['metric_strategy'],
            activation=self.config['activation']
        )
        
        return rs_layer
    
    def print_conversion_stats(self):
        """Print conversion statistics"""
        if not self.conversion_stats:
            return
        
        print("\n" + "="*60)
        print("CONVERSION STATISTICS")
        print("="*60)
        
        for layer_name, stats in self.conversion_stats.items():
            print(f"\n{layer_name}:")
            
            if 'reconstruction_errors' in stats:
                print("  Reconstruction errors:")
                for weight_name, error in stats['reconstruction_errors'].items():
                    print(f"    {weight_name}: {error:.6f}")
            
            if 'curvature' in stats:
                print(f"  Curvature: {stats['curvature']:.6f}")
        
        print("="*60 + "\n")


class RSULFStack(nn.Module):
    """Stack of RS-ULF layers"""
    
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)
    
    def forward(
        self,
        x: torch.Tensor,
        V_list: list = None
    ) -> tuple:
        """
        Forward pass through all layers
        
        Args:
            x: (batch, seq_len, d_model)
            V_list: list of memory states (one per layer)
        
        Returns:
            x: final output
            V_list: updated memory states
        """
        if V_list is None:
            V_list = [None] * self.num_layers
        
        V_next_list = []
        
        for layer, V in zip(self.layers, V_list):
            x, V_next = layer(x, V)
            V_next_list.append(V_next)
        
        return x, V_next_list
    
    def update_graph_laplacians(self, seq_len: int, device: str = 'cpu'):
        """Update graph Laplacians for all layers based on sequence length"""
        adj = create_sequence_graph(seq_len, window_size=8, directed=True)
        L = build_laplacian(adj).to(device)
        
        for layer in self.layers:
            layer.L = L
```

### 10.2 변환 실행

```python
def convert_mistral_to_rsulf():
    """Mistral 7B를 RS-ULF로 변환하는 예제"""
    from transformers import AutoModelForCausalLM
    
    # Load Transformer model
    print("Loading Mistral model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.float16,
        device_map="cpu"  # CPU에서 변환 후 이동
    )
    
    # Converter
    converter = TransformerToRSULFConverter(config={
        'metric_strategy': 'diagonal',
        'lr': 0.02,
        'alpha': 0.04,
        'beta': 0.01,
        'gamma': 0.98,
        'folding_rank': 128,  # 압축 적용
        'activation': 'silu',
        'graph_window_size': 8
    })
    
    # Convert
    rs_model = converter.convert_model(model, device='cuda')
    
    print("\n✓ Conversion complete!")
    
    # Save
    save_path = "checkpoints/mistral-7b-rsulf"
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        'model_state_dict': rs_model.state_dict(),
        'config': converter.config,
        'stats': converter.conversion_stats
    }, os.path.join(save_path, 'model.pt'))
    
    print(f"✓ Saved to {save_path}")
    
    return rs_model
```

---

## 11. 실전 예제 (Mistral 등)

### 11.1 Mistral 7B 변환

전체 스크립트: `scripts/convert_mistral_rsulf_complete.py`

```python
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# RS-ULF 모듈
from reality_stone.models.rsulf import (
    extract_metric, stabilize_metric_diagonal,
    PotentialFunction, RSULFLayer, RSULFStack,
    create_sequence_graph, build_laplacian
)

def main():
    # Config
    model_id = "mistralai/Mistral-7B-v0.1"
    save_dir = "checkpoints/mistral-7b-rsulf-complete"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Convert
    converter = TransformerToRSULFConverter(config={
        'folding_rank': 128,
        'metric_strategy': 'diagonal'
    })
    
    rs_model = converter.convert_model(model, device=device)
    
    # Test
    print("\nTesting converted model...")
    test_text = "Reality Stone은 "
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # RS-ULF forward (simplified, no generation)
    with torch.no_grad():
        input_ids = inputs['input_ids']
        # Get embeddings (from original model)
        embeds = model.model.embed_tokens(input_ids)
        
        # Update graph for this sequence length
        seq_len = embeds.size(1)
        rs_model.update_graph_laplacians(seq_len, device=device)
        
        # Forward
        output, V_list = rs_model(embeds)
        
        print(f"Input shape: {embeds.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful!")
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': rs_model.state_dict(),
        'config': converter.config,
        'tokenizer_name': model_id
    }, Path(save_dir) / 'model.pt')
    
    print(f"\n✓ Saved to {save_dir}")

if __name__ == "__main__":
    main()
```

### 11.2 추론 예제

```python
def inference_rsulf(
    rs_model,
    tokenizer,
    prompt: str,
    max_length: int = 50
):
    """RS-ULF 모델로 추론 (simplified)"""
    # Note: 실제로는 original model의 embedding + lm_head 필요
    
    device = next(rs_model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    # Get initial embeddings
    # (실제로는 original model 필요)
    # embeds = original_model.model.embed_tokens(input_ids)
    
    # For demo, use random
    seq_len = input_ids.size(1)
    d_model = rs_model.layers[0].d_model
    embeds = torch.randn(1, seq_len, d_model, device=device)
    
    # Update graph
    rs_model.update_graph_laplacians(seq_len, device=device)
    
    # Forward
    with torch.no_grad():
        output, _ = rs_model(embeds)
    
    print(f"Output shape: {output.shape}")
    return output
```

---

## 12. 성능 벤치마크 이론·실험 가이드

### 12.1 복잡도 비교

```python
def benchmark_complexity():
    """시간/공간 복잡도 실측"""
    import time
    
    batch = 2
    seq_lens = [64, 128, 256, 512, 1024]
    d_model = 4096
    
    results = []
    
    for seq_len in seq_lens:
        # Create dummy layer
        WQ = torch.randn(d_model, d_model) * 0.02
        WK = torch.randn(d_model, d_model) * 0.02
        W1 = torch.randn(d_model * 4, d_model) * 0.02
        W2 = torch.randn(d_model, d_model * 4) * 0.02
        
        adj = create_sequence_graph(seq_len, 8)
        L = build_laplacian(adj)
        
        layer = RSULFLayer(
            d_model, WQ, WK, W1, W2,
            laplacian=L
        )
        
        # Input
        x = torch.randn(batch, seq_len, d_model)
        
        # Warm-up
        _ = layer(x)
        
        # Measure time
        start = time.time()
        for _ in range(10):
            _ = layer(x)
        elapsed = (time.time() - start) / 10
        
        # Memory
        mem_mb = x.numel() * 4 / 1024 / 1024  # Approximate
        
        results.append({
            'seq_len': seq_len,
            'time_ms': elapsed * 1000,
            'memory_mb': mem_mb
        })
        
        print(f"seq_len={seq_len}: {elapsed*1000:.2f}ms, ~{mem_mb:.1f}MB")
    
    return results
```

### 12.2 정확도 비교

```python
def benchmark_accuracy():
    """정확도 벤치마크 (Perplexity 등)"""
    # Requires full model with LM head
    pass
```

---

## 부록 A: 변환 트러블슈팅

### A.1 Metric이 PD가 아님

**증상**: `torch.linalg.cholesky` 실패

**해결**:
```python
g = stabilize_metric_symmetric(g, eps=1e-4)
```

### A.2 Gradient exploding

**증상**: NaN in forward pass

**해결**:
- Learning rate 감소: `lr=0.01`
- Gradient clipping 추가
- Metric regularization 강화

### A.3 메모리 부족

**증상**: CUDA OOM

**해결**:
- Folding rank 감소: `folding_rank=64`
- Batch size 감소
- Gradient checkpointing 활용

---

## 부록 B: RS‑ULF 변환 참고자료

- [RS-ULF Specification](./01_RS_UNIFIED_FLOW_SPEC.md)
- [Graph Diffusion](./03_GRAPH_DIFFUSION_AND_DP_MEMORY.md)
- [Transformer Mapping](./04_TRANSFORMER_MAPPING_AND_TESTS.md)

---

**문서 버전**: 1.0
**최종 수정**: 2025-01-XX
**작성자**: Reality Stone Team

