import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

# -----------------------------------------------------------------------------
# Step 9: Metric Extraction & Stabilization
# -----------------------------------------------------------------------------

def extract_metric(WQ: torch.Tensor, WK: torch.Tensor) -> torch.Tensor:
    """
    Extracts metric g from Query and Key weights.
    g = WQ^T @ WK
    """
    return torch.matmul(WQ.t(), WK)

def stabilize_metric(g: torch.Tensor, strategy: str = "diagonal", eps: float = 1e-6) -> torch.Tensor:
    """
    Stabilizes the metric g to ensure it is Positive Definite (PD).
    """
    if strategy == "diagonal":
        # Strategy A: Diagonal metric
        diag = torch.diag(g)
        # Ensure positive diagonal
        diag = torch.abs(diag) + eps
        return torch.diag(diag)
        
    elif strategy == "sym_abs":
        # Symmetrize
        g_sym = 0.5 * (g + g.t())
        # Add to diagonal to ensure PD
        eye = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
        g_sym = g_sym + eye * eps
        return g_sym
        
    else:
        # Default to diagonal as it's safest
        return stabilize_metric(g, "diagonal", eps)

# -----------------------------------------------------------------------------
# Step 10: Curvature Approximation
# -----------------------------------------------------------------------------

def curvature_from_qk(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Approximates sectional curvature K(q, k).
    K approx (q . k) / (|q||k|)
    """
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    k_norm = torch.norm(k, dim=-1, keepdim=True)
    dot = torch.sum(q * k, dim=-1, keepdim=True)
    return dot / (q_norm * k_norm + eps)

# -----------------------------------------------------------------------------
# Step 7: Riemannian Operators
# -----------------------------------------------------------------------------

def riemannian_gradient(grad_f: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
    """
    Computes Riemannian gradient.
    grad_g f = g^{-1} grad f
    """
    return torch.matmul(grad_f, g_inv)

# -----------------------------------------------------------------------------
# Step 11: Folding (Dimension Reduction + Metric Upgrade)
# -----------------------------------------------------------------------------

def fold_metric_layer(WQ: torch.Tensor, reduction_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Folds WQ to reduce dimension while preserving metric structure.
    Returns (WQ_folded, Reconstruction_Info)
    """
    d = WQ.size(1)
    target_d = int(d * reduction_ratio)
    if target_d < 1: target_d = 1
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(WQ, full_matrices=False)
    except RuntimeError:
        # Fallback for stability
        return WQ[:, :target_d], None
        
    # Keep top singular values
    S_top = S[:target_d]
    U_top = U[:, :target_d]
    V_top = Vh[:target_d, :]

    WQ_folded = U_top @ torch.diag(S_top)

    return WQ_folded, V_top

# -----------------------------------------------------------------------------
# Step 12: RSULF Unified Layer
# -----------------------------------------------------------------------------

class RSULF(nn.Module):
    def __init__(
        self,
        d_model: int,
        WQ: Optional[torch.Tensor] = None,
        WK: Optional[torch.Tensor] = None,
        W1: Optional[torch.Tensor] = None,
        W2: Optional[torch.Tensor] = None,
        L_matrix: Optional[torch.Tensor] = None,
        lr: float = 0.02,
        alpha: float = 0.04,
        beta: float = 0.01,
        gamma: float = 0.98,
        metric_strategy: str = "diagonal"
    ):
        super().__init__()
        self.d_model = d_model
        
        # Initialize weights if not provided
        if WQ is None: WQ = torch.randn(d_model, d_model) * 0.02
        if WK is None: WK = torch.randn(d_model, d_model) * 0.02
        # W1: (4*d, d) for F.linear(x, W1) -> x @ W1.T
        if W1 is None: W1 = torch.randn(d_model * 4, d_model) * 0.02
        # W2: (d, 4*d) for F.linear(h, W2) -> h @ W2.T
        if W2 is None: W2 = torch.randn(d_model, d_model * 4) * 0.02
        if L_matrix is None: L_matrix = torch.eye(d_model) 
        
        self.WQ = nn.Parameter(WQ)
        self.WK = nn.Parameter(WK)
        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)
        
        # Laplacian matrix
        self.register_buffer("L", L_matrix)
        
        # Hyperparameters
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.metric_strategy = metric_strategy
        
    def get_metric_g(self) -> torch.Tensor:
        # If WQ, WK are folded (d, r), then g is (r, r)
        # WQ.t() is (r, d), WK is (d, r). Result (r, r).
        # If they are full rank (d, d), g is (d, d).
        g = extract_metric(self.WQ, self.WK)
        g_stable = stabilize_metric(g, self.metric_strategy)
        return g_stable
    
    def get_potential_phi(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(F.linear(x, self.W1)) 
        y = F.linear(h, self.W2)         
        phi = 0.5 * torch.sum(y ** 2, dim=-1)
        return phi

    def get_potential_gradient(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            phi = self.get_potential_phi(x_in)
            grad = torch.autograd.grad(phi.sum(), x_in, create_graph=True)[0]
        return grad

    def forward(self, x: torch.Tensor, V: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # 1. Metric & Inverse
        # If folded, g is (r, r). If full, g is (d, d).
        g = self.get_metric_g() 
        r = g.size(0)
        
        if self.metric_strategy == "diagonal":
            g_diag = torch.diagonal(g)
            g_inv_diag = 1.0 / (g_diag + 1e-6)
            g_inv = torch.diag(g_inv_diag)
        else:
            g_inv = torch.inverse(g + 1e-6 * torch.eye(r, device=device))
            
        # 2. Potential Gradient (d)
        grad_phi = self.get_potential_gradient(x) # (B, L, D)
        
        # 3. Compute Update Vector v
        
        # Term 1: -eta * g^{-1} * grad_Phi
        # If g is (r, r) and grad_phi is (d), we must project grad_phi.
        # Assumption: If folded, WQ is (d, r). 
        # We project grad_phi onto the subspace defined by WQ?
        # v = -eta * WQ @ g^{-1} @ WQ.T @ grad_phi
        
        if r < dim:
            # Folded case: Project gradient -> Inverse Metric -> Project Back
            # WQ is (d, r)
            # grad_phi (B, L, d) @ WQ (d, r) -> (B, L, r)
            grad_proj = torch.matmul(grad_phi, self.WQ) 
            
            # (B, L, r) @ (r, r) -> (B, L, r)
            v_sub = -self.lr * torch.matmul(grad_proj, g_inv)
            
            # Project back: (B, L, r) @ WQ.T (r, d) -> (B, L, d)
            v1 = torch.matmul(v_sub, self.WQ.t())
        else:
            # Full rank case
            v1 = -self.lr * torch.matmul(grad_phi, g_inv)
        
        # Term 2: alpha * Delta_g x (Simple Laplacian)
        x_mean = x.mean(dim=1, keepdim=True)
        v2 = self.alpha * (x - x_mean)
        
        # Term 3: beta * L x
        v3 = self.beta * torch.matmul(x, self.L.t())
        
        # Term 4: gamma * V
        v4 = 0.0
        if V is not None:
            if V.shape == x.shape:
                v4 = self.gamma * V
            else:
                v4 = self.gamma * V.unsqueeze(-1)
        
        v = v1 + v2 + v3 + v4
        
        # 4. Exponential Map (Retraction)
        x_next = x + v
        
        # 5. Update V
        phi_val = self.get_potential_phi(x_next)
        if V is not None and V.shape == x.shape:
            V_next = v # Momentum
        else:
            if V is None: V_next = phi_val
            else: V_next = self.gamma * V + phi_val
                
        return x_next, V_next


class RSULFStack(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        x: torch.Tensor,
        V_list: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        if V_list is None:
            V_list = [None] * len(self.layers)
        next_V: List[Optional[torch.Tensor]] = []
        h = x
        for layer, V in zip(self.layers, V_list):
            h, V = layer(h, V)
            next_V.append(V)
        return h, next_V
