import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

ArrayLike = Union[torch.Tensor, "np.ndarray"]


def _to_tensor(arr: ArrayLike) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.float()
    if HAS_NUMPY and isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()
    raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(arr)}")


class RSULFLayerCUDA(nn.Module):
    def __init__(
        self,
        wq: ArrayLike,
        wk: ArrayLike,
        w1: ArrayLike,
        w2: ArrayLike,
        d_model: int = 4096,
        r: int = 1024,
        eta: float = 0.01,
        alpha: float = 0.02,
        beta: float = 0.01,
        gamma: float = 0.99,
        seq_len: int = 128,
        window: int = 8,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.r = r
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seq_len = seq_len
        self.window = window
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        wq_t = _to_tensor(wq)
        wk_t = _to_tensor(wk)
        w1_t = _to_tensor(w1)
        w2_t = _to_tensor(w2)
        
        d_q = wq_t.shape[0]
        d_k = wk_t.shape[0]
        if d_k < d_q:
            repeat = d_q // d_k
            wk_t = wk_t.repeat(repeat, 1)
        
        g_diag = torch.abs(torch.sum(wq_t * wk_t, dim=0))
        g_diag = torch.clamp(g_diag, min=1e-6, max=1e6)
        g_inv = 1.0 / g_diag
        
        self.register_buffer("g_diag", g_diag.to(device))
        self.register_buffer("g_inv", g_inv.to(device))
        
        ffn_u1, ffn_s1, ffn_v1 = self._randomized_svd(w1_t, r)
        ffn_u2, ffn_s2, ffn_v2 = self._randomized_svd(w2_t, r)
        
        self.register_buffer("ffn_u1", ffn_u1.to(device))
        self.register_buffer("ffn_s1", ffn_s1.to(device))
        self.register_buffer("ffn_v1", ffn_v1.to(device))
        self.register_buffer("ffn_u2", ffn_u2.to(device))
        self.register_buffer("ffn_s2", ffn_s2.to(device))
        self.register_buffer("ffn_v2", ffn_v2.to(device))
        
        laplacian = self._create_causal_laplacian(seq_len, window)
        self.register_buffer("laplacian", laplacian.to(device))
        
        g = wq_t.T @ wk_t
        frob_g = torch.sum(g ** 2)
        frob_approx = torch.sum(ffn_s1 ** 2)
        tail = frob_g - frob_approx
        self.curvature = float(torch.sqrt(torch.clamp(tail, min=0.0)))
        
        self.v_mem: Optional[torch.Tensor] = None
    
    def _randomized_svd(
        self, 
        a: torch.Tensor, 
        k: int, 
        n_oversamples: int = 5, 
        n_iter: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, n = a.shape
        l = min(k + n_oversamples, m, n)
        
        omega = torch.randn(n, l, device=a.device, dtype=a.dtype)
        y = a @ omega
        
        for _ in range(n_iter):
            z = a.T @ y
            y = a @ z
        
        q, _ = torch.linalg.qr(y)
        b = q.T @ a
        
        u_tilde, s, vh = torch.linalg.svd(b, full_matrices=False)
        
        k_actual = min(k, len(s))
        u = q @ u_tilde[:, :k_actual]
        s = s[:k_actual]
        v = vh[:k_actual, :].T
        
        return u, s, v
    
    def _create_causal_laplacian(self, seq_len: int, window: int) -> torch.Tensor:
        a = torch.zeros(seq_len, seq_len, device=self.device)
        for i in range(seq_len):
            start = max(0, i - window)
            for j in range(start, i):
                dist = float(i - j)
                a[i, j] = 1.0 / (1.0 + dist)
        
        d_vec = a.sum(dim=1)
        l = torch.diag(d_vec) - a
        return l
    
    def forward(
        self, 
        x: torch.Tensor, 
        v_mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        
        h1 = x @ self.ffn_v1
        h1_scaled = h1 * self.ffn_s1
        pre_act = h1_scaled @ self.ffn_u1.T
        h_act = F.silu(pre_act)
        
        p1 = h_act @ self.ffn_v2
        p1_scaled = p1 * self.ffn_s2
        f_x = p1_scaled @ self.ffn_u2.T
        
        phi_val = 0.5 * torch.sum(f_x ** 2) / batch
        
        dh_temp = f_x @ self.ffn_u2
        dh_temp_s = dh_temp * self.ffn_s2
        dh = dh_temp_s @ self.ffn_v2.T
        
        sigmoid_pre = torch.sigmoid(pre_act)
        d_sigma = sigmoid_pre + pre_act * sigmoid_pre * (1 - sigmoid_pre)
        d_pre = dh * d_sigma
        
        dx_temp = d_pre @ self.ffn_u1
        dx_temp_s = dx_temp * self.ffn_s1
        grad_phi = dx_temp_s @ self.ffn_v1.T
        
        grad_phi = grad_phi * self.g_inv
        
        if v_mem is not None:
            v_new = self.gamma * v_mem + (1.0 - self.gamma) * phi_val
        else:
            v_new = torch.full((batch,), phi_val.item(), device=x.device, dtype=x.dtype)
        
        term_opt = -self.eta * grad_phi
        
        x_mean = x.mean(dim=0, keepdim=True)
        diffusion = self.alpha * (x - x_mean)
        
        graph = torch.zeros_like(x)
        if abs(self.beta) > 0.0 and self.seq_len > 0:
            if batch >= self.seq_len and batch % self.seq_len == 0:
                num_seq = batch // self.seq_len
                x_reshaped = x.view(num_seq, self.seq_len, -1)
                gx = torch.einsum("ij,njd->nid", self.laplacian, x_reshaped)
                graph = gx.reshape(batch, -1) * self.beta
        
        v = term_opt + diffusion + graph
        
        delta = torch.zeros_like(x)
        if abs(self.curvature) > 1e-6:
            v_norm_sq = torch.sum(v ** 2, dim=1, keepdim=True)
            scale = -0.5 * self.curvature * v_norm_sq
            delta = scale * x
        
        x_next = x + v + delta
        
        return x_next, v_new
    
    def param_count(self) -> Tuple[int, int, float]:
        d = self.d_model
        r = self.r
        ffn_dim = self.ffn_u1.shape[0]
        
        original_attn = 4 * d * d
        original_ffn = 2 * d * ffn_dim + ffn_dim * d
        original = original_attn + original_ffn
        
        compressed_metric = 2 * d * r + r
        compressed_ffn = 2 * (ffn_dim * r + d * r + r)
        compressed_laplacian = self.seq_len * self.seq_len
        compressed = compressed_metric + compressed_ffn + compressed_laplacian
        
        ratio = original / compressed
        return compressed, original, ratio


class RSULFWrapperCUDA(nn.Module):
    def __init__(self, rsulf_layer: RSULFLayerCUDA):
        super().__init__()
        self.rsulf = rsulf_layer
        self.v_mem: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        out, v_new = self.rsulf(x_flat, self.v_mem)
        self.v_mem = v_new
        return out.view(batch, seq, -1)


class RSULFLMHeadCUDA(nn.Module):
    def __init__(
        self, 
        rsulf_layers: list, 
        hidden_size: int, 
        vocab_size: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.rsulf_wrappers = nn.ModuleList([
            RSULFWrapperCUDA(layer) for layer in rsulf_layers
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for wrapper in self.rsulf_wrappers:
            x = wrapper(x)
        return self.lm_head(x)

