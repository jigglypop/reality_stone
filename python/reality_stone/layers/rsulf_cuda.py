import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

try:
    from reality_stone import _rust, build_causal_laplacian
    if _rust is not None:
        PyRSULFLayer = _rust.PyRSULFLayer
        PyGeodesicMemory = _rust.PyGeodesicMemory
        SplineCache = _rust.SplineCache
        PyRiemannianDiffusion = _rust.PyRiemannianDiffusion
        HAS_RUST = True
    else:
        HAS_RUST = False
except ImportError:
    try:
        from reality_stone._rust import PyRSULFLayer, PyGeodesicMemory, SplineCache, PyRiemannianDiffusion, build_causal_laplacian
        HAS_RUST = True
    except ImportError:
        HAS_RUST = False


class RSULFLayerCUDA(nn.Module):
    def __init__(
        self,
        wq: np.ndarray,
        wk: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        d_model: int = 4096,
        r: int = 1024,
        eta: float = 0.01,
        alpha: float = 0.02,
        beta: float = 0.01,
        gamma: float = 0.99,
        seq_len: int = 128,
        window: int = 8,
        global_basis: Optional[Dict] = None,
        original_block: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not HAS_RUST:
            raise RuntimeError("reality_stone._rust not available")
        
        if global_basis is not None:
            self._layer = PyRSULFLayer.new_with_basis(
                wq.astype(np.float32),
                wk.astype(np.float32),
                w1.astype(np.float32),
                w2.astype(np.float32),
                global_basis["u"].astype(np.float32),
                global_basis["rank"],
                d_model, r, eta, alpha, beta, gamma, seq_len, window
            )
        else:
            self._layer = PyRSULFLayer.new_fast(
                wq.astype(np.float32),
                wk.astype(np.float32),
                w1.astype(np.float32),
                w2.astype(np.float32),
                d_model, r, eta, alpha, beta, gamma, seq_len, window
            )
        self.d_model = d_model
        self.r = r
        self.seq_len = seq_len
        self.window = window
        self.original_block = original_block
        self._cuda_available = False
        self._components = self._layer.export_components()
        
        self._ffn_u1 = np.asarray(self._components["ffn_u1"], dtype=np.float32)
        self._ffn_s1 = np.asarray(self._components["ffn_s1"], dtype=np.float32)
        self._ffn_v1 = np.asarray(self._components["ffn_v1"], dtype=np.float32)
        self._ffn_u2 = np.asarray(self._components["ffn_u2"], dtype=np.float32)
        self._ffn_s2 = np.asarray(self._components["ffn_s2"], dtype=np.float32)
        self._ffn_v2 = np.asarray(self._components["ffn_v2"], dtype=np.float32)
        self._curvature = float(self._components["curvature"])
        self._g_inv = torch.from_numpy(np.asarray(self._components["g_inv"], dtype=np.float32))
        
        self.runtime_batch: Optional[int] = None
        self.runtime_seq_len: Optional[int] = None

        self.ln1_weight = nn.Parameter(torch.ones(d_model))
        self.ln1_bias = nn.Parameter(torch.zeros(d_model))
        self.ln2_weight = nn.Parameter(torch.ones(d_model))
        self.ln2_bias = nn.Parameter(torch.zeros(d_model))
        
        self.wq = nn.Parameter(torch.from_numpy(wq).float(), requires_grad=False)
        self.wk = nn.Parameter(torch.from_numpy(wk).float(), requires_grad=False)
        self.wv = nn.Parameter(torch.eye(d_model).float(), requires_grad=False)
        self.wo = nn.Parameter(torch.eye(d_model).float(), requires_grad=False)
        
        self.bq = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.bk = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.bv = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.bo = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        
        self.ffn_u1 = nn.Parameter(torch.from_numpy(self._ffn_u1).float(), requires_grad=False)
        self.ffn_s1 = nn.Parameter(torch.from_numpy(self._ffn_s1).float(), requires_grad=False)
        self.ffn_v1 = nn.Parameter(torch.from_numpy(self._ffn_v1).float(), requires_grad=False)
        self.ffn_u2 = nn.Parameter(torch.from_numpy(self._ffn_u2).float(), requires_grad=False)
        self.ffn_s2 = nn.Parameter(torch.from_numpy(self._ffn_s2).float(), requires_grad=False)
        self.ffn_v2 = nn.Parameter(torch.from_numpy(self._ffn_v2).float(), requires_grad=False)
        self.g_inv_param = nn.Parameter(self._g_inv, requires_grad=False)
        
        self.b1 = nn.Parameter(torch.zeros(self._ffn_u1.shape[0]), requires_grad=False)
        self.b2 = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        
        self.use_hybrid_mode = True

    def set_ln1(self, weight, bias=None):
        self.ln1_weight.data = torch.from_numpy(weight).float()
        if bias is not None:
            self.ln1_bias.data = torch.from_numpy(bias).float()

    def set_ln2(self, weight, bias=None):
        self.ln2_weight.data = torch.from_numpy(weight).float()
        if bias is not None:
            self.ln2_bias.data = torch.from_numpy(bias).float()

    def set_attention_weights(self, wv, wo):
        self.wv.data = torch.from_numpy(wv).float()
        self.wo.data = torch.from_numpy(wo).float()

    def set_biases(self, bq=None, bk=None, bv=None, bo=None, b1=None, b2=None):
        if bq is not None: self.bq.data = torch.from_numpy(bq).float()
        if bk is not None: self.bk.data = torch.from_numpy(bk).float()
        if bv is not None: self.bv.data = torch.from_numpy(bv).float()
        if bo is not None: self.bo.data = torch.from_numpy(bo).float()
        if b1 is not None: self.b1.data = torch.from_numpy(b1).float()
        if b2 is not None: self.b2.data = torch.from_numpy(b2).float()

    def forward(
        self,
        x: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = x.shape

        u = F.layer_norm(x, (dim,), self.ln1_weight, self.ln1_bias)

        q = F.linear(u, self.wq, self.bq)
        k = F.linear(u, self.wk, self.bk)
        v = F.linear(u, self.wv, self.bv)

        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        attn_out = attn_out.squeeze(1)
        
        attn_out = F.linear(attn_out, self.wo, self.bo)
        
        x_mid = x + attn_out
        
        w = F.layer_norm(x_mid, (dim,), self.ln2_weight, self.ln2_bias)
        
        w_flat = w.reshape(-1, dim)
        
        h = w_flat @ self.ffn_v1
        h = h * self.ffn_s1.unsqueeze(0)
        h = h @ self.ffn_u1.T
        
        h = h + self.b1.unsqueeze(0)
        
        h = F.gelu(h)
        
        out = h @ self.ffn_v2
        out = out * self.ffn_s2.unsqueeze(0)
        out = out @ self.ffn_u2.T
        
        out = out + self.b2.unsqueeze(0)
        
        ffn_out = out.view(batch, seq_len, dim)
        
        # g_inv scaling removed to match GPT-2 Euclidean fidelity
        # g_inv = self.g_inv_param.unsqueeze(0)
        # ffn_out = ffn_out * g_inv
        
        ffn_out = ffn_out * 1.0 
        
        x_out = x_mid + ffn_out
        
        return x_out, None

    def param_count(self) -> Tuple[int, int, float]:
        return self._layer.param_count()

    @property
    def curvature(self) -> float:
        return self._layer.curvature

    @property
    def eta(self) -> float:
        return self._layer.eta

    @property
    def alpha(self) -> float:
        return self._layer.alpha

    @property
    def beta(self) -> float:
        return self._layer.beta

    @property
    def gamma(self) -> float:
        return self._layer.gamma

    @property
    def g_diag(self) -> np.ndarray:
        return self._layer.g_diag

    @property
    def g_inv(self) -> np.ndarray:
        return self._layer.g_inv


class RSULFWrapperCUDA(nn.Module):
    def __init__(self, rsulf_layer: RSULFLayerCUDA):
        super().__init__()
        self.rsulf = rsulf_layer
        self.original_block = rsulf_layer.original_block
        self.v_mem: Optional[torch.Tensor] = None
        self.d_model = rsulf_layer.d_model

    def reset_memory(self):
        self.v_mem = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.original_block is not None:
            out = self.original_block(x)
            if isinstance(out, tuple):
                out = out[0]
            return out
        
        out, _ = self.rsulf(x, None)
        return out


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
