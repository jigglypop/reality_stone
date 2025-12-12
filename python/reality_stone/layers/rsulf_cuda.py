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
        use_fast: bool = True,
        calibration_samples: int = 1024,
        num_heads: int = 1,
        pfc_mode: str = "bilinear",
        pfc_curvature: float = 0.0,
        pfc_max_rel: float = 0.02,
        pfc_window: int = 0,
        pfc_speed_gate: float = 1.0,
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
            if use_fast:
                self._layer = PyRSULFLayer.new_fast(
                    wq.astype(np.float32),
                    wk.astype(np.float32),
                    w1.astype(np.float32),
                    w2.astype(np.float32),
                    d_model, r, eta, alpha, beta, gamma, seq_len, window, calibration_samples
                )
            else:
                wq_f = wq.astype(np.float32)
                wk_f = wk.astype(np.float32)
                if wk_f.shape[0] < wq_f.shape[0]:
                    repeat = wq_f.shape[0] // wk_f.shape[0]
                    wk_f = np.tile(wk_f, (repeat, 1))
                b = wq_f.T @ wk_f
                g_sym = (b + b.T) * 0.5
                g_diag = np.abs(np.diag(g_sym)).astype(np.float32)
                g_diag[g_diag < 1e-6] = 1e-6
                g_diag[g_diag > 1e6] = 1e6
                self._layer = PyRSULFLayer.new_with_metric(
                    wq_f,
                    wk_f,
                    w1.astype(np.float32),
                    w2.astype(np.float32),
                    g_diag,
                    d_model, r, eta, alpha, beta, gamma, seq_len, window
                )
        self.d_model = d_model
        self.r = r
        self.seq_len = seq_len
        self.window = window
        self.num_heads = int(max(1, num_heads))
        self.pfc_mode = str(pfc_mode).lower().strip()
        self.pfc_curvature = float(pfc_curvature)
        self.pfc_max_rel = float(pfc_max_rel)
        self.pfc_window = int(max(0, pfc_window))
        self.pfc_speed_gate = float(max(0.0, pfc_speed_gate))
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

    def _pfc_cap_and_project(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # Remove component parallel to h (keeps correction in tangent-ish direction)
        hh = (h * h).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        ch = (corr * h).mean(dim=-1, keepdim=True)
        corr = corr - (ch / hh) * h

        # Relative RMS cap
        h_rms = (h * h).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        c_rms = (corr * corr).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        max_rel = float(self.pfc_max_rel)
        scale = (max_rel * h_rms / c_rms).clamp(max=1.0)
        return corr * scale

    def _pfc_gate(self, v: torch.Tensor) -> torch.Tensor:
        # Gate by local "drift speed" to focus curvature force where path is bending.
        # pfc_speed_gate=0 disables gating.
        sg = float(self.pfc_speed_gate)
        if sg <= 0.0:
            return torch.ones_like(v[..., :1])
        speed = (v * v).mean(dim=-1, keepdim=True).sqrt()
        ref = speed.mean(dim=1, keepdim=True).clamp(min=1e-6)
        gate = (speed / ref).clamp(0.0, 3.0)
        gate = gate.pow(sg)
        return gate

    def _pfc_bilinear(self, h_seq: torch.Tensor) -> torch.Tensor:
        c = float(self.pfc_curvature)
        if c == 0.0:
            return h_seq
        if h_seq.size(1) < 2:
            return h_seq
        w = int(self.pfc_window)
        if w <= 0:
            w = int(h_seq.size(1) - 1)
        if w > int(h_seq.size(1) - 1):
            w = int(h_seq.size(1) - 1)
        v1 = self.ffn_v1
        u2 = self.ffn_u2
        r = int(v1.shape[1])
        if r <= 0:
            return h_seq
        h_out = h_seq.clone()
        h_f = h_seq.float()
        h_tail = h_f[:, -w:, :]
        h_prev = h_f[:, -(w + 1):-1, :]
        v_tail = h_tail - h_prev
        v1_f = v1.float()
        u2_f = u2.float()
        vv1 = v_tail.reshape(-1, h_f.size(-1)) @ v1_f
        hu2 = h_tail.reshape(-1, h_f.size(-1)) @ u2_f
        corr = (vv1 * hu2) @ v1_f.T
        corr = corr * (c / float(r))
        corr = corr.view_as(h_tail)
        corr = self._pfc_cap_and_project(h_tail, corr)
        corr = corr * self._pfc_gate(v_tail)
        h_tail_out = (h_tail + corr).to(dtype=h_seq.dtype)
        h_out[:, -w:, :] = h_tail_out
        return h_out

    def _pfc_accel(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        Universal PFC (trajectory-only):
        use discrete acceleration a_t = h_t - 2 h_{t-1} + h_{t-2} as a proxy for path curvature,
        then damp it: h_t' = h_t - c * a_t (with gating + relative cap).
        """
        c = float(self.pfc_curvature)
        if c == 0.0:
            return h_seq
        if h_seq.size(1) < 3:
            return h_seq

        w = int(self.pfc_window)
        # need 2 prev tokens, so max tail tokens is (T-2)
        if w <= 0:
            w = int(h_seq.size(1) - 2)
        w = min(w, int(h_seq.size(1) - 2))
        if w <= 0:
            return h_seq

        h_out = h_seq.clone()
        h_f = h_seq.float()
        h_t = h_f[:, -w:, :]                 # (..., t)
        h_t1 = h_f[:, -(w + 1):-1, :]        # (..., t-1)
        h_t2 = h_f[:, -(w + 2):-2, :]        # (..., t-2)

        v = h_t - h_t1
        a = h_t - 2.0 * h_t1 + h_t2

        corr = (-c) * a
        corr = self._pfc_cap_and_project(h_t, corr)
        corr = corr * self._pfc_gate(v)
        h_tail_out = (h_t + corr).to(dtype=h_seq.dtype)
        h_out[:, -w:, :] = h_tail_out
        return h_out

    def _apply_pfc(self, h_seq: torch.Tensor) -> torch.Tensor:
        mode = self.pfc_mode
        if mode in ("0", "off", "none", "false", ""):
            return h_seq
        if mode in ("bilinear", "ffn", "legacy"):
            return self._pfc_bilinear(h_seq)
        if mode in ("accel", "acceleration", "geodesic", "universal"):
            return self._pfc_accel(h_seq)
        # Unknown mode: fail safe (no correction)
        return h_seq

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

        n_head = self.num_heads
        head_dim = dim // n_head
        if head_dim * n_head != dim:
            n_head = 1
            head_dim = dim

        q = q.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, n_head, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        
        attn_out = F.linear(attn_out, self.wo, self.bo)
        
        x_mid = x + attn_out
        x_mid = self._apply_pfc(x_mid)
        
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
