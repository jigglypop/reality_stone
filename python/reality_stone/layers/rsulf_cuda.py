import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np

try:
    # Try importing from the package level first (where __init__ logic ran)
    from reality_stone import _rust
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
        # Fallback: direct import
        from reality_stone._rust import PyRSULFLayer, PyGeodesicMemory, SplineCache, PyRiemannianDiffusion
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
        self.original_block = original_block

    def forward(
        self,
        x: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        
        # Ensure input is on CPU and float32 for Rust backend
        if x.is_cuda:
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = x.detach().numpy().astype(np.float32)
        
        v_np = None
        if v_mem is not None:
            if v_mem.is_cuda:
                v_np = v_mem.detach().cpu().numpy().astype(np.float32)
            else:
                v_np = v_mem.detach().numpy().astype(np.float32)
        
        try:
            out_np, v_new_np = self._layer.forward(x_np, v_np)
        except Exception as e:
            print(f"[RSULFLayerCUDA] Rust forward error: {e}")
            print(f"  x_shape: {x_np.shape}, v_shape: {v_np.shape if v_np is not None else 'None'}")
            raise e
        
        out = torch.from_numpy(out_np).to(device=device, dtype=dtype)
        v_new = torch.from_numpy(v_new_np).to(device=device, dtype=dtype)
        return out, v_new

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
        
        batch, seq, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        x_flat = x.view(-1, dim)
        out, _ = self.rsulf(x_flat, None)
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

