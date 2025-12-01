import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

try:
    from reality_stone._rust import PyRSULFLayer
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
    ):
        super().__init__()
        if not HAS_RUST:
            raise RuntimeError("reality_stone._rust not available")
        
        self._layer = PyRSULFLayer(
            wq.astype(np.float32),
            wk.astype(np.float32),
            w1.astype(np.float32),
            w2.astype(np.float32),
            d_model, r, eta, alpha, beta, gamma, seq_len, window
        )
        self.d_model = d_model
        self.r = r
        self.seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        x_np = x.detach().cpu().numpy().astype(np.float32)
        
        v_np = None
        if v_mem is not None:
            v_np = v_mem.detach().cpu().numpy().astype(np.float32)
        
        out_np, v_new_np = self._layer.forward(x_np, v_np)
        
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
        self.v_mem: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        out, v_new = self.rsulf(x_flat, self.v_mem)
        self.v_mem = v_new.detach()
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

