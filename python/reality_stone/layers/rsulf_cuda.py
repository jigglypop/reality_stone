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
        self.v_mem: Optional[torch.Tensor] = None
        self.prev_x: Optional[torch.Tensor] = None
        
        self.geodesic_memory = PyGeodesicMemory(rsulf_layer.d_model, 0.05) if HAS_RUST else None
        self.spline_cache = SplineCache(rsulf_layer.curvature, rsulf_layer.d_model) if HAS_RUST else None
        self.diffusion = PyRiemannianDiffusion(rsulf_layer.d_model, 0.01, 0.1) if HAS_RUST else None
        self.time_step = 0
        
        self.norm = nn.LayerNorm(rsulf_layer.d_model, elementwise_affine=True)
        
        # Load LayerNorm params if attached to rsulf_layer
        if hasattr(rsulf_layer, "ln_1_weight") and rsulf_layer.ln_1_weight is not None:
            with torch.no_grad():
                self.norm.weight.copy_(torch.from_numpy(rsulf_layer.ln_1_weight))
                if hasattr(rsulf_layer, "ln_1_bias") and rsulf_layer.ln_1_bias is not None:
                    self.norm.bias.copy_(torch.from_numpy(rsulf_layer.ln_1_bias))
        
        if HAS_RUST:
            pass

    def reset_memory(self):
        self.v_mem = None
        if self.geodesic_memory:
            self.geodesic_memory.reset()
        if self.spline_cache:
            self.spline_cache.clear()
        self.time_step = 0
        self.prev_x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        
        batch, seq, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        if seq == 1 and self.geodesic_memory is not None:
            x_curr = x[0, 0, :].detach().cpu().numpy().astype(np.float32)
            self.geodesic_memory.push(self.time_step, x_curr)
            
            window_size = self.rsulf._layer.window if hasattr(self.rsulf._layer, 'window') else 8
            
            t_start = max(0, self.time_step - window_size + 1)
            t_end = self.time_step + 1
            timestamps = np.arange(t_start, t_end, dtype=np.float32)
            
            # 1. KV Cache Spline (Improve Context using Spline)
            if self.spline_cache:
                # Try to reconstruct context from Spline instead of raw memory
                x_context_np = self.spline_cache.batch_reconstruct(timestamps)
            else:
                context_list = []
                for t in timestamps:
                    val = self.geodesic_memory.query(t)
                    context_list.append(val)
                x_context_np = np.stack(context_list)
            
            x_context = torch.from_numpy(x_context_np).to(device, dtype=dtype)
            
            out_window, v_new_window = self.rsulf(x_context, None)

            out = out_window[-1, :].view(batch, seq, -1)
            
            if self.spline_cache:
                out_vec = out.detach().view(-1)
                out_np = out_vec.cpu().numpy().astype(np.float32)
                if self.prev_x is None:
                    v_np = np.zeros_like(out_np)
                else:
                    prev_np = self.prev_x.cpu().numpy().astype(np.float32)
                    v_np = out_np - prev_np
                self.prev_x = out_vec.detach()
                self.spline_cache.add_point(float(self.time_step), out_np, v_np)
                
            # 2. Graph Diffusion (Stabilize Output)
            if self.diffusion and x.is_cuda:
                out_np = out.detach().cpu().numpy().astype(np.float32)
                out_reshaped = out_np.reshape(1, -1)
                diffused_np = self.diffusion.step_cpu(out_reshaped, out_reshaped)
                out = torch.from_numpy(diffused_np).to(device, dtype=dtype).view(batch, seq, -1)

            self.time_step += 1
            return out

        elif seq > 1:
            self.reset_memory()
            x_np = x[0, :, :].detach().cpu().numpy().astype(np.float32)
            
            if self.geodesic_memory:
                for t in range(seq):
                    self.geodesic_memory.push(t, x_np[t])
            x_flat = x.view(-1, dim)
            out, v_new = self.rsulf(x_flat, None)
            if self.spline_cache:
                out_np = out.detach().cpu().numpy().astype(np.float32)
                vel_np = np.zeros_like(out_np)
                if seq > 1:
                    vel_np[1:, :] = out_np[1:, :] - out_np[:-1, :]
                for t in range(seq):
                    self.spline_cache.add_point(float(t), out_np[t], vel_np[t])
            if self.diffusion and x.is_cuda:
                out_np = out.detach().cpu().numpy().astype(np.float32)
                diffused_np = self.diffusion.step_cpu(out_np, out_np)
                out = torch.from_numpy(diffused_np).to(device, dtype=dtype)
            self.time_step = seq
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

