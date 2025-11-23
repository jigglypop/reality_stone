"""Riemannian Adam optimizer for Poincare ball manifold."""

import torch
from torch.optim import Optimizer
import numpy as np
from typing import List, Optional, Callable

try:
    import reality_stone._rust as _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class PoincareRiemannianAdam(Optimizer):
    """Riemannian Adam optimizer for parameters on the Poincare ball.
    
    This optimizer implements Adam optimization adapted for the Poincare ball manifold
    with curvature c. It performs the following steps:
    
    1. Convert Euclidean gradient to Riemannian gradient using the metric tensor
    2. Update moment estimates (m, v) in the tangent space
    3. Apply exponential map to move along the geodesic
    4. Project back to the manifold to ensure numerical stability
    
    Args:
        params: Iterable of parameters to optimize
        c: Curvature of the Poincare ball (positive scalar)
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
    
    Example:
        >>> import torch
        >>> import reality_stone as rs
        >>> 
        >>> # Parameters on Poincare ball with c=1.0
        >>> prototypes = torch.nn.Parameter(torch.randn(10, 128) * 0.1)
        >>> optimizer = rs.optim.PoincareRiemannianAdam([prototypes], c=1.0, lr=1e-3)
        >>> 
        >>> # Training loop
        >>> for epoch in range(100):
        ...     optimizer.zero_grad()
        ...     loss = compute_loss(prototypes)
        ...     loss.backward()
        ...     optimizer.step()
    """
    
    def __init__(
        self,
        params,
        c: float,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        max_norm_eps: float = 1e-7,  # Relaxed boundary constraint for f32
    ):
        if not _HAS_RUST:
            raise RuntimeError(
                "Rust extension not available. "
                "Please build with: uv run maturin develop --features cuda"
            )
        
        if c <= 0:
            raise ValueError(f"Curvature c must be positive, got {c}")
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        if max_norm_eps < 0.0:
            raise ValueError(f"Invalid max_norm_eps value: {max_norm_eps}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, c=c, max_norm_eps=max_norm_eps)
        super().__init__(params, defaults)
        self._step = 0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step += 1
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            c = group["c"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach()
                state = self.state[p]
                
                # Initialize state on first step
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                
                m = state["m"]
                v = state["v"]
                
                # Convert to numpy for Rust call
                x_np = p.detach().cpu().numpy().astype(np.float32)
                g_np = grad.cpu().numpy().astype(np.float32)
                m_np = m.cpu().numpy().astype(np.float32)
                v_np = v.cpu().numpy().astype(np.float32)
                
                # Handle 1D tensors (e.g. bias) by reshaping to 2D
                is_1d = x_np.ndim == 1
                if is_1d:
                    x_np = x_np.reshape(1, -1)
                    g_np = g_np.reshape(1, -1)
                    m_np = m_np.reshape(1, -1)
                    v_np = v_np.reshape(1, -1)
                
                # Call Rust core implementation
                max_norm_eps = group.get("max_norm_eps", 1e-7)
                x_new_np, m_new_np, v_new_np = _rust.poincare.poincare_riemannian_adam_step_cpu(  # type: ignore[attr-defined]
                    x_np,
                    g_np,
                    m_np,
                    v_np,
                    self._step,
                    float(c),
                    float(lr),
                    float(beta1),
                    float(beta2),
                    float(eps),
                    float(max_norm_eps),
                )
                
                # Restore shape if 1D
                if is_1d:
                    x_new_np = x_new_np.reshape(-1)
                    m_new_np = m_new_np.reshape(-1)
                    v_new_np = v_new_np.reshape(-1)
                
                # Update parameter and state
                p.copy_(torch.from_numpy(x_new_np).to(p.device))
                state["m"] = torch.from_numpy(m_new_np)
                state["v"] = torch.from_numpy(v_new_np)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

