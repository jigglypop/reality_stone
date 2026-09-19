"""STDP learning with eligibility traces (17_AgentLoop.md F.14).

Implements spike-timing-dependent plasticity with 3-factor learning:
  e_ij[k+1] = r_e * e_ij[k] + (A+ * p_i * s_j - A- * s_i * q_j)
  dW_ij = lr * g[t] * e_ij
  W_{t+1} = Proj(W_t + dW_t)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    from .constants import (
        STDP_R_PLUS, STDP_R_MINUS, STDP_R_E,
        STDP_A_PLUS, STDP_A_MINUS, STDP_SPIKE_THRESHOLD,
        STDP_LR, STDP_ALPHA_G, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
    )
except ImportError:
    from reality_stone.clarus.constants import (
        STDP_R_PLUS, STDP_R_MINUS, STDP_R_E,
        STDP_A_PLUS, STDP_A_MINUS, STDP_SPIKE_THRESHOLD,
        STDP_LR, STDP_ALPHA_G, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
    )


@dataclass
class STDPConfig:
    dim: int
    r_plus: float = STDP_R_PLUS
    r_minus: float = STDP_R_MINUS
    r_e: float = STDP_R_E
    a_plus: float = STDP_A_PLUS
    a_minus: float = STDP_A_MINUS
    spike_threshold: float = STDP_SPIKE_THRESHOLD
    lr: float = STDP_LR
    alpha_g: float = STDP_ALPHA_G
    # ``legacy`` preserves the historical trace indexing.  ``causal`` is an
    # explicit opt-in convention for a row=post, column=pre recurrent matrix:
    # a pre spike on i followed by a post spike on j potentiates W[j, i].
    orientation: str = "legacy"


class EligibilityTracker:
    """Tracks pre/post synaptic traces and eligibility matrix."""

    def __init__(self, config: STDPConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.pre_trace = torch.zeros(config.dim, device=self.device)
        self.post_trace = torch.zeros(config.dim, device=self.device)
        self.eligibility = torch.zeros(config.dim, config.dim, device=self.device)

    def update(self, activation: torch.Tensor) -> None:
        """Update traces and eligibility from current activation (one R iteration)."""
        spike = (activation.abs() > self.config.spike_threshold).float()
        self.pre_trace = self.config.r_plus * self.pre_trace + spike
        self.post_trace = self.config.r_minus * self.post_trace + spike
        if self.config.orientation == "legacy":
            ltp = self.config.a_plus * torch.outer(self.pre_trace, spike)
            ltd = self.config.a_minus * torch.outer(spike, self.post_trace)
        elif self.config.orientation == "causal":
            # BrainRuntime uses ``W @ activation``.  Thus rows are receivers
            # and columns are senders; keep the causal direction unambiguous.
            ltp = self.config.a_plus * torch.outer(spike, self.pre_trace)
            ltd = self.config.a_minus * torch.outer(self.post_trace, spike)
        else:
            raise ValueError("orientation must be legacy or causal")
        self.eligibility = self.config.r_e * self.eligibility + (ltp - ltd)

    def reset(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.eligibility.zero_()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pre_trace": self.pre_trace.detach().cpu(),
            "post_trace": self.post_trace.detach().cpu(),
            "eligibility": self.eligibility.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.pre_trace = state["pre_trace"].to(self.device).float()
        self.post_trace = state["post_trace"].to(self.device).float()
        self.eligibility = state["eligibility"].to(self.device).float()


def compute_learning_gate(
    critic_score: float,
    prev_critic_score: float,
    active_ratio: float,
    target_active: float = ACTIVE_RATIO,
    target_struct: float = STRUCT_RATIO,
    target_bg: float = BACKGROUND_RATIO,
    struct_ratio: float = 0.26,
    bg_ratio: float = 0.69,
    alpha_g: float = STDP_ALPHA_G,
    dt: float = 1.0,
) -> float:
    """g[t] = alpha_g * d(c_bar)/dt + (1-alpha_g) * bootstrap_deviation (F.14.2)."""
    critic_derivative = (critic_score - prev_critic_score) / max(dt, 1e-8)
    bootstrap_dev = (
        (active_ratio - target_active) ** 2
        + (struct_ratio - target_struct) ** 2
        + (bg_ratio - target_bg) ** 2
    )
    return alpha_g * critic_derivative + (1.0 - alpha_g) * bootstrap_dev


def structural_projection(
    weight: torch.Tensor,
    density: float = ACTIVE_RATIO,
    theta_on: float = 0.01,
    theta_off: float = 0.005,
) -> torch.Tensor:
    """Proj(W) = TopK(RowNorm(Hyst(W; theta_on, theta_off)), k) (F.14.3)."""
    mask = weight.abs() > theta_on
    below = weight.abs() < theta_off
    hyst = weight * mask.float()
    hyst[below] = 0.0

    row_norms = hyst.norm(dim=1, keepdim=True).clamp(min=1e-8)
    hyst = hyst / row_norms

    k = max(1, int(math.ceil(density * weight.shape[0])))
    for i in range(weight.shape[0]):
        row = hyst[i]
        if (row != 0).sum() > k:
            _, topk_idx = torch.topk(row.abs(), k)
            new_row = torch.zeros_like(row)
            new_row[topk_idx] = row[topk_idx]
            hyst[i] = new_row

    return hyst


def apply_stdp_update(
    weight: torch.Tensor,
    tracker: EligibilityTracker,
    gate: float,
    lr: float = STDP_LR,
    density: float = ACTIVE_RATIO,
) -> torch.Tensor:
    """Full STDP weight update: W_{t+1} = Proj(W + lr * g * e)."""
    dw = lr * gate * tracker.eligibility
    new_w = weight + dw
    return structural_projection(new_w, density=density)
