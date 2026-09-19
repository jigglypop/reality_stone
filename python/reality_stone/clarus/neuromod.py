"""Four neuromodulator system (17_AgentLoop.md F.19 / F.24.4).

g_DA[t+1] = g_DA[t] + (1/tau_DA)(g0_DA - g_DA[t]) + alpha_DA * c_pred[t]
g_NE[t+1] = g_NE[t] + (1/tau_NE)(g0_NE - g_NE[t]) + alpha_NE * c_nov[t]
g_5HT[t+1] = g_5HT[t] + (1/tau_5HT)(g0_5HT - g_5HT[t]) + alpha_5HT * (-discount[t])
g_ACh[t+1] = g_ACh[t] + (1/tau_ACh)(g0_ACh - g_ACh[t]) + alpha_ACh * salience[t]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from .constants import (
        NEURO_TAU_DA, NEURO_TAU_NE, NEURO_TAU_5HT, NEURO_TAU_ACH,
        NEURO_BASELINE_DA, NEURO_BASELINE_NE, NEURO_BASELINE_5HT, NEURO_BASELINE_ACH,
        NEURO_ALPHA_DA, NEURO_ALPHA_NE, NEURO_ALPHA_5HT, NEURO_ALPHA_ACH,
    )
except ImportError:
    from reality_stone.clarus.constants import (
        NEURO_TAU_DA, NEURO_TAU_NE, NEURO_TAU_5HT, NEURO_TAU_ACH,
        NEURO_BASELINE_DA, NEURO_BASELINE_NE, NEURO_BASELINE_5HT, NEURO_BASELINE_ACH,
        NEURO_ALPHA_DA, NEURO_ALPHA_NE, NEURO_ALPHA_5HT, NEURO_ALPHA_ACH,
    )


@dataclass
class NeuromodulatorState:
    """Four neuromodulator levels."""
    da: float = NEURO_BASELINE_DA
    ne: float = NEURO_BASELINE_NE
    sht: float = NEURO_BASELINE_5HT
    ach: float = NEURO_BASELINE_ACH

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.da, self.ne, self.sht, self.ach)


def step_neuromodulators(
    state: NeuromodulatorState,
    c_pred: float = 0.0,
    c_nov: float = 0.0,
    discount: float = 0.0,
    salience: float = 0.0,
) -> NeuromodulatorState:
    """Update 4 neuromodulator levels for one timestep (F.24.4)."""
    da = state.da + (1.0 / NEURO_TAU_DA) * (NEURO_BASELINE_DA - state.da) + NEURO_ALPHA_DA * c_pred
    ne = state.ne + (1.0 / NEURO_TAU_NE) * (NEURO_BASELINE_NE - state.ne) + NEURO_ALPHA_NE * c_nov
    sht = state.sht + (1.0 / NEURO_TAU_5HT) * (NEURO_BASELINE_5HT - state.sht) + NEURO_ALPHA_5HT * (-discount)
    ach = state.ach + (1.0 / NEURO_TAU_ACH) * (NEURO_BASELINE_ACH - state.ach) + NEURO_ALPHA_ACH * salience
    return NeuromodulatorState(
        da=max(0.0, min(da, 2.0)),
        ne=max(0.0, min(ne, 2.0)),
        sht=max(0.0, min(sht, 2.0)),
        ach=max(0.0, min(ach, 2.0)),
    )


@dataclass
class ModulationEffect:
    """Effects of neuromodulation on runtime parameters (F.19)."""
    n_iter_boost: float
    encode_threshold_scale: float
    temperature_scale: float
    exploration_boost: float


def apply_modulation(
    state: NeuromodulatorState,
    base_n_iter: int = 60,
    delta_n: int = 30,
    base_encode_threshold: float = 0.3,
    base_temperature: float = 0.3148,
    beta_5ht: float = 0.5,
) -> ModulationEffect:
    """Map neuromodulator levels to runtime parameter modulations (F.19)."""
    sigmoid_ne = 1.0 / (1.0 + math.exp(-2.0 * (state.ne - 0.5)))
    n_iter = base_n_iter + int(delta_n * sigmoid_ne)
    theta_encode = base_encode_threshold / (1.0 + state.ach)
    t_eff = base_temperature * (1.0 + beta_5ht * state.sht)
    exploration = state.da * 0.5
    return ModulationEffect(
        n_iter_boost=float(n_iter),
        encode_threshold_scale=theta_encode,
        temperature_scale=t_eff,
        exploration_boost=exploration,
    )
