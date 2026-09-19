"""Reference brain runtime for the Python control plane.

This module intentionally keeps policy in Python while delegating reusable
numeric kernels to `reality_stone.clarus.ce_ops` / `reality_stone.clarus._rust`.

Concept layout from the refactor plan:
- `BrainRuntimeConfig`: global mode, lifecycle, and energy-budget policy
- `HippocampusMemory`: minimal fast-memory / replay subsystem
- `BrainRuntime`: sparse lifecycle + mode switching + snapshot continuity
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict

import numpy as np
import torch

try:
    from .ce_ops import checked_sparse_csr_tensor, pack_sparse
    from .constants import (
        MEMORY_TRACE_DECAY, ADAPTATION_DECAY, ADAPTATION_COUPLING,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE, ADAPTATION_CLAMP,
        TAU_W_STEPS, TAU_S_STEPS, SLEEP_PRESSURE_MAX, REM_TAU_FACTOR,
        NORM_EPS, NOISE_SIGMA, DALE_EI_RATIO, DALE_INH_GAIN,
        AXON_DELAY_MAX, CIRCADIAN_PERIOD, CIRCADIAN_AMP, CIRCADIAN_BASE,
        NREM_LENGTH_DECAY, FORGET_TAU, RECALL_SIMILARITY_THRESHOLD,
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BAND_DELTA, BAND_THETA, BAND_ALPHA,
        BAND_BETA, BAND_GAMMA,
    )
    from .stdp import (
        STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update,
    )
except ImportError:
    from reality_stone.clarus.ce_ops import checked_sparse_csr_tensor, pack_sparse
    from reality_stone.clarus.constants import (
        MEMORY_TRACE_DECAY, ADAPTATION_DECAY, ADAPTATION_COUPLING,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE, ADAPTATION_CLAMP,
        TAU_W_STEPS, TAU_S_STEPS, SLEEP_PRESSURE_MAX, REM_TAU_FACTOR,
        NORM_EPS, NOISE_SIGMA, DALE_EI_RATIO, DALE_INH_GAIN,
        AXON_DELAY_MAX, CIRCADIAN_PERIOD, CIRCADIAN_AMP, CIRCADIAN_BASE,
        NREM_LENGTH_DECAY, FORGET_TAU, RECALL_SIMILARITY_THRESHOLD,
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BAND_DELTA, BAND_THETA, BAND_ALPHA,
        BAND_BETA, BAND_GAMMA,
    )
    from reality_stone.clarus.stdp import (
        STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update,
    )

try:
    from ._rust import nn_brain_step as _rust_brain_step
    _HAS_RUST_KERNEL = True
except ImportError:
    _HAS_RUST_KERNEL = False

_MODE_TO_INT = {
    "WAKE": 0,
    "NREM": 1,
    "REM": 2,
}


class RuntimeMode(str, Enum):
    WAKE = "WAKE"
    NREM = "NREM"
    REM = "REM"


class ModuleLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    DORMANT = "DORMANT"
    SLEEPING = "SLEEPING"


_LIFECYCLE_TO_CODE = {
    ModuleLifecycle.ACTIVE: 0,
    ModuleLifecycle.IDLE: 1,
    ModuleLifecycle.DORMANT: 2,
    ModuleLifecycle.SLEEPING: 3,
}
_CODE_TO_LIFECYCLE = {value: key for key, value in _LIFECYCLE_TO_CODE.items()}


try:
    from .utils import normalize_vector as _normalize
except ImportError:
    from reality_stone.clarus.utils import normalize_vector as _normalize


@dataclass
class BrainRuntimeConfig:
    """Global runtime knobs for sparse activation, modes, and replay."""
    dim: int
    active_ratio: float = 0.125
    idle_threshold: float = 0.08
    active_threshold: float = 0.22
    # Explicit experiment-only selection override. Default false preserves the
    # legacy salience/budget path; true makes every module active after a step.
    force_all_active_selection: bool = False
    bit_lower_threshold: float = 0.10
    bit_upper_threshold: float = 0.30
    neuronwise_active_threshold: tuple[float, ...] | None = None
    neuronwise_bit_lower_threshold: tuple[float, ...] | None = None
    neuronwise_bit_upper_threshold: tuple[float, ...] | None = None
    refractory_scale: float = 0.35
    replay_gain: float = 0.28
    goal_gain: float = 0.20
    external_gain: float = 0.45
    zero_tol: float = 0.0
    dormant_after: int = 3
    sleeping_after: int = 6
    wake_threshold: float = 0.18
    memory_capacity: int = 32
    memory_topk: int = 4
    noise_sigma: float = NOISE_SIGMA
    dale_law: bool = True
    axon_delay: bool = True
    max_axon_delay: int = AXON_DELAY_MAX
    forget_tau: float = FORGET_TAU
    # Optional local competition/homeostasis group.  The default ``None``
    # leaves the legacy runtime byte-for-byte on its old path.  When enabled,
    # the state is Torch-only and is part of snapshots.
    competition_indices: tuple[int, ...] | None = None
    competition_lateral_gain: float = 0.0
    competition_homeostasis_gain: float = 0.0
    competition_homeostasis_rate: float = 0.0
    competition_homeostasis_decay: float = 0.0
    competition_novelty_decay: float = 0.8
    competition_delay_ticks: int = 1
    competition_epsilon: float = 1e-8
    # Optional exchangeable multiplicative jitter applied only to a delivered
    # positive competition packet.  It is zero-preserving (no packet means no
    # hidden drive) and is disabled by default.  The seed is part of the
    # structural runtime receipt so snapshot continuation is deterministic.
    competition_jitter_sigma: float = 0.0
    competition_jitter_seed: int = 0
    # Optional strict k-WTA budget derived from the explicit source coordinates
    # in the axon packet arriving on this tick.  Defaults preserve the legacy
    # one-winner max-relative competition path exactly.
    competition_input_indices: tuple[int, ...] | None = None
    competition_k_from_delayed_input: bool = False
    competition_factorize_delayed_input: bool = False
    # F1 self-organization (docs/7_AGI/12_Equation.md A.2 condition #2).
    # When enabled, the runtime feeds the empirical active ratio
    #   p_emp = |A_t| / dim
    # back into the next budget so it contracts toward ACTIVE_RATIO (epsilon^2).
    f1_self_measure: bool = False
    f1_pull_strength: float = 0.5
    f1_ema_alpha: float = 0.1
    f1_min_ratio: float = 0.005
    f1_max_ratio: float = 0.5
    # F14 local learning. Kept off by default so inference/runtime tests remain
    # deterministic unless the caller explicitly opts into plastic weights.
    stdp_enabled: bool = False
    stdp_interval: int = 1
    stdp_apply_interval: int = 10
    stdp_lr: float = 0.001
    stdp_density: float = ACTIVE_RATIO
    stdp_gate_threshold: float = 1e-3
    stdp_spike_threshold: float = 0.3
    # Experimental alternative to the legacy critic-derivative gate.  The
    # default is unchanged; external_signed must be explicitly selected and
    # supplied a causal signal by the caller.
    stdp_gate_mode: str = "critic_derivative"
    stdp_orientation: str = "legacy"
    # Kept enabled by default for legacy-compatible online episodic encoding.
    # Native recall experiments may explicitly seal this path after cutoff.
    hippocampal_encoding_enabled: bool = True

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        if self.dim <= 0:
            raise ValueError("runtime dimension must be positive")
        for name in (
            "neuronwise_active_threshold",
            "neuronwise_bit_lower_threshold",
            "neuronwise_bit_upper_threshold",
        ):
            setattr(self, name, self._normalize_neuronwise_threshold(name))
        self.competition_indices = self._normalize_competition_indices()
        self.competition_input_indices = self._normalize_competition_input_indices()
        self.validate_local_competition()
        if self.has_neuronwise_bit_threshold:
            self.effective_bit_thresholds()
        self.active_ratio = min(max(float(self.active_ratio), 0.0), 1.0)
        self.force_all_active_selection = bool(self.force_all_active_selection)
        self.memory_topk = max(1, int(self.memory_topk))
        self.memory_capacity = max(1, int(self.memory_capacity))
        self.f1_pull_strength = min(max(float(self.f1_pull_strength), 0.0), 1.0)
        self.f1_ema_alpha = min(max(float(self.f1_ema_alpha), 0.0), 1.0)
        self.f1_min_ratio = min(max(float(self.f1_min_ratio), 0.0), 1.0)
        self.f1_max_ratio = min(max(float(self.f1_max_ratio), self.f1_min_ratio), 1.0)
        self.stdp_interval = max(1, int(self.stdp_interval))
        self.stdp_apply_interval = max(self.stdp_interval, int(self.stdp_apply_interval))
        self.stdp_lr = max(float(self.stdp_lr), 0.0)
        self.stdp_density = min(max(float(self.stdp_density), 0.0), 1.0)
        self.stdp_gate_threshold = max(float(self.stdp_gate_threshold), 0.0)
        self.stdp_spike_threshold = max(float(self.stdp_spike_threshold), 0.0)
        if self.stdp_gate_mode not in {"critic_derivative", "external_signed"}:
            raise ValueError("stdp_gate_mode must be critic_derivative or external_signed")
        if self.stdp_orientation not in {"legacy", "causal"}:
            raise ValueError("stdp_orientation must be legacy or causal")
        self.hippocampal_encoding_enabled = bool(self.hippocampal_encoding_enabled)

    def _normalize_competition_indices(self) -> tuple[int, ...] | None:
        value = self.competition_indices
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError("competition_indices must be a list or tuple")
        try:
            normalized = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise TypeError("competition_indices entries must be integers") from exc
        if len(normalized) < 2:
            raise ValueError("competition_indices must contain at least two neurons")
        if len(set(normalized)) != len(normalized):
            raise ValueError("competition_indices must be distinct")
        if any(index < 0 or index >= self.dim for index in normalized):
            raise ValueError("competition_indices entries must be within runtime dimension")
        return normalized

    def _normalize_competition_input_indices(self) -> tuple[int, ...] | None:
        value = self.competition_input_indices
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError("competition_input_indices must be a list or tuple")
        try:
            normalized = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise TypeError("competition_input_indices entries must be integers") from exc
        if not normalized:
            raise ValueError("competition_input_indices must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("competition_input_indices must be distinct")
        if any(index < 0 or index >= self.dim for index in normalized):
            raise ValueError("competition_input_indices entries must be within runtime dimension")
        return normalized

    @property
    def has_local_competition(self) -> bool:
        return self.competition_indices is not None

    def validate_local_competition(self) -> None:
        self.competition_input_indices = self._normalize_competition_input_indices()
        values = {
            "competition_lateral_gain": self.competition_lateral_gain,
            "competition_homeostasis_gain": self.competition_homeostasis_gain,
            "competition_homeostasis_rate": self.competition_homeostasis_rate,
            "competition_homeostasis_decay": self.competition_homeostasis_decay,
            "competition_novelty_decay": self.competition_novelty_decay,
            "competition_epsilon": self.competition_epsilon,
            "competition_jitter_sigma": self.competition_jitter_sigma,
        }
        try:
            normalized = {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise TypeError("local competition parameters must be real numbers") from exc
        if not all(math.isfinite(value) for value in normalized.values()):
            raise ValueError("local competition parameters must be finite")
        for name, value in normalized.items():
            setattr(self, name, value)
        self.competition_delay_ticks = int(self.competition_delay_ticks)
        if self.competition_lateral_gain < 0.0 or self.competition_homeostasis_gain < 0.0:
            raise ValueError("local competition gains must be nonnegative")
        if not 0.0 <= self.competition_homeostasis_rate <= 1.0:
            raise ValueError("competition_homeostasis_rate must be in [0, 1]")
        if not 0.0 <= self.competition_homeostasis_decay <= 1.0:
            raise ValueError("competition_homeostasis_decay must be in [0, 1]")
        if not 0.0 <= self.competition_novelty_decay < 1.0:
            raise ValueError("competition_novelty_decay must be in [0, 1)")
        if self.competition_delay_ticks < 1:
            raise ValueError("competition_delay_ticks must be positive")
        if self.competition_epsilon <= 0.0:
            raise ValueError("competition_epsilon must be positive")
        if not 0.0 <= self.competition_jitter_sigma < 1.0:
            raise ValueError("competition_jitter_sigma must be in [0, 1)")
        if isinstance(self.competition_jitter_seed, bool) or not isinstance(
            self.competition_jitter_seed, (int, np.integer)
        ):
            raise TypeError("competition_jitter_seed must be an integer")
        self.competition_jitter_seed = int(self.competition_jitter_seed)
        if not isinstance(self.competition_k_from_delayed_input, (bool, np.bool_)):
            raise TypeError("competition_k_from_delayed_input must be boolean")
        self.competition_k_from_delayed_input = bool(
            self.competition_k_from_delayed_input
        )
        if not isinstance(self.competition_factorize_delayed_input, (bool, np.bool_)):
            raise TypeError("competition_factorize_delayed_input must be boolean")
        self.competition_factorize_delayed_input = bool(
            self.competition_factorize_delayed_input
        )
        if (
            self.competition_k_from_delayed_input
            and self.competition_factorize_delayed_input
        ):
            raise ValueError("adaptive and factorized competition are mutually exclusive")
        if self.competition_k_from_delayed_input or self.competition_factorize_delayed_input:
            if not self.has_local_competition or self.competition_input_indices is None:
                raise ValueError(
                    "input-aware competition requires competition and input indices"
                )
            if not self.axon_delay:
                raise ValueError("input-aware competition requires axon_delay=True")
            if self.competition_lateral_gain != 1.0:
                raise ValueError(
                    "strict input-aware competition requires competition_lateral_gain=1"
                )
            if (
                self.competition_factorize_delayed_input
                and self.competition_jitter_sigma != 0.0
            ):
                raise ValueError("factorized competition requires jitter_sigma=0")
        elif self.competition_input_indices is not None:
            raise ValueError(
                "competition_input_indices require an input-aware competition mode"
            )
        if not self.has_local_competition and any(
            value != 0.0 for value in (
                self.competition_lateral_gain,
                self.competition_homeostasis_gain,
                self.competition_homeostasis_rate,
                self.competition_homeostasis_decay,
                self.competition_jitter_sigma,
            )
        ):
            raise ValueError("competition_indices are required when competition is active")

    def _normalize_neuronwise_threshold(self, name: str) -> tuple[float, ...] | None:
        value = getattr(self, name)
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple")
        try:
            normalized = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} entries must be real numbers") from exc
        if len(normalized) != self.dim:
            raise ValueError(f"{name} length must match runtime dimension")
        if not all(math.isfinite(item) for item in normalized):
            raise ValueError(f"{name} entries must be finite")
        return normalized

    @property
    def has_neuronwise_bit_threshold(self) -> bool:
        return (
            self.neuronwise_bit_lower_threshold is not None
            or self.neuronwise_bit_upper_threshold is not None
        )

    def effective_active_thresholds(self) -> tuple[float, ...]:
        values = self._normalize_neuronwise_threshold("neuronwise_active_threshold")
        if values is not None:
            return values
        return (float(self.active_threshold),) * self.dim

    def effective_bit_thresholds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lower_vector = self._normalize_neuronwise_threshold(
            "neuronwise_bit_lower_threshold"
        )
        upper_vector = self._normalize_neuronwise_threshold(
            "neuronwise_bit_upper_threshold"
        )
        if lower_vector is None and upper_vector is None:
            return (
                (float(self.bit_lower_threshold),) * self.dim,
                (float(self.bit_upper_threshold),) * self.dim,
            )
        lower = (
            lower_vector
            if lower_vector is not None
            else (float(self.bit_lower_threshold),) * self.dim
        )
        upper = (
            upper_vector
            if upper_vector is not None
            else (float(self.bit_upper_threshold),) * self.dim
        )
        if not all(math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("effective neuronwise bit thresholds must be finite")
        if any(low >= high for low, high in zip(lower, upper)):
            raise ValueError(
                "effective neuronwise bit lower thresholds must be below upper thresholds"
            )
        return lower, upper

    def energy_budget(self, mode: RuntimeMode) -> int:
        base = max(1, int(round(self.dim * self.active_ratio)))
        if mode is RuntimeMode.NREM:
            return max(1, int(round(base * 0.5)))
        if mode is RuntimeMode.REM:
            return max(1, int(round(base * 0.75)))
        return base

    def activation_decay(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.18,
            RuntimeMode.NREM: 0.34,
            RuntimeMode.REM: 0.22,
        }[mode]

    def activation_gain(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.82,
            RuntimeMode.NREM: 0.52,
            RuntimeMode.REM: 0.68,
        }[mode]

    def refractory_decay(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.12,
            RuntimeMode.NREM: 0.26,
            RuntimeMode.REM: 0.18,
        }[mode]

    def refractory_gain(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.24,
            RuntimeMode.NREM: 0.12,
            RuntimeMode.REM: 0.18,
        }[mode]

    def replay_mix(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.08,
            RuntimeMode.NREM: self.replay_gain,
            RuntimeMode.REM: self.replay_gain * 1.25,
        }[mode]


@dataclass
class RuntimeStep:
    """High-level runtime summary returned to the Python control plane."""
    step: int
    mode: RuntimeMode
    energy: float
    active_modules: int
    replay_norm: float
    sleep_pressure: float
    arousal: float
    lifecycle_counts: Dict[str, int]
    stdp_gate: float = 0.0
    stdp_updates: int = 0


@dataclass
class BrainRuntimeSnapshot:
    """Serializable runtime state used for warm snapshots / restore."""
    config: BrainRuntimeConfig
    weight: torch.Tensor
    activation: torch.Tensor
    refractory: torch.Tensor
    memory_trace: torch.Tensor
    adaptation: torch.Tensor
    stp_u: torch.Tensor
    stp_x: torch.Tensor
    bitfield: torch.Tensor
    goal: torch.Tensor
    lifecycle: torch.Tensor
    inactive_steps: torch.Tensor
    mode: RuntimeMode
    sleep_pressure: float
    arousal: float
    step: int
    hippocampus: dict[str, object]
    mode_occupancy: Dict[str, int] = field(default_factory=dict)
    active_ratio_ema: float = -1.0
    stdp_tracker: dict[str, torch.Tensor] | None = None
    stdp_prev_critic_score: float = 0.0
    stdp_updates: int = 0
    circadian_phase: float = 0.0
    circadian_value: float = 0.0
    nrem_cycle_count: int = 0
    delay_buffer: torch.Tensor | None = None
    delay_idx: int = 0
    competition_homeostasis: torch.Tensor | None = None
    competition_usage_buffer: torch.Tensor | None = None
    competition_usage_idx: int = 0
    competition_packet_envelope: torch.Tensor | None = None
    brainwave_history: tuple[float, ...] = ()
    last_stdp_gate: float = 0.0
    stdp_pending_learning_signal: float = 0.0


@dataclass
class HippocampusMemory:
    """Minimal fast-memory subsystem: encode, recall, replay priority."""
    dim: int
    capacity: int = 32
    device: str | torch.device = "cpu"
    _keys: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _values: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _priority: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        self.capacity = max(1, int(self.capacity))
        self.device = torch.device(self.device)

    def __len__(self) -> int:
        return len(self._priority)

    def encode(
        self,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        priority: float = 1.0,
    ) -> None:
        key = _normalize(key).to(self.device)
        value = key if value is None else value.detach().float().to(self.device)
        priority = float(max(priority, 1e-6))
        if len(self._priority) >= self.capacity:
            drop_idx = min(range(len(self._priority)), key=self._priority.__getitem__)
            self._keys.pop(drop_idx)
            self._values.pop(drop_idx)
            self._priority.pop(drop_idx)
        self._keys.append(key)
        self._values.append(value)
        self._priority.append(priority)

    def decay_priorities(self, steps: int = 1) -> None:
        """Exponential priority decay: P *= exp(-dt/tau_forget). (15_Equations D)"""
        if not self._priority:
            return
        factor = math.exp(-steps / FORGET_TAU)
        self._priority = [p * factor for p in self._priority]

    def recall(self, cue: torch.Tensor, *, topk: int = 4) -> torch.Tensor:
        if not self._keys:
            return torch.zeros(self.dim, device=self.device)
        cue = _normalize(cue).to(self.device)
        keys = torch.stack(self._keys, dim=0)
        values = torch.stack(self._values, dim=0)
        priority = torch.tensor(self._priority, dtype=torch.float32, device=self.device)
        similarity = keys @ cue
        above_threshold = similarity >= RECALL_SIMILARITY_THRESHOLD
        if not above_threshold.any():
            return torch.zeros(self.dim, device=self.device)
        score = similarity + priority.log()
        score = score.masked_fill(~above_threshold, float("-inf"))
        k = min(max(int(topk), 1), int(above_threshold.sum().item()))
        top_score, top_idx = torch.topk(score, k=k)
        weights = torch.softmax(top_score, dim=0)
        return torch.sum(values[top_idx] * weights.unsqueeze(1), dim=0)

    def replay(self, mode: RuntimeMode) -> torch.Tensor:
        if not self._keys:
            return torch.zeros(self.dim, device=self.device)
        k = 1 if mode is RuntimeMode.NREM else min(3, len(self._keys))
        priority = torch.tensor(self._priority, dtype=torch.float32, device=self.device)
        top_idx = torch.topk(priority, k=k).indices
        values = torch.stack(self._values, dim=0)[top_idx]
        weights = torch.softmax(priority[top_idx], dim=0)
        return torch.sum(values * weights.unsqueeze(1), dim=0)

    def state_dict(self) -> dict[str, object]:
        keys = torch.stack(self._keys, dim=0).cpu() if self._keys else torch.empty((0, self.dim))
        values = torch.stack(self._values, dim=0).cpu() if self._values else torch.empty((0, self.dim))
        return {
            "dim": self.dim,
            "capacity": self.capacity,
            "keys": keys,
            "values": values,
            "priority": list(self._priority),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        device: str | torch.device = "cpu",
    ) -> "HippocampusMemory":
        mem = cls(int(state["dim"]), capacity=int(state["capacity"]), device=device)
        keys = state.get("keys", torch.empty((0, mem.dim)))
        values = state.get("values", torch.empty((0, mem.dim)))
        priority = state.get("priority", [])
        if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
            for idx, score in enumerate(priority):
                mem._keys.append(keys[idx].to(mem.device).float())
                mem._values.append(values[idx].to(mem.device).float())
                mem._priority.append(float(score))
        return mem


class BrainRuntime:
    """Reference runtime stack.

    Layering:
    - kernel/coupling: sparse recurrent update over `weight`
    - mode update: `RuntimeMode`
    - hippocampus/replay: `HippocampusMemory`
    - global summary: `RuntimeStep` and `BrainRuntimeSnapshot`
    """
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        config: BrainRuntimeConfig,
        backend: str = "auto",
        device: str | torch.device | None = None,
    ) -> None:
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError("weight must be a square matrix")
        if weight.shape[0] != config.dim:
            raise ValueError("weight dimension must match BrainRuntimeConfig.dim")

        self.config = config
        self.device = torch.device(device) if device is not None else weight.device
        self.backend = backend
        if self.backend == "rust" and self.config.axon_delay:
            raise ValueError("the Rust brain kernel does not support axon_delay=True")
        if self.backend == "rust" and self.config.has_neuronwise_bit_threshold:
            self.config.effective_bit_thresholds()
            raise ValueError(
                "the Rust brain kernel does not support neuronwise bit thresholds"
            )
        if self.backend == "rust" and self.config.has_local_competition:
            raise ValueError("the Rust brain kernel does not support local competition")
        self.weight = weight.detach().float().to(self.device)
        pack_backend = "torch" if self.backend == "cuda" else self.backend
        values, col_idx, row_ptr = pack_sparse(
            self.weight.detach().cpu(),
            zero_tol=self.config.zero_tol,
            backend=pack_backend,
        )
        self.values = values.to(self.device)
        self.col_idx = col_idx.to(self.device)
        self.row_ptr = row_ptr.to(self.device)
        self.sparse_weight = checked_sparse_csr_tensor(
            self.row_ptr.to(torch.int64),
            self.col_idx.to(torch.int64),
            self.values,
            size=self.weight.shape,
            device=self.device,
            dtype=self.weight.dtype,
        )

        self.activation = torch.zeros(self.config.dim, device=self.device)
        self.refractory = torch.zeros(self.config.dim, device=self.device)
        self.memory_trace = torch.zeros(self.config.dim, device=self.device)
        self.adaptation = torch.zeros(self.config.dim, device=self.device)
        self.stp_u = torch.full((self.config.dim,), 0.5, device=self.device)
        self.stp_x = torch.ones(self.config.dim, device=self.device)
        self.bitfield = torch.zeros(self.config.dim, dtype=torch.uint8, device=self.device)
        self.goal = torch.zeros(self.config.dim, device=self.device)
        self.lifecycle = torch.full(
            (self.config.dim,),
            _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT],
            dtype=torch.int64,
            device=self.device,
        )
        self.inactive_steps = torch.zeros(self.config.dim, dtype=torch.int64, device=self.device)
        self.mode = RuntimeMode.WAKE
        self.sleep_pressure = 0.0
        self.arousal = 0.0
        self.step_index = 0
        self.circadian_phase = 0.0
        self._circadian_value = CIRCADIAN_BASE + CIRCADIAN_AMP
        self.nrem_cycle_count = 0
        self.mode_occupancy: Dict[str, int] = {
            RuntimeMode.WAKE.value: 0,
            RuntimeMode.NREM.value: 0,
            RuntimeMode.REM.value: 0,
        }
        self.active_ratio_ema: float = float(self.config.active_ratio)

        # Dale's Law: E:I = 80:20 sign mask
        n_exc = int(self.config.dim * DALE_EI_RATIO)
        self.dale_sign = torch.ones(self.config.dim, device=self.device)
        self.dale_sign[n_exc:] = -DALE_INH_GAIN
        if self.config.dale_law:
            self.weight = self.weight.abs() * self.dale_sign.unsqueeze(1)
            self._rebuild_sparse()

        # Axon delay buffer: ring of source-qualified presynaptic packets.
        # Each packet freezes STP and lifecycle eligibility at emission, so an
        # already emitted signal is neither erased nor created by source state
        # changes while it is in flight.
        if self.config.axon_delay:
            self._delay_buffer = torch.zeros(
                self.config.max_axon_delay, self.config.dim, device=self.device
            )
            self._delay_idx = 0
        else:
            self._delay_buffer = None
            self._delay_idx = 0

        self._competition_signature = (
            tuple(self.config.competition_indices or ()),
            int(self.config.competition_delay_ticks),
            float(self.config.competition_jitter_sigma),
            int(self.config.competition_jitter_seed),
            tuple(self.config.competition_input_indices or ()),
            bool(self.config.competition_k_from_delayed_input),
            bool(self.config.competition_factorize_delayed_input),
        )
        if self.config.has_local_competition:
            self._competition_indices = torch.tensor(
                self.config.competition_indices,
                dtype=torch.long,
                device=self.device,
            )
            self._competition_input_indices = (
                torch.tensor(
                    self.config.competition_input_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                if self.config.competition_input_indices is not None
                else None
            )
            self.competition_homeostasis = torch.zeros(
                self.config.dim, device=self.device,
            )
            self._competition_usage_buffer = torch.zeros(
                self.config.competition_delay_ticks,
                self.config.dim,
                device=self.device,
            )
            self._competition_usage_idx = 0
            self.competition_packet_envelope = torch.zeros((), device=self.device)
        else:
            self._competition_indices = None
            self._competition_input_indices = None
            self.competition_homeostasis = None
            self._competition_usage_buffer = None
            self._competition_usage_idx = 0
            self.competition_packet_envelope = None

        # Brainwave history for FFT
        self._brainwave_history: list[float] = []
        self._brainwave_max_len = 1024

        self.hippocampus = HippocampusMemory(
            self.config.dim,
            capacity=self.config.memory_capacity,
            device=self.device,
        )
        self.stdp_tracker = None
        if self.config.stdp_enabled:
            self.stdp_tracker = EligibilityTracker(
                STDPConfig(
                    dim=self.config.dim,
                    spike_threshold=self.config.stdp_spike_threshold,
                    lr=self.config.stdp_lr,
                    orientation=self.config.stdp_orientation,
                ),
                device=self.device,
            )
        self._stdp_prev_critic_score = 0.0
        self._stdp_updates = 0
        self._last_stdp_gate = 0.0
        self._stdp_pending_learning_signal = 0.0

    def _rebuild_sparse(self) -> None:
        """Rebuild CSR sparse weight from dense weight."""
        pack_backend = "torch" if self.backend == "cuda" else self.backend
        values, col_idx, row_ptr = pack_sparse(
            self.weight.detach().cpu(),
            zero_tol=self.config.zero_tol,
            backend=pack_backend,
        )
        self.values = values.to(self.device)
        self.col_idx = col_idx.to(self.device)
        self.row_ptr = row_ptr.to(self.device)
        self.sparse_weight = checked_sparse_csr_tensor(
            self.row_ptr.to(torch.int64),
            self.col_idx.to(torch.int64),
            self.values,
            size=self.weight.shape,
            device=self.device,
            dtype=self.weight.dtype,
        )

    def install_bounded_recurrent_delta(self, delta: torch.Tensor, *, max_frobenius_norm: float) -> float:
        """Opt-in bounded recurrent write and CSR rebuild for controlled experiments.

        The default runtime never calls this method.  It provides a single
        observable mutation boundary for experiments that must write the real
        recurrent matrix rather than retaining a side association matrix.
        """
        candidate = delta.detach().float().to(self.device)
        if candidate.shape != self.weight.shape or not torch.isfinite(candidate).all():
            raise ValueError("delta must be finite and match recurrent weight shape")
        bound = float(max_frobenius_norm)
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("max_frobenius_norm must be finite and positive")
        norm = float(candidate.norm().item())
        if norm > bound:
            candidate = candidate * (bound / norm)
        self.weight = self.weight + candidate
        if self.config.dale_law:
            self.weight = self.weight.abs() * self.dale_sign.unsqueeze(1)
        self._rebuild_sparse()
        return float(candidate.norm().item())

    def _apply_dale_sign(self) -> None:
        if not self.config.dale_law:
            return
        self.weight = self.weight.abs() * self.dale_sign.unsqueeze(1)

    def _apply_runtime_stdp(
        self,
        active_count: int,
        energy: float,
        critic_score: float | None = None,
        learning_signal: float | None = None,
    ) -> float:
        """Optional F14 closed-loop plasticity over the runtime weight matrix.

        The learning gate (F.14.2) is ``g = alpha_g * d(c_bar)/dt + (1-alpha_g) *
        bootstrap_dev``. ``c_bar`` is the critic signal. When a Layer-F agent
        supplies its own critic via ``critic_score`` it drives the gate; otherwise
        the runtime falls back to internal ``energy`` as a critic proxy so that
        standalone (agent-less) runtimes still self-organize.
        """
        if self.stdp_tracker is None:
            self._last_stdp_gate = 0.0
            return 0.0

        if self.config.stdp_gate_mode == "external_signed" and learning_signal is not None:
            if not math.isfinite(float(learning_signal)):
                raise ValueError("learning_signal must be finite")
            self._stdp_pending_learning_signal += float(learning_signal)

        tick = self.step_index + 1
        if tick % self.config.stdp_interval != 0:
            self._last_stdp_gate = 0.0
            return 0.0

        self.stdp_tracker.update(self.activation)
        if tick % self.config.stdp_apply_interval != 0:
            self._last_stdp_gate = 0.0
            return 0.0

        gate_drive = float(energy if critic_score is None else critic_score)
        active_ratio = float(active_count) / float(max(self.config.dim, 1))
        legacy_gate = compute_learning_gate(
            critic_score=gate_drive,
            prev_critic_score=self._stdp_prev_critic_score,
            active_ratio=active_ratio,
            alpha_g=self.stdp_tracker.config.alpha_g,
        )
        # Keep the legacy derivative state current even in the experimental
        # mode, so switching modes cannot expose a stale critic difference.
        self._stdp_prev_critic_score = gate_drive
        if self.config.stdp_gate_mode == "external_signed":
            gate = self._stdp_pending_learning_signal
            self._stdp_pending_learning_signal = 0.0
        else:
            gate = legacy_gate
        self._last_stdp_gate = float(gate)

        if abs(gate) <= self.config.stdp_gate_threshold:
            return float(gate)

        self.weight = apply_stdp_update(
            self.weight,
            self.stdp_tracker,
            gate,
            lr=self.config.stdp_lr,
            density=self.config.stdp_density,
        ).to(self.device)
        self.weight.fill_diagonal_(0.0)
        self._apply_dale_sign()
        self._rebuild_sparse()
        self.stdp_tracker.reset()
        self._stdp_updates += 1
        return float(gate)

    def brainwave_observable(self) -> dict[str, float]:
        """Compute global brainwave and band powers via FFT (Layer B / F.21)."""
        psi = float(self.activation.abs().mean().item())
        self._brainwave_history.append(psi)
        if len(self._brainwave_history) > self._brainwave_max_len:
            self._brainwave_history = self._brainwave_history[-self._brainwave_max_len:]
        result: dict[str, float] = {"psi_global": psi}
        if len(self._brainwave_history) < 8:
            return result
        sig = torch.tensor(self._brainwave_history, dtype=torch.float32)
        fft_vals = torch.fft.rfft(sig - sig.mean())
        power = (fft_vals.abs() ** 2) / len(sig)
        fs = 1000.0  # 1 step = 1ms
        freqs = torch.fft.rfftfreq(len(sig), d=1.0 / fs)
        for name, (lo, hi) in [
            ("delta", BAND_DELTA), ("theta", BAND_THETA),
            ("alpha", BAND_ALPHA), ("beta", BAND_BETA), ("gamma", BAND_GAMMA),
        ]:
            mask = (freqs >= lo) & (freqs < hi)
            result[name] = float(power[mask].sum().item()) if mask.any() else 0.0
        return result

    def energy_full(self) -> float:
        """Full energy E({a_i}) per 15_Equations.md B.3."""
        coupling = -0.5 * torch.dot(self.activation, self._matvec(self.activation))
        local = -(self.refractory * self.activation).sum()
        adapt = -ADAPTATION_COUPLING * (self.adaptation * self.activation).sum()
        return float((coupling + local + adapt).item())

    def compute_self_state(self) -> dict[str, float]:
        """Layer E: Self_t = S(G_t) -- global self-state summary."""
        active_frac = float(self.active_mask().float().mean().item())
        target = torch.tensor([ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO])
        lc = self.lifecycle_counts()
        total = max(sum(lc.values()), 1)
        current = torch.tensor([
            lc.get("ACTIVE", 0) / total,
            (lc.get("IDLE", 0) + lc.get("SLEEPING", 0)) / total,
            lc.get("DORMANT", 0) / total,
        ])
        bootstrap_deviation = float((current - target).norm().item())
        return {
            "active_fraction": active_frac,
            "bootstrap_deviation": bootstrap_deviation,
            "sleep_pressure": self.sleep_pressure,
            "arousal": self.arousal,
            "mode": self.mode.value,
            "energy": self.energy_full(),
            "consciousness_depth": 0.0,  # filled by agent layer
        }

    def set_goal(self, goal: torch.Tensor | None) -> None:
        if goal is None:
            self.goal.zero_()
            return
        goal = goal.detach().float().to(self.device)
        if goal.numel() != self.config.dim:
            raise ValueError("goal size must match runtime dimension")
        self.goal = goal.view(self.config.dim)

    def active_mask(self) -> torch.Tensor:
        return self.lifecycle == _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]

    def lifecycle_counts(self) -> Dict[str, int]:
        counts = {}
        for code, lifecycle in _CODE_TO_LIFECYCLE.items():
            counts[lifecycle.value] = int((self.lifecycle == code).sum().item())
        return counts

    def mode_occupancy_kl(self, eps: float = 1e-9) -> Dict[str, float]:
        """F3 ergodic gate (docs/7_AGI/12_Equation.md A.3).

        Reports the empirical mode occupancy measure pi_brain on the 3-simplex
        and its KL divergence to the runtime target tuple
        p* = (BACKGROUND_RATIO, STRUCT_RATIO, ACTIVE_RATIO), a frozen
        operational axiom (provenance: pstar-br8-adjudication-20260823); it
        carries no cosmological interpretation.

        Mapping: WAKE -> BACKGROUND_RATIO, NREM -> STRUCT_RATIO,
        REM -> ACTIVE_RATIO.
        """
        total = sum(self.mode_occupancy.values())
        if total <= 0:
            return {
                "samples": 0,
                "pi_wake": 0.0,
                "pi_nrem": 0.0,
                "pi_rem": 0.0,
                "kl_to_p_star": float("nan"),
            }
        pi_wake = self.mode_occupancy.get(RuntimeMode.WAKE.value, 0) / total
        pi_nrem = self.mode_occupancy.get(RuntimeMode.NREM.value, 0) / total
        pi_rem = self.mode_occupancy.get(RuntimeMode.REM.value, 0) / total
        pi = (pi_wake, pi_nrem, pi_rem)
        p_star = (BACKGROUND_RATIO, STRUCT_RATIO, ACTIVE_RATIO)
        kl = 0.0
        for p_i, q_i in zip(pi, p_star):
            if p_i > eps:
                kl += p_i * (np.log(p_i + eps) - np.log(q_i + eps))
        return {
            "samples": total,
            "pi_wake": pi_wake,
            "pi_nrem": pi_nrem,
            "pi_rem": pi_rem,
            "kl_to_p_star": float(kl),
        }

    def reset_mode_occupancy(self) -> None:
        """Zero the F3 mode occupancy counter (e.g. between sleep cycles)."""
        for key in self.mode_occupancy:
            self.mode_occupancy[key] = 0

    def _f1_effective_budget(self, mode: RuntimeMode) -> int:
        """Self-measured energy budget (gate F1, docs/7_AGI/12_Equation.md A.2 #2).

        Static fallback: config.energy_budget(mode). When f1_self_measure is
        on, the empirical EMA p_emp is convexly pulled toward ACTIVE_RATIO:
            r_eff = clip(beta * ACTIVE_RATIO + (1 - beta) * ema, lo, hi).
        Mode multipliers are preserved (1.0/0.5/0.75 for WAKE/NREM/REM).
        """
        if not self.config.f1_self_measure:
            return self.config.energy_budget(mode)
        beta = self.config.f1_pull_strength
        r_eff = beta * ACTIVE_RATIO + (1.0 - beta) * self.active_ratio_ema
        r_eff = min(max(r_eff, self.config.f1_min_ratio), self.config.f1_max_ratio)
        base = max(1, int(round(self.config.dim * r_eff)))
        if mode is RuntimeMode.NREM:
            return max(1, int(round(base * 0.5)))
        if mode is RuntimeMode.REM:
            return max(1, int(round(base * 0.75)))
        return base

    def _f1_update_ema(self, active_count: int) -> None:
        if not self.config.f1_self_measure:
            return
        p_emp = float(active_count) / float(self.config.dim)
        alpha = self.config.f1_ema_alpha
        self.active_ratio_ema = (1.0 - alpha) * self.active_ratio_ema + alpha * p_emp

    def bridge_gate_report(self) -> Dict[str, Dict[str, float]]:
        """AGI bridge gate aggregator (docs/7_AGI/12_Equation.md appendix A).

        Returns whatever measurements are currently available. Gate keys
        always exist; values are scalar reports or empty dicts when the
        underlying signal is not yet measurable.

        Current coverage:
        - F1 (self-organization, A.2 #2): empirical p_emp EMA vs ACTIVE_RATIO.
          Always reported; deviation is meaningful even when feedback is off.
        - F2 (ISS ball, A.1): exposed only when relax has been driven by a
          higher-level engine (see reality_stone.clarus.ce_ops.relax hist['iss']).
          BrainRuntime itself does not run the gradient relax, so F2 here is
          left empty by design.
        - F3 (ergodic KL, A.3): wraps mode_occupancy_kl().
        - F4 (PCI regression, A.4) is an experiment-level gate.
        """
        return {
            "F1_self_organization": {
                "active_ratio_ema": float(self.active_ratio_ema),
                "active_ratio_target": float(ACTIVE_RATIO),
                "deviation": float(self.active_ratio_ema - ACTIVE_RATIO),
                "self_measure_on": float(self.config.f1_self_measure),
            },
            "F2_iss_ball": {},
            "F3_ergodic_kl": self.mode_occupancy_kl(),
            "F4_pci_regression": {},
        }

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.sparse_weight, x.unsqueeze(1)).squeeze(1)

    def _assert_local_competition_config(self) -> None:
        self.config.validate_local_competition()
        signature = (
            tuple(self.config.competition_indices or ()),
            int(self.config.competition_delay_ticks),
            float(self.config.competition_jitter_sigma),
            int(self.config.competition_jitter_seed),
            tuple(self.config.competition_input_indices or ()),
            bool(self.config.competition_k_from_delayed_input),
            bool(self.config.competition_factorize_delayed_input),
        )
        if signature != self._competition_signature:
            raise ValueError(
                "competition indices, delay, and jitter are structural "
                "runtime configuration and cannot be mutated after construction"
            )

    def _advance_local_competition(self) -> int | None:
        if self._competition_usage_buffer is None:
            return None
        self._assert_local_competition_config()
        slot = self._competition_usage_idx % self.config.competition_delay_ticks
        delayed_usage = self._competition_usage_buffer[slot].detach().clone()
        self._competition_usage_buffer[slot].zero_()
        assert self.competition_homeostasis is not None
        self.competition_homeostasis = (
            (1.0 - self.config.competition_homeostasis_decay)
            * self.competition_homeostasis
            + self.config.competition_homeostasis_rate * delayed_usage
        ).clamp(0.0, 1.0)
        return int(slot)

    def _apply_local_competition(
        self,
        recurrent: torch.Tensor,
        *,
        input_packet_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._competition_indices is None:
            return recurrent, None
        assert self.competition_homeostasis is not None
        assert self.competition_packet_envelope is not None
        indices = self._competition_indices
        raw_positive = recurrent[indices].clamp_min(0.0)
        packet_mass = raw_positive.sum()
        envelope = self.competition_packet_envelope
        novelty = (packet_mass - envelope).clamp_min(0.0) / (
            self.config.competition_epsilon + packet_mass
        )
        self.competition_packet_envelope = torch.maximum(
            self.config.competition_novelty_decay * envelope,
            packet_mass.detach(),
        )
        if (
            self.config.competition_jitter_sigma > 0.0
            and float(packet_mass.item()) > self.config.competition_epsilon
        ):
            generator = torch.Generator(device=raw_positive.device)
            seed = (
                int(self.config.competition_jitter_seed)
                + 104_729 * int(self.step_index)
            ) & 0x7FFF_FFFF_FFFF_FFFF
            generator.manual_seed(seed)
            normal = torch.randn(
                raw_positive.shape,
                generator=generator,
                device=raw_positive.device,
                dtype=raw_positive.dtype,
            )
            sigma = self.config.competition_jitter_sigma
            raw_positive = raw_positive * (1.0 + sigma * torch.tanh(normal))
        attenuated = raw_positive * torch.exp(
            -self.config.competition_homeostasis_gain
            * self.competition_homeostasis[indices]
        )
        if input_packet_count is None or int(input_packet_count) <= 1:
            # Preserve the legacy singleton branch byte-for-byte.
            peers = attenuated.unsqueeze(0).expand(indices.numel(), -1).clone()
            peers.fill_diagonal_(float("-inf"))
            strongest_peer = peers.max(dim=1).values
            competed = (
                attenuated
                - self.config.competition_lateral_gain * strongest_peer
            ).clamp_min(0.0)
        elif int(input_packet_count) == 2:
            # Strict two-route branch.  The third-largest value is the
            # threshold; a tie at the 2/3 boundary fails closed.
            ordered = torch.sort(attenuated, descending=True, stable=True).values
            if float((ordered[1] - ordered[2]).item()) <= 0.0:
                competed = torch.zeros_like(attenuated)
            else:
                competed = (attenuated - ordered[2]).clamp_min(0.0)
        else:
            # Capacity above two has not been admitted by the TR17 contract.
            competed = torch.zeros_like(attenuated)
        result = recurrent.clone()
        result[indices] = competed
        return result, novelty.detach()

    def _apply_factorized_local_competition(
        self,
        recurrent: torch.Tensor,
        delayed_pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select each arriving source contribution before summing routes."""
        if self._competition_indices is None or self._competition_input_indices is None:
            raise RuntimeError("factorized competition indices are missing")
        active_sources = self._competition_input_indices[
            delayed_pre[self._competition_input_indices].abs()
            > self.config.competition_epsilon
        ]
        if active_sources.numel() <= 1:
            return self._apply_local_competition(
                recurrent,
                input_packet_count=int(active_sources.numel()),
            )
        assert self.competition_homeostasis is not None
        assert self.competition_packet_envelope is not None
        indices = self._competition_indices
        raw_total = recurrent[indices].clamp_min(0.0)
        packet_mass = raw_total.sum()
        envelope = self.competition_packet_envelope
        novelty = (packet_mass - envelope).clamp_min(0.0) / (
            self.config.competition_epsilon + packet_mass
        )
        self.competition_packet_envelope = torch.maximum(
            self.config.competition_novelty_decay * envelope,
            packet_mass.detach(),
        )
        attenuation = torch.exp(
            -self.config.competition_homeostasis_gain
            * self.competition_homeostasis[indices]
        )
        selected_sum = torch.zeros_like(raw_total)
        for source_index in active_sources.tolist():
            contribution = (
                self.weight[indices, int(source_index)]
                * delayed_pre[int(source_index)]
            ).clamp_min(0.0)
            attenuated = contribution * attenuation
            peers = attenuated.unsqueeze(0).expand(indices.numel(), -1).clone()
            peers.fill_diagonal_(float("-inf"))
            selected_sum += (attenuated - peers.max(dim=1).values).clamp_min(0.0)
        result = recurrent.clone()
        result[indices] = selected_sum
        return result, novelty.detach()

    def _commit_local_competition_usage(
        self,
        activation: torch.Tensor,
        novelty: torch.Tensor | None,
        slot: int | None,
    ) -> None:
        if self._competition_usage_buffer is None:
            return
        if novelty is None or slot is None or self._competition_indices is None:
            raise RuntimeError("local competition usage commit is missing step state")
        positive = activation[self._competition_indices].clamp_min(0.0).square()
        normalized = positive / (
            self.config.competition_epsilon + positive.sum()
        )
        usage = torch.zeros_like(activation)
        usage[self._competition_indices] = novelty * normalized
        self._competition_usage_buffer[slot] = usage.detach()
        self._competition_usage_idx += 1

    def _select_active(self, salience: torch.Tensor, budget: int) -> torch.Tensor:
        if self.config.force_all_active_selection:
            return torch.ones_like(salience, dtype=torch.bool)
        budget = max(0, min(int(budget), salience.numel()))
        mask = torch.zeros_like(salience, dtype=torch.bool)
        if budget == 0:
            return mask
        active_threshold = torch.tensor(
            self.config.effective_active_thresholds(),
            dtype=salience.dtype,
            device=salience.device,
        )
        eligible = salience >= active_threshold
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return mask
        budget = min(budget, eligible_count)
        scored = salience.masked_fill(~eligible, float("-inf"))
        _, idx = torch.topk(scored, k=budget)
        mask[idx] = True
        return mask

    def _auto_mode(self, external_norm: float) -> RuntimeMode:
        if self.mode is RuntimeMode.WAKE:
            if self.sleep_pressure > 1.0 and external_norm < self.config.wake_threshold:
                return RuntimeMode.NREM
            return RuntimeMode.WAKE
        if self.mode is RuntimeMode.NREM:
            if external_norm > self.config.wake_threshold * 1.5:
                return RuntimeMode.WAKE
            if self.sleep_pressure < 0.45:
                return RuntimeMode.REM
            return RuntimeMode.NREM
        if external_norm > self.config.wake_threshold or self.sleep_pressure < 0.15:
            return RuntimeMode.WAKE
        return RuntimeMode.REM

    def _update_sleep_state(self, mode: RuntimeMode, active_count: int, external_norm: float) -> None:
        """Borbely 2-Process model with circadian (15_Equations.md C.2)."""
        self.arousal = float(external_norm)
        tau_w_inv = 1.0 / TAU_W_STEPS
        tau_s_inv = 1.0 / TAU_S_STEPS

        # Process C: circadian modulation
        self.circadian_phase += 1.0
        circadian = CIRCADIAN_BASE + CIRCADIAN_AMP * math.cos(
            2.0 * math.pi * self.circadian_phase / CIRCADIAN_PERIOD
        )

        # Process S: homeostatic pressure
        if mode is RuntimeMode.WAKE:
            self.sleep_pressure += (SLEEP_PRESSURE_MAX - self.sleep_pressure) * tau_w_inv
        elif mode is RuntimeMode.NREM:
            self.sleep_pressure -= self.sleep_pressure * tau_s_inv
            self.nrem_cycle_count += 1
        else:
            self.sleep_pressure -= self.sleep_pressure * tau_s_inv * REM_TAU_FACTOR
        self.sleep_pressure = float(max(0.0, min(self.sleep_pressure, SLEEP_PRESSURE_MAX)))
        self._circadian_value = circadian

    def nrem_target_length(self) -> float:
        """T_NREM(n) = T0 * alpha^n -- decreasing NREM length within a night."""
        base = TAU_S_STEPS * 2.0
        return base * (NREM_LENGTH_DECAY ** self.nrem_cycle_count)

    def _update_lifecycle(self, salience: torch.Tensor, active_mask: torch.Tensor) -> None:
        self.inactive_steps = torch.where(
            active_mask,
            torch.zeros_like(self.inactive_steps),
            self.inactive_steps + 1,
        )
        lifecycle = torch.full_like(
            self.lifecycle,
            _LIFECYCLE_TO_CODE[ModuleLifecycle.IDLE],
        )
        lifecycle[salience < self.config.idle_threshold] = _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT]
        lifecycle[
            (self.inactive_steps >= self.config.dormant_after)
            & (salience < self.config.idle_threshold)
        ] = _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT]
        lifecycle[self.inactive_steps >= self.config.sleeping_after] = _LIFECYCLE_TO_CODE[
            ModuleLifecycle.SLEEPING
        ]
        lifecycle[active_mask] = _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]
        self.lifecycle = lifecycle

    def _energy(self, recurrent: torch.Tensor, replay: torch.Tensor) -> float:
        coupling = 0.5 * torch.dot(self.activation, recurrent).abs()
        local = (self.refractory.mean()
                 + 0.25 * self.memory_trace.abs().mean()
                 + 0.10 * self.adaptation.abs().mean())
        replay_term = 0.1 * replay.abs().mean()
        total = coupling + local + replay_term
        return float(total.item())

    def _use_rust(self) -> bool:
        self._assert_local_competition_config()
        if self.config.has_local_competition:
            if self.backend == "rust":
                raise ValueError("the Rust brain kernel does not support local competition")
            return False
        if self.config.axon_delay:
            if self.backend == "rust":
                raise ValueError("the Rust brain kernel does not support axon_delay=True")
            return False
        if self.config.has_neuronwise_bit_threshold:
            self.config.effective_bit_thresholds()
            if self.backend == "rust":
                raise ValueError(
                    "the Rust brain kernel does not support neuronwise bit thresholds"
                )
            return False
        if not _HAS_RUST_KERNEL:
            if self.backend == "rust":
                raise RuntimeError(
                    "backend='rust' was requested but reality_stone.clarus._rust is not built; "
                    "use backend='torch' or build it with .codex/hooks/build-native.cmd"
                )
            return False
        if self.backend == "rust":
            return True
        if self.backend == "auto" and self.device.type == "cpu":
            return True
        return False

    def _step_rust(
        self,
        external: torch.Tensor,
        replay: torch.Tensor,
        mode: RuntimeMode,
    ) -> tuple[int, float]:
        """Delegate the cell-step hot path to the Rust kernel."""
        self._assert_local_competition_config()
        if self.config.has_local_competition:
            raise ValueError("the Rust brain kernel does not support local competition")
        if self.config.axon_delay:
            raise ValueError("the Rust brain kernel does not support axon_delay=True")
        if self.config.has_neuronwise_bit_threshold:
            self.config.effective_bit_thresholds()
            raise ValueError(
                "the Rust brain kernel does not support neuronwise bit thresholds"
            )
        budget = self.config.energy_budget(mode)
        mode_int = _MODE_TO_INT.get(mode.value, 0)
        act_np = self.activation.detach().cpu().numpy().astype(np.float32)
        ref_np = self.refractory.detach().cpu().numpy().astype(np.float32)
        mem_np = self.memory_trace.detach().cpu().numpy().astype(np.float32)
        adapt_np = self.adaptation.detach().cpu().numpy().astype(np.float32)
        su_np = self.stp_u.detach().cpu().numpy().astype(np.float32)
        sx_np = self.stp_x.detach().cpu().numpy().astype(np.float32)
        bit_np = self.bitfield.detach().cpu().numpy().astype(np.uint8)
        active_np = self.active_mask().detach().cpu().numpy().astype(np.uint8)
        ext_np = external.detach().cpu().numpy().astype(np.float32)
        goal_np = self.goal.detach().cpu().numpy().astype(np.float32)
        replay_np = replay.detach().cpu().numpy().astype(np.float32)
        noise_scale = {
            RuntimeMode.WAKE: 1.0,
            RuntimeMode.NREM: 0.3,
            RuntimeMode.REM: 0.7,
        }[mode]
        gen = torch.Generator(device=self.activation.device)
        gen.manual_seed(self.step_index * 31337 + 7)
        noise = self.config.noise_sigma * noise_scale * torch.randn(
            self.activation.shape,
            generator=gen,
            device=self.activation.device,
            dtype=self.activation.dtype,
        )
        noise_np = noise.detach().cpu().numpy().astype(np.float32)
        val_np = self.values.detach().cpu().numpy().astype(np.float32)
        col_np = self.col_idx.detach().cpu().numpy().astype(np.int32)
        row_np = self.row_ptr.detach().cpu().numpy().astype(np.int32)

        (new_act, new_ref, new_mem, new_adapt,
         new_su, new_sx, new_bit, active_count, energy) = _rust_brain_step(
            val_np, col_np, row_np,
            act_np, ref_np, mem_np, adapt_np, su_np, sx_np, bit_np,
            active_np, ext_np, goal_np, replay_np, noise_np,
            mode_int, budget,
            self.config.activation_decay(mode),
            self.config.activation_gain(mode),
            self.config.refractory_decay(mode),
            self.config.refractory_gain(mode),
            self.config.replay_mix(mode),
            self.config.refractory_scale,
            self.config.goal_gain,
            self.config.external_gain,
            self.config.bit_lower_threshold,
            self.config.bit_upper_threshold,
            STP_TAU_FAC_INV,
            STP_TAU_REC,
            STP_U_BASE,
            ADAPTATION_COUPLING,
            ADAPTATION_DECAY,
            MEMORY_TRACE_DECAY,
            ADAPTATION_CLAMP,
        )
        self.activation = torch.from_numpy(np.array(new_act, dtype=np.float32)).to(self.device)
        self.refractory = torch.from_numpy(np.array(new_ref, dtype=np.float32)).to(self.device)
        self.memory_trace = torch.from_numpy(np.array(new_mem, dtype=np.float32)).to(self.device)
        self.adaptation = torch.from_numpy(np.array(new_adapt, dtype=np.float32)).to(self.device)
        self.stp_u = torch.from_numpy(np.array(new_su, dtype=np.float32)).to(self.device)
        self.stp_x = torch.from_numpy(np.array(new_sx, dtype=np.float32)).to(self.device)
        self.bitfield = torch.from_numpy(np.array(new_bit, dtype=np.uint8)).to(self.device)
        return int(active_count), float(energy)

    def _compute_salience(
        self,
        activation: torch.Tensor,
        external: torch.Tensor,
        replay: torch.Tensor,
        refractory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute module salience for active selection (shared by step logic)."""
        return (
            activation.abs()
            + 0.35 * external.abs()
            + 0.25 * replay.abs()
            + 0.20 * self.goal.abs()
            - 0.15 * refractory
        )

    def _step_torch(
        self,
        external: torch.Tensor,
        replay: torch.Tensor,
        mode: RuntimeMode,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Pure-torch cell step (fallback path). Eq A.1--A.7, J.19--J.20.

        Returns (salience, recurrent, energy) to avoid recomputation in step().
        """
        prev_active = self.active_mask().float()
        competition_slot = self._advance_local_competition()

        spike = prev_active
        stp_u = self.stp_u + (-STP_TAU_FAC_INV * self.stp_u + STP_U_BASE * (1.0 - self.stp_u) * spike)
        stp_x = self.stp_x + (STP_TAU_REC * (1.0 - self.stp_x) - self.stp_u * self.stp_x * spike)
        stp_u = stp_u.clamp(0.0, 1.0)
        stp_x = stp_x.clamp(0.0, 1.0)

        pre = stp_u * stp_x * self.activation * prev_active

        # Axon delay: deliver the packet emitted exactly L calls earlier.
        # Read-before-write makes an L-slot ring a true L-call drive delay.
        input_packet_count = None
        if self._delay_buffer is not None:
            slot = self._delay_idx % self.config.max_axon_delay
            # Keep the delivered packet stable across the read-before-write
            # ring update below.  A view would alias the overwritten slot and
            # make provenance-aware competition inspect the newly emitted
            # packet instead of the packet that actually arrived this tick.
            delayed_pre = self._delay_buffer[slot].clone()
            recurrent = self._matvec(delayed_pre)
            if (
                self.config.competition_k_from_delayed_input
                or self.config.competition_factorize_delayed_input
            ):
                if self._competition_input_indices is None:
                    raise RuntimeError("adaptive competition input indices are missing")
                input_packet_count = int(torch.count_nonzero(
                    delayed_pre[self._competition_input_indices].abs()
                    > self.config.competition_epsilon
                ).item())
            self._delay_buffer[slot] = pre.detach()
            self._delay_idx += 1
        else:
            recurrent = self._matvec(pre)

        if self.config.competition_factorize_delayed_input:
            recurrent, competition_novelty = self._apply_factorized_local_competition(
                recurrent,
                delayed_pre,
            )
        else:
            recurrent, competition_novelty = self._apply_local_competition(
                recurrent,
                input_packet_count=input_packet_count,
            )

        adapt_force = ADAPTATION_COUPLING * self.adaptation

        # Noise injection (15_Equations A.2): mode-scaled, seeded for reproducibility
        noise_scale = {
            RuntimeMode.WAKE: 1.0,
            RuntimeMode.NREM: 0.3,
            RuntimeMode.REM: 0.7,
        }[mode]
        gen = torch.Generator(device=self.activation.device)
        gen.manual_seed(self.step_index * 31337 + 7)
        noise = self.config.noise_sigma * noise_scale * torch.randn(
            self.activation.shape, generator=gen, device=self.activation.device, dtype=self.activation.dtype
        )

        drive = (
            recurrent
            + self.config.external_gain * external
            + self.config.goal_gain * self.goal
            + self.config.replay_mix(mode) * replay
            - self.config.refractory_scale * self.refractory
            - adapt_force
            + noise
        )
        activation = (
            (1.0 - self.config.activation_decay(mode)) * self.activation
            + self.config.activation_gain(mode) * torch.tanh(drive)
        ).clamp(-1.0, 1.0)
        refractory = (
            (1.0 - self.config.refractory_decay(mode)) * self.refractory
            + self.config.refractory_gain(mode) * activation.square()
        )
        memory_trace = (1.0 - MEMORY_TRACE_DECAY) * self.memory_trace + MEMORY_TRACE_DECAY * activation
        adaptation = (
            (1.0 - ADAPTATION_DECAY) * self.adaptation + ADAPTATION_DECAY * activation.square()
        ).clamp(0.0, ADAPTATION_CLAMP)

        bit_lower_values, bit_upper_values = self.config.effective_bit_thresholds()
        bit_lower = torch.tensor(
            bit_lower_values,
            dtype=activation.dtype,
            device=activation.device,
        )
        bit_upper = torch.tensor(
            bit_upper_values,
            dtype=activation.dtype,
            device=activation.device,
        )
        bitfield = self.bitfield.clone()
        bitfield[activation >= bit_upper] = 1
        bitfield[activation <= bit_lower] = 0

        self.activation = activation
        self.refractory = refractory
        self.memory_trace = memory_trace
        self.adaptation = adaptation
        self.stp_u = stp_u
        self.stp_x = stp_x
        self.bitfield = bitfield
        self._commit_local_competition_usage(
            activation,
            competition_novelty,
            competition_slot,
        )

        salience = self._compute_salience(activation, external, replay, refractory)
        energy = self._energy(recurrent, replay)
        return salience, recurrent, energy

    def step(
        self,
        *,
        external_input: torch.Tensor | None = None,
        cue: torch.Tensor | None = None,
        force_mode: RuntimeMode | None = None,
        critic_score: float | None = None,
        learning_signal: float | None = None,
    ) -> RuntimeStep:
        external = (
            torch.zeros(self.config.dim, device=self.device)
            if external_input is None
            else external_input.detach().float().to(self.device).view(self.config.dim)
        )
        cue = self.activation if cue is None else cue.detach().float().to(self.device).view(self.config.dim)
        external_norm = float(external.norm().item())
        mode = force_mode or self._auto_mode(external_norm)
        replay = self.hippocampus.recall(cue, topk=self.config.memory_topk)
        if mode is not RuntimeMode.WAKE and len(self.hippocampus) > 0:
            replay = 0.5 * replay + 0.5 * self.hippocampus.replay(mode)

        if self._use_rust():
            active_count, energy = self._step_rust(external, replay, mode)
            salience = self._compute_salience(self.activation, external, replay, self.refractory)
        else:
            salience, _recurrent, energy = self._step_torch(external, replay, mode)

        active_mask = self._select_active(salience, self._f1_effective_budget(mode))
        active_count = int(active_mask.sum().item())
        self._f1_update_ema(active_count)
        stdp_gate = self._apply_runtime_stdp(
            active_count,
            energy,
            critic_score=critic_score,
            learning_signal=learning_signal,
        )
        self.mode = mode
        self.mode_occupancy[mode.value] = self.mode_occupancy.get(mode.value, 0) + 1
        self._update_lifecycle(salience, active_mask)

        priority = float((salience[active_mask].mean().item() if active_count else salience.mean().item()) + external_norm)
        if self.config.hippocampal_encoding_enabled and mode is RuntimeMode.WAKE and (external_norm > NORM_EPS or self.goal.norm().item() > NORM_EPS):
            self.hippocampus.encode(self.activation, value=self.memory_trace, priority=priority)
        elif self.config.hippocampal_encoding_enabled and mode is not RuntimeMode.WAKE and len(self.hippocampus) > 0:
            consolidated = 0.85 * self.activation + 0.15 * replay
            self.hippocampus.encode(consolidated, value=self.memory_trace, priority=priority * 0.5)

        self.hippocampus.decay_priorities()
        self._update_sleep_state(mode, active_count, external_norm)
        self.brainwave_observable()
        self.step_index += 1
        return RuntimeStep(
            step=self.step_index,
            mode=self.mode,
            energy=energy,
            active_modules=active_count,
            replay_norm=float(replay.norm().item()),
            sleep_pressure=self.sleep_pressure,
            arousal=self.arousal,
            lifecycle_counts=self.lifecycle_counts(),
            stdp_gate=stdp_gate,
            stdp_updates=self._stdp_updates,
        )

    def reset_evaluation_state(self) -> None:
        """Clear all transient dynamics without changing recurrent weights or stores.

        This is an opt-in evaluation boundary for independent-probe experiments.
        It intentionally leaves ``weight`` and the current hippocampal object
        untouched; callers that require memory cutoff must clear that store
        separately before calling this method.
        """
        self.activation.zero_(); self.refractory.zero_(); self.memory_trace.zero_()
        self.adaptation.zero_(); self.stp_u.fill_(0.5); self.stp_x.fill_(1.0)
        self.bitfield.zero_(); self.goal.zero_(); self.inactive_steps.zero_()
        self.lifecycle.fill_(_LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT])
        self.mode = RuntimeMode.WAKE; self.sleep_pressure = 0.0; self.arousal = 0.0
        self.step_index = 0; self.circadian_phase = 0.0
        self._circadian_value = CIRCADIAN_BASE + CIRCADIAN_AMP; self.nrem_cycle_count = 0
        self.mode_occupancy = {RuntimeMode.WAKE.value: 0, RuntimeMode.NREM.value: 0, RuntimeMode.REM.value: 0}
        self.active_ratio_ema = float(self.config.active_ratio); self._brainwave_history = []
        if self._delay_buffer is not None: self._delay_buffer.zero_()
        self._delay_idx = 0
        if self.competition_homeostasis is not None:
            self.competition_homeostasis.zero_()
        if self._competition_usage_buffer is not None:
            self._competition_usage_buffer.zero_()
        self._competition_usage_idx = 0
        if self.competition_packet_envelope is not None:
            self.competition_packet_envelope.zero_()
        if self.stdp_tracker is not None: self.stdp_tracker.reset()
        self._stdp_prev_critic_score = 0.0; self._stdp_updates = 0
        self._last_stdp_gate = 0.0; self._stdp_pending_learning_signal = 0.0

    def snapshot(self) -> BrainRuntimeSnapshot:
        return BrainRuntimeSnapshot(
            config=deepcopy(self.config),
            weight=self.weight.detach().cpu().clone(),
            activation=self.activation.detach().cpu().clone(),
            refractory=self.refractory.detach().cpu().clone(),
            memory_trace=self.memory_trace.detach().cpu().clone(),
            adaptation=self.adaptation.detach().cpu().clone(),
            stp_u=self.stp_u.detach().cpu().clone(),
            stp_x=self.stp_x.detach().cpu().clone(),
            bitfield=self.bitfield.detach().cpu().clone(),
            goal=self.goal.detach().cpu().clone(),
            lifecycle=self.lifecycle.detach().cpu().clone(),
            inactive_steps=self.inactive_steps.detach().cpu().clone(),
            mode=self.mode,
            sleep_pressure=float(self.sleep_pressure),
            arousal=float(self.arousal),
            step=self.step_index,
            hippocampus=deepcopy(self.hippocampus.state_dict()),
            mode_occupancy=dict(self.mode_occupancy),
            active_ratio_ema=float(self.active_ratio_ema),
            stdp_tracker=(
                deepcopy(self.stdp_tracker.state_dict())
                if self.stdp_tracker is not None
                else None
            ),
            stdp_prev_critic_score=float(self._stdp_prev_critic_score),
            stdp_updates=int(self._stdp_updates),
            circadian_phase=float(self.circadian_phase),
            circadian_value=float(self._circadian_value),
            nrem_cycle_count=int(self.nrem_cycle_count),
            delay_buffer=(
                self._delay_buffer.detach().cpu().clone()
                if self._delay_buffer is not None
                else None
            ),
            delay_idx=int(self._delay_idx),
            competition_homeostasis=(
                self.competition_homeostasis.detach().cpu().clone()
                if self.competition_homeostasis is not None
                else None
            ),
            competition_usage_buffer=(
                self._competition_usage_buffer.detach().cpu().clone()
                if self._competition_usage_buffer is not None
                else None
            ),
            competition_usage_idx=int(self._competition_usage_idx),
            competition_packet_envelope=(
                self.competition_packet_envelope.detach().cpu().clone()
                if self.competition_packet_envelope is not None
                else None
            ),
            brainwave_history=tuple(float(value) for value in self._brainwave_history),
            last_stdp_gate=float(self._last_stdp_gate),
            stdp_pending_learning_signal=float(self._stdp_pending_learning_signal),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BrainRuntimeSnapshot,
        *,
        backend: str = "auto",
        device: str | torch.device | None = None,
    ) -> "BrainRuntime":
        runtime = cls(
            snapshot.weight.detach().cpu().clone(),
            config=deepcopy(snapshot.config),
            backend=backend,
            device=device,
        )
        runtime.activation = snapshot.activation.to(runtime.device).float().clone()
        runtime.refractory = snapshot.refractory.to(runtime.device).float().clone()
        runtime.memory_trace = snapshot.memory_trace.to(runtime.device).float().clone()
        runtime.adaptation = snapshot.adaptation.to(runtime.device).float().clone()
        runtime.stp_u = snapshot.stp_u.to(runtime.device).float().clone()
        runtime.stp_x = snapshot.stp_x.to(runtime.device).float().clone()
        runtime.bitfield = snapshot.bitfield.to(runtime.device).to(torch.uint8).clone()
        runtime.goal = snapshot.goal.to(runtime.device).float().clone()
        runtime.lifecycle = snapshot.lifecycle.to(runtime.device).to(torch.int64).clone()
        runtime.inactive_steps = snapshot.inactive_steps.to(runtime.device).to(torch.int64).clone()
        runtime.mode = snapshot.mode
        runtime.sleep_pressure = float(snapshot.sleep_pressure)
        runtime.arousal = float(snapshot.arousal)
        runtime.step_index = int(snapshot.step)
        runtime.circadian_phase = float(snapshot.circadian_phase)
        runtime._circadian_value = float(snapshot.circadian_value)
        runtime.nrem_cycle_count = int(snapshot.nrem_cycle_count)
        if runtime._delay_buffer is not None and snapshot.delay_buffer is None:
            raise ValueError("snapshot delay buffer is required when axon delay is enabled")
        if runtime._delay_buffer is None and snapshot.delay_buffer is not None:
            raise ValueError("snapshot delay buffer requires axon delay to be enabled")
        if runtime._delay_buffer is not None and snapshot.delay_buffer is not None:
            expected_shape = (runtime.config.max_axon_delay, runtime.config.dim)
            if tuple(snapshot.delay_buffer.shape) != expected_shape:
                raise ValueError(
                    "snapshot delay buffer shape must match "
                    f"(max_axon_delay, dim)={expected_shape}"
                )
            runtime._delay_buffer = snapshot.delay_buffer.to(runtime.device).float().clone()
        runtime._delay_idx = int(snapshot.delay_idx)
        snapshot_homeostasis = getattr(snapshot, "competition_homeostasis", None)
        snapshot_usage = getattr(snapshot, "competition_usage_buffer", None)
        snapshot_envelope = getattr(snapshot, "competition_packet_envelope", None)
        if runtime.config.has_local_competition:
            if snapshot_homeostasis is None or snapshot_usage is None or snapshot_envelope is None:
                raise ValueError("snapshot local competition state is required")
            if tuple(snapshot_homeostasis.shape) != (runtime.config.dim,):
                raise ValueError("snapshot competition homeostasis shape must match dim")
            expected_usage_shape = (
                runtime.config.competition_delay_ticks,
                runtime.config.dim,
            )
            if tuple(snapshot_usage.shape) != expected_usage_shape:
                raise ValueError(
                    "snapshot competition usage buffer shape must match "
                    f"(competition_delay_ticks, dim)={expected_usage_shape}"
                )
            if snapshot_envelope.numel() != 1:
                raise ValueError("snapshot competition packet envelope must be scalar")
            runtime.competition_homeostasis = (
                snapshot_homeostasis.to(runtime.device).float().clone()
            )
            runtime._competition_usage_buffer = (
                snapshot_usage.to(runtime.device).float().clone()
            )
            runtime._competition_usage_idx = int(
                getattr(snapshot, "competition_usage_idx", 0)
            )
            runtime.competition_packet_envelope = (
                snapshot_envelope.to(runtime.device).float().reshape(()).clone()
            )
        elif any(value is not None for value in (
            snapshot_homeostasis, snapshot_usage, snapshot_envelope,
        )):
            raise ValueError("snapshot local competition state requires configured indices")
        runtime._brainwave_history = [float(value) for value in snapshot.brainwave_history]
        runtime.hippocampus = HippocampusMemory.from_state_dict(
            snapshot.hippocampus,
            device=runtime.device,
        )
        if snapshot.mode_occupancy:
            for key in runtime.mode_occupancy:
                runtime.mode_occupancy[key] = int(snapshot.mode_occupancy.get(key, 0))
        if snapshot.active_ratio_ema >= 0.0:
            runtime.active_ratio_ema = float(snapshot.active_ratio_ema)
        runtime._stdp_prev_critic_score = float(snapshot.stdp_prev_critic_score)
        runtime._stdp_updates = int(snapshot.stdp_updates)
        runtime._last_stdp_gate = float(snapshot.last_stdp_gate)
        runtime._stdp_pending_learning_signal = float(
            getattr(snapshot, "stdp_pending_learning_signal", 0.0)
        )
        if runtime.stdp_tracker is not None and snapshot.stdp_tracker is not None:
            runtime.stdp_tracker.load_state_dict(snapshot.stdp_tracker)
        return runtime
