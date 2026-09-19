"""Research helpers for comparing HF language models with CE runtime probes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


@dataclass
class RuntimeProbeResult:
    mse: float
    cosine: float
    active_ratio: float
    stdp_updates: int
    stdp_gate: float


@dataclass
class PhaseLockProbeResult:
    initial_alignment: float
    final_alignment: float
    initial_coherence: float
    final_coherence: float
    steps: int


@dataclass
class PhaseNetworkProbeResult:
    initial_coherence: float
    final_coherence: float
    initial_alignment: float | None
    final_alignment: float | None
    steps: int


def pca_projection(hidden: torch.Tensor, max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Project hidden states to a compact orthonormal basis for runtime probes."""
    hidden = hidden.float()
    if hidden.ndim != 2:
        raise ValueError("hidden must be a 2D tensor")
    max_dim = max(1, int(max_dim))
    if hidden.shape[1] <= max_dim:
        basis = torch.eye(hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
        return hidden, basis

    centered = hidden - hidden.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:max_dim].T.contiguous()
    return centered @ basis, basis


def apply_projection(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    hidden = hidden.float()
    if hidden.shape[1] == basis.shape[0]:
        centered = hidden - hidden.mean(dim=0, keepdim=True)
        return centered @ basis.to(hidden.device)
    if hidden.shape[1] == basis.shape[1]:
        return hidden
    raise ValueError("hidden dimension does not match projection basis")


def hopfield_from_hidden(hidden: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Build a stable symmetric coupling matrix from hidden-state covariance."""
    hidden = hidden.float()
    if hidden.ndim != 2:
        raise ValueError("hidden must be a 2D tensor")
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    n, dim = centered.shape
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.T)
    cov.fill_diagonal_(0.0)
    lam_max = float(torch.linalg.eigvalsh(cov)[-1].item())
    if lam_max >= -float(ridge):
        cov = cov - (lam_max + float(ridge)) * torch.eye(dim, dtype=cov.dtype, device=cov.device)
    return cov


def normalized_drive(hidden: torch.Tensor, gain: float = 0.4) -> torch.Tensor:
    hidden = hidden.float()
    return F.normalize(hidden, dim=1) * float(gain)


def periodic_stimulus(
    steps: int,
    dim: int,
    *,
    frequency: float,
    amplitude: float = 1.0,
    phase_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a protist-scale periodic cue as complex oscillator modes."""
    steps = max(1, int(steps))
    dim = max(1, int(dim))
    if phase_offsets is None:
        phase_offsets = torch.linspace(0.0, math.pi, dim)
    phase_offsets = phase_offsets.float()
    if phase_offsets.numel() != dim:
        raise ValueError("phase_offsets must match dim")
    time = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    phase = 2.0 * math.pi * float(frequency) * time + phase_offsets.unsqueeze(0)
    radius = torch.full_like(phase, float(amplitude))
    return torch.polar(radius, phase)


def phase_coherence(state: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Kuramoto order parameter for internal phase agreement."""
    state = _as_complex(state)
    mask = state.abs() > float(eps)
    if not bool(mask.any().item()):
        return 0.0
    phasor = state[mask] / state[mask].abs().clamp_min(float(eps))
    return float(phasor.mean().abs().item())


def phase_alignment(state: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Mean cosine alignment against an external grounding phase."""
    state = _as_complex(state)
    reference = _as_complex(reference).to(state.device)
    if state.shape != reference.shape:
        raise ValueError("state and reference must have the same shape")
    mask = (state.abs() > float(eps)) & (reference.abs() > float(eps))
    if not bool(mask.any().item()):
        return 0.0
    delta = torch.angle(state[mask]) - torch.angle(reference[mask])
    return float(torch.cos(delta).mean().item())


def phase_grounding_risk(
    state: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Risk is high when internal phase agreement lacks external grounding."""
    coherence = phase_coherence(state, eps=eps)
    alignment = phase_alignment(state, reference, eps=eps)
    grounding = min(max((alignment + 1.0) * 0.5, 0.0), 1.0)
    return float(coherence * (1.0 - grounding))


def phase_grounding_suppression(
    logits: torch.Tensor,
    candidate_states: torch.Tensor,
    reference: torch.Tensor,
    *,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress candidates whose coherent phase is not grounded in the cue."""
    if candidate_states.ndim != 2:
        raise ValueError("candidate_states must be a 2D tensor")
    logits = logits.float()
    if logits.shape[0] != candidate_states.shape[0]:
        raise ValueError("logits and candidate_states must have the same first dimension")
    risks = torch.tensor(
        [phase_grounding_risk(candidate, reference) for candidate in candidate_states],
        dtype=logits.dtype,
        device=logits.device,
    )
    return logits - float(strength) * risks, risks


def phase_coupling_step(
    state: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    neighbor_coupling: float,
    external: torch.Tensor | None = None,
    external_coupling: float = 0.0,
) -> torch.Tensor:
    """Kuramoto-style graph coupling for early network synchronization."""
    state = _as_complex(state)
    adjacency = adjacency.to(state.device).float()
    if adjacency.shape != (state.numel(), state.numel()):
        raise ValueError("adjacency must be square with size matching state")

    theta = torch.angle(state)
    pair_delta = theta.unsqueeze(0) - theta.unsqueeze(1)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    neighbor_pull = (adjacency * torch.sin(pair_delta)).sum(dim=1) / degree
    phase = theta + float(neighbor_coupling) * neighbor_pull

    if external is not None:
        external = _as_complex(external).to(state.device)
        if external.shape != state.shape:
            raise ValueError("external must match state shape")
        phase = phase + float(external_coupling) * torch.angle(external * state.conj())

    return torch.polar(state.abs(), phase)


def phase_network_probe(
    initial_state: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    steps: int,
    neighbor_coupling: float,
    external: torch.Tensor | None = None,
    external_coupling: float = 0.0,
) -> PhaseNetworkProbeResult:
    """Measure whether graph coupling improves collective phase coherence."""
    state = _as_complex(initial_state)
    initial_alignment = None if external is None else phase_alignment(state, external)
    initial_coherence = phase_coherence(state)
    for _ in range(max(0, int(steps))):
        state = phase_coupling_step(
            state,
            adjacency,
            neighbor_coupling=neighbor_coupling,
            external=external,
            external_coupling=external_coupling,
        )
    return PhaseNetworkProbeResult(
        initial_coherence=initial_coherence,
        final_coherence=phase_coherence(state),
        initial_alignment=initial_alignment,
        final_alignment=None if external is None else phase_alignment(state, external),
        steps=max(0, int(steps)),
    )


def phase_lock_step(
    state: torch.Tensor,
    reference: torch.Tensor,
    *,
    coupling: float,
    amplitude_rate: float = 0.0,
) -> torch.Tensor:
    """Move oscillator phases toward an external cue without changing topology."""
    state = _as_complex(state)
    reference = _as_complex(reference).to(state.device)
    if state.shape != reference.shape:
        raise ValueError("state and reference must have the same shape")

    coupling = min(max(float(coupling), 0.0), 1.0)
    amplitude_rate = min(max(float(amplitude_rate), 0.0), 1.0)
    delta = torch.angle(reference * state.conj())
    phase = torch.angle(state) + coupling * delta
    radius = (1.0 - amplitude_rate) * state.abs() + amplitude_rate * reference.abs()
    return torch.polar(radius, phase)


def phase_lock_probe(
    stimulus: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    coupling: float,
    amplitude_rate: float = 0.0,
) -> PhaseLockProbeResult:
    """Measure whether a simple agent phase-locks to a periodic cue."""
    stimulus = _as_complex(stimulus)
    if stimulus.ndim != 2:
        raise ValueError("stimulus must be a 2D tensor")
    state = _as_complex(initial_state).to(stimulus.device)
    if state.shape != stimulus[0].shape:
        raise ValueError("initial_state must match one stimulus frame")

    initial_alignment = phase_alignment(state, stimulus[0])
    initial_coherence = phase_coherence(state)
    for reference in stimulus:
        state = phase_lock_step(
            state,
            reference,
            coupling=coupling,
            amplitude_rate=amplitude_rate,
        )
    return PhaseLockProbeResult(
        initial_alignment=initial_alignment,
        final_alignment=phase_alignment(state, stimulus[-1]),
        initial_coherence=initial_coherence,
        final_coherence=phase_coherence(state),
        steps=int(stimulus.shape[0]),
    )


def _as_complex(state: torch.Tensor) -> torch.Tensor:
    if state.is_complex():
        return state
    return torch.complex(state.float(), torch.zeros_like(state).float())


def build_runtime(
    weight: torch.Tensor,
    *,
    active_ratio: float,
    stdp_enabled: bool,
    stdp_lr: float,
    stdp_apply_interval: int,
    stdp_density: float,
) -> BrainRuntime:
    return BrainRuntime(
        weight.detach().cpu().float(),
        config=BrainRuntimeConfig(
            dim=weight.shape[0],
            active_ratio=active_ratio,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            stdp_enabled=stdp_enabled,
            stdp_apply_interval=stdp_apply_interval,
            stdp_lr=stdp_lr,
            stdp_density=stdp_density,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.0,
        ),
        backend="torch",
        device="cpu",
    )


def train_runtime_stdp(runtime: BrainRuntime, train_drive: torch.Tensor, steps: int) -> BrainRuntime:
    if train_drive.ndim != 2:
        raise ValueError("train_drive must be a 2D tensor")
    steps = max(0, int(steps))
    for idx in range(steps):
        external = train_drive[idx % train_drive.shape[0]].cpu()
        runtime.step(external_input=external, force_mode=RuntimeMode.WAKE)
    return runtime


def evaluate_transition_probe(runtime: BrainRuntime, eval_drive: torch.Tensor) -> RuntimeProbeResult:
    """Measure whether runtime activations predict the next hidden-state drive."""
    if eval_drive.ndim != 2:
        raise ValueError("eval_drive must be a 2D tensor")
    if eval_drive.shape[0] < 2:
        raise ValueError("eval_drive must contain at least two states")

    preds = []
    targets = []
    active = []
    last_step = None
    for idx in range(eval_drive.shape[0] - 1):
        step = runtime.step(external_input=eval_drive[idx].cpu(), force_mode=RuntimeMode.WAKE)
        preds.append(F.normalize(runtime.activation.detach().cpu(), dim=0))
        targets.append(F.normalize(eval_drive[idx + 1].detach().cpu(), dim=0))
        active.append(step.active_modules / max(runtime.config.dim, 1))
        last_step = step

    pred = torch.stack(preds)
    target = torch.stack(targets)
    mse = float(F.mse_loss(pred, target).item())
    cosine = float(F.cosine_similarity(pred, target, dim=1).mean().item())
    return RuntimeProbeResult(
        mse=mse,
        cosine=cosine,
        active_ratio=float(sum(active) / max(len(active), 1)),
        stdp_updates=0 if last_step is None else int(last_step.stdp_updates),
        stdp_gate=0.0 if last_step is None else float(last_step.stdp_gate),
    )
