"""BrainRuntime adapter for the CloudCell Kleisli contract.

For a complete runtime state ``S``, control input ``U``, and observation
``O = RuntimeStep``, this adapter has the explicit type

    CloudCell: U -> (S -> D(S x O)).

``D`` is represented by :class:`FiniteDistribution`.  BrainRuntime currently
uses step-indexed deterministic pseudo-randomness, so its exact kernel is the
Dirac lift of the restored runtime transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .markov_kleisli import KleisliArrow, deterministic_kleisli_arrow
from .runtime import BrainRuntime, BrainRuntimeSnapshot, RuntimeMode, RuntimeStep


@dataclass(frozen=True)
class CloudCellInput:
    """All exogenous arguments accepted by one BrainRuntime transition."""

    external_input: torch.Tensor | None = None
    cue: torch.Tensor | None = None
    force_mode: RuntimeMode | None = None
    critic_score: float | None = None


def brain_runtime_transition(
    state: BrainRuntimeSnapshot,
    control: CloudCellInput,
    *,
    backend: str = "torch",
    device: str | torch.device | None = "cpu",
) -> tuple[BrainRuntimeSnapshot, RuntimeStep]:
    """Evaluate one transition without mutating the supplied snapshot."""

    runtime = BrainRuntime.from_snapshot(state, backend=backend, device=device)
    observation = runtime.step(
        external_input=control.external_input,
        cue=control.cue,
        force_mode=control.force_mode,
        critic_score=control.critic_score,
    )
    return runtime.snapshot(), observation


def brain_runtime_kleisli_arrow(
    *,
    backend: str = "torch",
    device: str | torch.device | None = "cpu",
) -> KleisliArrow[CloudCellInput, BrainRuntimeSnapshot, RuntimeStep]:
    """Return BrainRuntime as a Dirac-valued state/probability Kleisli arrow."""

    return deterministic_kleisli_arrow(
        lambda state, control: brain_runtime_transition(
            state,
            control,
            backend=backend,
            device=device,
        )
    )


__all__ = [
    "CloudCellInput",
    "brain_runtime_kleisli_arrow",
    "brain_runtime_transition",
]
