"""Certified local-state/shared-state transition kernel.

This is an isolated V10 research primitive.  It is deliberately smaller than
an agent runtime: observations enter the transition, recurrent state leaves
it, and no target, label, oracle probability, or hand-written decision can
bypass that state.  Biological identity and AGI capability are not claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence

import numpy as np


KernelLesion = Literal["intact", "cross_cut", "local_reset", "cloud_reset"]


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not bool or encoded text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite and dimensionless")
    return result


def _exact_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _vector(values: Sequence[float], *, width: int, name: str) -> np.ndarray:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a width-{width} sequence") from error
    if len(raw) != width:
        raise ValueError(f"{name} must have shape ({width},)")
    return np.asarray(
        tuple(_finite_float(value, f"{name}[{index}]") for index, value in enumerate(raw)),
        dtype=np.float64,
    )


def _matrix(values: Sequence[Sequence[float]], *, rows: int, width: int, name: str) -> np.ndarray:
    try:
        raw_rows = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a ({rows}, {width}) sequence") from error
    if len(raw_rows) != rows:
        raise ValueError(f"{name} must have shape ({rows}, {width})")
    return np.stack(
        tuple(
            _vector(row, width=width, name=f"{name}[{index}]") for index, row in enumerate(raw_rows)
        )
    )


@dataclass(frozen=True)
class LocalCloudKernelConfig:
    """Dimensionless coefficients for the synchronous Jacobi transition."""

    local_count: int = 4
    width: int = 4
    local_retentions: tuple[float, ...] = (0.62, 0.62, 0.62, 0.62)
    cloud_retention: float = 0.72
    input_gain: float = 0.50
    local_to_cloud: float = 0.06
    cloud_to_local: float = 0.06
    interaction_gain: float = 0.20
    contraction_cap: float = 0.95

    def __post_init__(self) -> None:
        local_count = _exact_positive_int(self.local_count, "local_count")
        _exact_positive_int(self.width, "width")
        if type(self.local_retentions) is not tuple:
            raise ValueError("local_retentions must be an exact built-in tuple")
        retentions = tuple(
            _finite_float(value, f"local_retentions[{index}]")
            for index, value in enumerate(self.local_retentions)
        )
        if len(retentions) != local_count:
            raise ValueError("local_retentions must match local_count")
        if any(not 0.0 <= value < 1.0 for value in retentions):
            raise ValueError("local retentions must lie in [0, 1)")
        object.__setattr__(self, "local_retentions", retentions)

        for name in (
            "cloud_retention",
            "input_gain",
            "local_to_cloud",
            "cloud_to_local",
            "interaction_gain",
            "contraction_cap",
        ):
            value = _finite_float(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if not 0.0 <= self.cloud_retention < 1.0:
            raise ValueError("cloud_retention must lie in [0, 1)")
        if (
            self.input_gain < 0.0
            or self.local_to_cloud < 0.0
            or self.cloud_to_local < 0.0
            or self.interaction_gain < 0.0
        ):
            raise ValueError("input and cross gains must be nonnegative")
        if not 0.0 < self.contraction_cap < 1.0:
            raise ValueError("contraction_cap must lie in (0, 1)")


@dataclass(frozen=True)
class LocalCloudState:
    local: tuple[tuple[float, ...], ...]
    cloud: tuple[float, ...]


@dataclass(frozen=True)
class LocalCloudObservation:
    """Raw feature availability split into private and shared channels."""

    local: tuple[tuple[float, ...], ...]
    shared: tuple[float, ...]


@dataclass(frozen=True)
class SmallGainCertificate:
    matrix: tuple[tuple[float, float], tuple[float, float]]
    spectral_radius: float
    weights: tuple[float, float]
    contraction_factor: float
    contraction_cap: float
    metric: Literal["weighted_local_cloud_sup"]
    certified: bool


@dataclass(frozen=True)
class KernelStep:
    state: LocalCloudState
    local_summary: tuple[float, ...]
    cloud_summary: tuple[float, ...]
    features: tuple[float, ...]
    lesion: KernelLesion


class LocalCloudTransitionKernel:
    """A target-free compositional state transition with explicit lesions."""

    def __init__(self, config: LocalCloudKernelConfig | None = None) -> None:
        self.config = config if config is not None else LocalCloudKernelConfig()
        self.certificate = self._small_gain_certificate()
        if not self.certificate.certified:
            raise ValueError("local/cloud recurrence is not certified in weighted_local_cloud_sup")

    def zero_state(self) -> LocalCloudState:
        return LocalCloudState(
            local=tuple((0.0,) * self.config.width for _ in range(self.config.local_count)),
            cloud=(0.0,) * self.config.width,
        )

    def _validated_state(self, state: LocalCloudState) -> tuple[np.ndarray, np.ndarray]:
        if type(state) is not LocalCloudState:
            raise ValueError("state must be an exact LocalCloudState")
        local = _matrix(
            state.local,
            rows=self.config.local_count,
            width=self.config.width,
            name="state.local",
        )
        cloud = _vector(state.cloud, width=self.config.width, name="state.cloud")
        if np.any(np.abs(local) > 1.0) or np.any(np.abs(cloud) > 1.0):
            raise ValueError("recurrent state must lie in the exact closed domain [-1, 1]")
        return local, cloud

    def _small_gain_certificate(self) -> SmallGainCertificate:
        config = self.config
        matrix = np.asarray(
            (
                (
                    max(config.local_retentions) + config.interaction_gain,
                    config.cloud_to_local + config.interaction_gain,
                ),
                (config.local_to_cloud, config.cloud_retention),
            ),
            dtype=np.float64,
        )
        eigenvalues = np.linalg.eigvals(matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        if spectral_radius >= 1.0:
            weights = (math.inf, math.inf)
            factor = math.inf
            certified = False
        else:
            solved = np.linalg.solve(np.eye(2, dtype=np.float64) - matrix, np.ones(2))
            ratios = matrix @ solved / solved
            weights = (float(solved[0]), float(solved[1]))
            factor = float(np.max(ratios))
            certified = bool(
                np.all(solved > 0.0) and factor < 1.0 and factor <= config.contraction_cap
            )
        return SmallGainCertificate(
            matrix=(
                (float(matrix[0, 0]), float(matrix[0, 1])),
                (float(matrix[1, 0]), float(matrix[1, 1])),
            ),
            spectral_radius=spectral_radius,
            weights=weights,
            contraction_factor=factor,
            contraction_cap=config.contraction_cap,
            metric="weighted_local_cloud_sup",
            certified=certified,
        )

    def step(
        self,
        state: LocalCloudState,
        observation: LocalCloudObservation,
        *,
        lesion: KernelLesion = "intact",
    ) -> KernelStep:
        if type(lesion) is not str or lesion not in {
            "intact",
            "cross_cut",
            "local_reset",
            "cloud_reset",
        }:
            raise ValueError("lesion must be one of the registered exact strings")
        local, cloud = self._validated_state(state)
        if type(observation) is not LocalCloudObservation:
            raise ValueError("observation must be an exact LocalCloudObservation")
        observed = _matrix(
            observation.local,
            rows=self.config.local_count,
            width=self.config.width,
            name="observation.local",
        )
        shared = _vector(
            observation.shared,
            width=self.config.width,
            name="observation.shared",
        )

        source_local = np.zeros_like(local) if lesion == "local_reset" else local
        source_cloud = np.zeros_like(cloud) if lesion == "cloud_reset" else cloud
        local_to_cloud = 0.0 if lesion == "cross_cut" else self.config.local_to_cloud
        cloud_to_local = 0.0 if lesion == "cross_cut" else self.config.cloud_to_local
        interaction_gain = 0.0 if lesion == "cross_cut" else self.config.interaction_gain

        pooled_local = np.mean(source_local, axis=0)
        local_next = np.empty_like(source_local)
        for index, retention in enumerate(self.config.local_retentions):
            local_next[index] = np.tanh(
                retention * source_local[index]
                + self.config.input_gain * observed[index]
                + cloud_to_local * source_cloud
                + interaction_gain * source_local[index] * source_cloud
            )
        cloud_next = np.tanh(
            self.config.cloud_retention * source_cloud
            + self.config.input_gain * shared
            + local_to_cloud * pooled_local
        )

        next_state = LocalCloudState(
            local=tuple(tuple(float(value) for value in row) for row in local_next),
            cloud=tuple(float(value) for value in cloud_next),
        )
        features = tuple(float(value) for value in local_next.reshape(-1)) + next_state.cloud
        return KernelStep(
            state=next_state,
            local_summary=tuple(float(value) for value in np.mean(local_next, axis=1)),
            cloud_summary=next_state.cloud,
            features=features,
            lesion=lesion,
        )

    def compose(
        self,
        state: LocalCloudState,
        observations: Sequence[LocalCloudObservation],
        *,
        lesion: KernelLesion = "intact",
    ) -> KernelStep:
        try:
            events = tuple(observations)
        except TypeError as error:
            raise ValueError("observations must be a nonempty finite sequence") from error
        if not events:
            local, cloud = self._validated_state(state)
            features = tuple(float(value) for value in local.reshape(-1)) + tuple(
                float(value) for value in cloud
            )
            return KernelStep(
                state=state,
                local_summary=tuple(float(value) for value in np.mean(local, axis=1)),
                cloud_summary=state.cloud,
                features=features,
                lesion=lesion,
            )
        result: KernelStep | None = None
        current = state
        for observation in events:
            result = self.step(current, observation, lesion=lesion)
            current = result.state
        if result is None:  # pragma: no cover - guarded by the nonempty branch above
            raise RuntimeError("unreachable empty composition")
        return result
