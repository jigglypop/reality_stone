"""Locked synthetic mechanism benchmark for the V9 nested-SCC controller.

The benchmark is deliberately narrow: an early slow cue and a later fast cue
must be combined into one of four actions.  It tests state mediation, not AGI.
Development and confirmation execution are guarded by a hash-bound preregistered
manifest; importing this module never runs or writes a benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .adaptive_scc_tower_controller import (
    AdaptiveTowerController,
    CausalEvent,
    CrossScaleCut,
    UpperReset,
)
from .nested_scc_tower import NestedTowerGenerator, TowerSpec

ArmName = Literal["v9", "stateless", "level0", "upper_reset", "cross_cut", "monolithic"]
ARMS: tuple[ArmName, ...] = (
    "v9",
    "stateless",
    "level0",
    "upper_reset",
    "cross_cut",
    "monolithic",
)
NON_LESION_COMPARATORS: tuple[ArmName, ...] = ("stateless", "level0", "monolithic")


def _exact_int(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class MemoryBenchmarkConfig:
    episode_count: int = 16
    slow_gap: int = 5
    post_fast_gap: int = 2
    noise_sigma: float = 0.05
    maximum_depth: int = 6
    shell_width: int = 4
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 424242
    improvement_threshold: float = 0.02
    lesion_loss_threshold: float = 0.05

    def __post_init__(self) -> None:
        for name, minimum in (
            ("episode_count", 4),
            ("slow_gap", 1),
            ("post_fast_gap", 1),
            ("maximum_depth", 1),
            ("bootstrap_samples", 100),
            ("bootstrap_seed", 0),
        ):
            object.__setattr__(self, name, _exact_int(getattr(self, name), name, minimum=minimum))
        if type(self.shell_width) is not int or self.shell_width != 4:
            raise ValueError("shell_width is frozen to the four registered pair actions")
        if self.episode_count % 4 != 0:
            raise ValueError("episode_count must be divisible by four for balanced targets")
        for name in ("noise_sigma", "improvement_threshold", "lesion_loss_threshold"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

    @property
    def fast_tick(self) -> int:
        return self.slow_gap + 1

    @property
    def decision_tick(self) -> int:
        return self.fast_tick + self.post_fast_gap


@dataclass(frozen=True)
class MemoryEpisode:
    observations: tuple[tuple[float, ...], ...]
    target_action: int
    slow_bit: int
    fast_bit: int


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    accuracies: dict[str, float]
    v9_state_mediation_violations: int
    nonfinite_outputs: int


@dataclass(frozen=True)
class MemoryBenchmarkResult:
    phase: str
    seed_count: int
    episode_count_per_seed: int
    mean_accuracies: dict[str, float]
    strongest_comparator: str
    paired_mean_improvement: float
    paired_bootstrap_interval: tuple[float, float]
    upper_reset_loss: float
    cross_cut_loss: float
    gates: dict[str, bool]
    overall: Literal["GO", "STOP"]
    integrity: dict[str, int]
    state_scalar_counts: dict[str, int]
    estimated_mac_per_tick: dict[str, int]
    seed_metrics: tuple[SeedMetrics, ...]


_SLOW = np.asarray(((1.0, 1.0, -1.0, -1.0), (-1.0, -1.0, 1.0, 1.0)))
_FAST = np.asarray(((1.0, -1.0, 1.0, -1.0), (-1.0, 1.0, -1.0, 1.0)))


def generate_seed_episodes(seed: int, config: MemoryBenchmarkConfig) -> tuple[MemoryEpisode, ...]:
    seed = _exact_int(seed, "seed", minimum=0)
    rng = np.random.default_rng(seed)
    targets = np.tile(np.arange(4, dtype=np.int64), config.episode_count // 4)
    rng.shuffle(targets)
    episodes: list[MemoryEpisode] = []
    for target in targets:
        slow_bit = int(target) // 2
        fast_bit = int(target) % 2
        observations = rng.normal(
            0.0,
            config.noise_sigma,
            size=(config.decision_tick + 1, config.shell_width),
        )
        observations[0] += _SLOW[slow_bit]
        observations[config.fast_tick] += _FAST[fast_bit]
        episodes.append(
            MemoryEpisode(
                observations=tuple(tuple(float(value) for value in row) for row in observations),
                target_action=int(target),
                slow_bit=slow_bit,
                fast_bit=fast_bit,
            )
        )
    return tuple(episodes)


def _tower_controller(config: MemoryBenchmarkConfig, *, level_zero: bool = False):
    spec = TowerSpec(
        shell_width=config.shell_width,
        maximum_depth=0 if level_zero else config.maximum_depth,
        upward_gain=0.0 if level_zero else 0.16,
        downward_gain=0.0 if level_zero else 0.14,
    )
    return AdaptiveTowerController(NestedTowerGenerator(spec))


def _tower_prediction(
    observations: Sequence[Sequence[float]],
    config: MemoryBenchmarkConfig,
    arm: Literal["v9", "level0", "upper_reset", "cross_cut"],
) -> tuple[int, int, int]:
    controller = _tower_controller(config, level_zero=arm == "level0")
    mediation_violations = 0
    nonfinite = 0
    token = None
    for tick, observation in enumerate(observations):
        if arm == "cross_cut":
            controller = controller.with_intervention(CrossScaleCut())
        elif arm == "upper_reset" and tick == len(observations) - 1:
            controller = controller.with_intervention(UpperReset())
        token = controller.observe(CausalEvent(tick, tuple(observation)))
    assert token is not None
    policy = controller.read_policy(token, (True,) * config.shell_width)
    if token is not controller.latest_token or policy.selected_action not in range(
        config.shell_width
    ):
        mediation_violations += 1
    if not all(math.isfinite(value) for value in policy.probabilities):
        nonfinite += 1
    return policy.selected_action, mediation_violations, nonfinite


def _stateless_prediction(observations: Sequence[Sequence[float]]) -> int:
    final = np.asarray(tuple(observations)[-1], dtype=np.float64)
    return int(np.argmax(final))


def _monolithic_prediction(
    observations: Sequence[Sequence[float]], config: MemoryBenchmarkConfig
) -> int:
    level_count = config.maximum_depth + 1
    state = np.zeros((level_count, config.shell_width), dtype=np.float64)
    retentions = np.linspace(0.24, 0.84, level_count, dtype=np.float64)
    for observation in observations:
        evidence = np.asarray(observation, dtype=np.float64)
        state = np.tanh(retentions[:, None] * state + 0.45 * evidence[None, :])
    weights = np.asarray([0.72**level for level in range(level_count)], dtype=np.float64)
    weights /= float(np.sum(weights))
    logits = np.sum(weights[:, None] * state, axis=0)
    return int(np.argmax(logits))


def predict_arm(
    arm: ArmName,
    observations: Sequence[Sequence[float]],
    config: MemoryBenchmarkConfig,
) -> tuple[int, int, int]:
    if arm in ("v9", "level0", "upper_reset", "cross_cut"):
        return _tower_prediction(observations, config, arm)
    if arm == "stateless":
        return _stateless_prediction(observations), 0, 0
    if arm == "monolithic":
        return _monolithic_prediction(observations, config), 0, 0
    raise ValueError("unknown registered arm")


def evaluate_seed(seed: int, config: MemoryBenchmarkConfig) -> SeedMetrics:
    episodes = generate_seed_episodes(seed, config)
    correct = {arm: 0 for arm in ARMS}
    mediation_violations = 0
    nonfinite = 0
    for episode in episodes:
        for arm in ARMS:
            prediction, violations, invalid = predict_arm(arm, episode.observations, config)
            correct[arm] += int(prediction == episode.target_action)
            if arm == "v9":
                mediation_violations += violations
            nonfinite += invalid
    return SeedMetrics(
        seed=seed,
        accuracies={arm: correct[arm] / len(episodes) for arm in ARMS},
        v9_state_mediation_violations=mediation_violations,
        nonfinite_outputs=nonfinite,
    )


def _paired_bootstrap(values: np.ndarray, config: MemoryBenchmarkConfig) -> tuple[float, float]:
    rng = np.random.default_rng(config.bootstrap_seed)
    indices = rng.integers(0, len(values), size=(config.bootstrap_samples, len(values)))
    sampled = np.mean(values[indices], axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def evaluate_seeds(
    seeds: Sequence[int], config: MemoryBenchmarkConfig, *, phase: str
) -> MemoryBenchmarkResult:
    raw_seeds = tuple(seeds)
    if not raw_seeds or any(type(seed) is not int or seed < 0 for seed in raw_seeds):
        raise ValueError("seeds must be a nonempty exact nonnegative integer sequence")
    if len(set(raw_seeds)) != len(raw_seeds):
        raise ValueError("duplicate seeds are forbidden")
    metrics = tuple(evaluate_seed(seed, config) for seed in raw_seeds)
    means = {arm: float(np.mean([metric.accuracies[arm] for metric in metrics])) for arm in ARMS}
    strongest = max(NON_LESION_COMPARATORS, key=lambda arm: (means[arm], arm))
    paired = np.asarray(
        [metric.accuracies["v9"] - metric.accuracies[strongest] for metric in metrics]
    )
    interval = _paired_bootstrap(paired, config)
    improvement = float(np.mean(paired))
    reset_loss = means["v9"] - means["upper_reset"]
    cut_loss = means["v9"] - means["cross_cut"]
    integrity = {
        "duplicate_seeds": len(raw_seeds) - len(set(raw_seeds)),
        "state_mediation_violations": sum(
            metric.v9_state_mediation_violations for metric in metrics
        ),
        "nonfinite_outputs": sum(metric.nonfinite_outputs for metric in metrics),
        "arm_alias_violations": int(len(set(ARMS)) != len(ARMS)),
    }
    gates = {
        "mean_improvement": improvement >= config.improvement_threshold,
        "paired_lcb_positive": interval[0] > 0.0,
        "upper_reset_loss": reset_loss >= config.lesion_loss_threshold,
        "cross_cut_loss": cut_loss >= config.lesion_loss_threshold,
        "causal_integrity": all(value == 0 for value in integrity.values()),
    }
    level_count = config.maximum_depth + 1
    state_counts = {
        "v9": level_count * config.shell_width,
        "stateless": 0,
        "level0": config.shell_width,
        "upper_reset": level_count * config.shell_width,
        "cross_cut": level_count * config.shell_width,
        "monolithic": level_count * config.shell_width,
    }
    v9_macs = (level_count + 2 * (level_count - 1)) * config.shell_width**2
    macs = {
        "v9": v9_macs,
        "stateless": 0,
        "level0": config.shell_width**2,
        "upper_reset": v9_macs,
        "cross_cut": v9_macs,
        "monolithic": 2 * level_count * config.shell_width,
    }
    return MemoryBenchmarkResult(
        phase=phase,
        seed_count=len(raw_seeds),
        episode_count_per_seed=config.episode_count,
        mean_accuracies=means,
        strongest_comparator=strongest,
        paired_mean_improvement=improvement,
        paired_bootstrap_interval=interval,
        upper_reset_loss=reset_loss,
        cross_cut_loss=cut_loss,
        gates=gates,
        overall="GO" if all(gates.values()) else "STOP",
        integrity=integrity,
        state_scalar_counts=state_counts,
        estimated_mac_per_tick=macs,
        seed_metrics=metrics,
    )


def preregistration_payload(
    *,
    repository_root: str | Path,
    config: MemoryBenchmarkConfig,
) -> dict[str, object]:
    root = Path(repository_root)
    sources = (
        "reality_stone/python/reality_stone/clarus/nested_scc_memory_benchmark.py",
        "reality_stone/python/reality_stone/clarus/nested_scc_tower.py",
        "reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py",
    )
    return {
        "schema": "clarus.v9-memory-prereg.v1",
        "status": "LOCKED_BEFORE_RESULTS",
        "config": asdict(config),
        "arms": ARMS,
        "non_lesion_comparators": NON_LESION_COMPARATORS,
        "development_seeds": [0, 255],
        "confirmation_seeds": [10000, 10255],
        "confirmation_policy": "FORBIDDEN_UNLESS_DEVELOPMENT_GO_AND_SEPARATE_AUDIT_PASS",
        "primary_gates": {
            "mean_improvement": ">= 0.02",
            "paired_bootstrap_lcb": "> 0.0",
            "upper_reset_loss": ">= 0.05",
            "cross_cut_loss": ">= 0.05",
            "causal_integrity": "all zero",
        },
        "source_sha256": {source: sha256_file(root / source) for source in sources},
    }


def verify_preregistration(
    preregistration: dict[str, object], *, repository_root: str | Path
) -> MemoryBenchmarkConfig:
    root = Path(repository_root)
    if preregistration.get("schema") != "clarus.v9-memory-prereg.v1":
        raise ValueError("preregistration schema mismatch")
    if preregistration.get("status") != "LOCKED_BEFORE_RESULTS":
        raise ValueError("preregistration was not locked before results")
    expected = preregistration_payload(
        repository_root=root,
        config=MemoryBenchmarkConfig(**preregistration["config"]),
    )
    if canonical_json(preregistration) != canonical_json(expected):
        raise ValueError("preregistration content or source hash mismatch")
    return MemoryBenchmarkConfig(**preregistration["config"])


def run_locked_phase(
    *,
    repository_root: str | Path,
    preregistration_path: str | Path,
    result_path: str | Path,
    phase: Literal["development", "confirmation"],
    authorization_path: str | Path,
    development_result_path: str | Path | None = None,
) -> dict[str, object]:
    """Execute exactly one preregistered phase after its explicit audit gate."""

    root = Path(repository_root).resolve()
    prereg_path = Path(preregistration_path).resolve()
    output = Path(result_path).resolve()
    authorization = Path(authorization_path).resolve()
    if output.exists():
        raise FileExistsError("a locked phase result already exists; reruns are forbidden")
    if not authorization.is_file() or "Gate: PASS" not in authorization.read_text(encoding="utf-8"):
        raise PermissionError("the phase authorization audit is absent or not PASS")
    preregistration = json.loads(prereg_path.read_text(encoding="utf-8"))
    config = verify_preregistration(preregistration, repository_root=root)
    if phase == "development":
        seeds = tuple(range(0, 256))
    elif phase == "confirmation":
        if development_result_path is None:
            raise PermissionError("confirmation requires the preserved development result")
        development = json.loads(Path(development_result_path).read_text(encoding="utf-8"))
        if development.get("result", {}).get("overall") != "GO":
            raise PermissionError("confirmation is forbidden because development was not GO")
        seeds = tuple(range(10000, 10256))
    else:
        raise ValueError("phase must be development or confirmation")
    result = evaluate_seeds(seeds, config, phase=phase)
    payload = {
        "schema": "clarus.v9-memory-result.v1",
        "phase": phase,
        "preregistration_sha256": sha256_file(prereg_path),
        "authorization_sha256": sha256_file(authorization),
        "result": asdict(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


__all__ = [
    "ARMS",
    "MemoryBenchmarkConfig",
    "MemoryBenchmarkResult",
    "MemoryEpisode",
    "SeedMetrics",
    "canonical_json",
    "evaluate_seed",
    "evaluate_seeds",
    "generate_seed_episodes",
    "predict_arm",
    "preregistration_payload",
    "run_locked_phase",
    "sha256_file",
    "verify_preregistration",
]
