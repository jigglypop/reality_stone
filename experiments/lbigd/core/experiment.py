from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.lbigd.core.dopamine import DopamineGate
from experiments.lbigd.core.lbo import dirichlet_energy, laplacian_mul
from experiments.lbigd.core.metric import ring, update as update_metric


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


@dataclass(frozen=True)
class ExperimentAResult:
    mse_lbo: float
    mse_no_lbo: float
    improvement: float


@dataclass(frozen=True)
class ExperimentBResult:
    mse_gate_on: float
    mse_gate_off: float
    improvement: float
    gate_triggers: int


def run_experiment_a(
    *,
    seed: int = 0,
    n: int = 64,
    steps: int = 400,
    dt: float = 0.15,
    alpha: float = 2.0,
    noise: float = 0.6,
    tail: int = 80,
) -> ExperimentAResult:
    """
    LBO 필요성 실험:
      - target y(저주파) + i.i.d. 노이즈 관측 x
      - 확산(-L u) 항이 있으면 노이즈가 구조적으로 감소
    """
    if tail <= 0:
        raise ValueError("tail must be > 0")

    w = ring(int(n), weight=1.0)
    y = _lowfreq_signal(n=int(n))

    def simulate(use_lbo: bool) -> float:
        rng = np.random.default_rng(int(seed))
        u = np.zeros((n,), dtype=np.float32)
        errors: list[float] = []
        for _ in range(int(steps)):
            x = y + float(noise) * rng.standard_normal((n,), dtype=np.float32)
            du = float(alpha) * (x - u)
            if use_lbo:
                du = du - laplacian_mul(w, u)
            u = u + float(dt) * du
            errors.append(mse(u, y))
        return float(np.mean(errors[-tail:]))

    e_lbo = simulate(use_lbo=True)
    e_nolbo = simulate(use_lbo=False)
    return ExperimentAResult(mse_lbo=e_lbo, mse_no_lbo=e_nolbo, improvement=(e_nolbo - e_lbo))


def run_experiment_b(
    *,
    seed: int = 0,
    n: int = 64,
    steps1: int = 250,
    steps2: int = 350,
    dt: float = 0.15,
    alpha: float = 2.0,
    noise: float = 0.6,
    tau: float = 0.35,
    topk: int = 6,
    metric_lr: float = 0.25,
    metric_decay: float = 0.01,
    w_max: float = 1.0,
    gate_ratio: float = 1.6,
    gate_hold: int = 12,
    tail: int = 120,
) -> ExperimentBResult:
    """
    사건항(도파민 게이트) 필요성 실험:
      - phase1: ring 위에서 매끄러운 target (오차 낮음)
      - phase2: node permutation으로 "기하가 바뀐" target (오차 급등)
      - gate on: 오차 급등 시에만 metric 업데이트로 새로운 기하에 적응
      - gate off: metric 고정 -> 오차 지속
    """
    if tail <= 0:
        raise ValueError("tail must be > 0")

    rng_perm = np.random.default_rng(int(seed))
    y_base = _lowfreq_signal(n=int(n))
    perm = rng_perm.permutation(int(n))
    y2 = y_base[perm]

    def simulate(gate_enabled: bool) -> tuple[float, int]:
        rng = np.random.default_rng(int(seed))
        w = ring(int(n), weight=1.0)
        u = np.zeros((n,), dtype=np.float32)
        gate = DopamineGate(ratio=float(gate_ratio), hold_steps=int(gate_hold))
        triggers = 0
        errors: list[float] = []

        total = int(steps1) + int(steps2)
        for t in range(total):
            y = y_base if t < int(steps1) else y2
            x = y + float(noise) * rng.standard_normal((n,), dtype=np.float32)

            du = float(alpha) * (x - u) - laplacian_mul(w, u)
            u = u + float(dt) * du

            err = mse(u, y)
            errors.append(err)

            if not gate_enabled:
                continue

            open_gate = gate.update(err)
            if open_gate:
                triggers += 1
                w = update_metric(
                    w,
                    u,
                    lr=float(metric_lr),
                    tau=float(tau),
                    topk=int(topk),
                    decay=float(metric_decay),
                    w_max=float(w_max),
                )

        return float(np.mean(errors[-tail:])), triggers

    e_on, trig = simulate(gate_enabled=True)
    e_off, _ = simulate(gate_enabled=False)
    return ExperimentBResult(
        mse_gate_on=e_on,
        mse_gate_off=e_off,
        improvement=(e_off - e_on),
        gate_triggers=int(trig),
    )


def _lowfreq_signal(*, n: int) -> np.ndarray:
    x = np.arange(int(n), dtype=np.float32)
    # 저주파 + 약간의 비대칭: permutation 이후 ring smoothing이 확실히 깨지도록
    return (0.8 * np.sin(2.0 * np.pi * x / float(n)) + 0.2 * np.sin(4.0 * np.pi * x / float(n))).astype(
        np.float32
    )


def sanity_energy(*, n: int = 64) -> float:
    """
    디버그/테스트용: u^T L u >= 0 확인에 쓰는 최소 유틸.
    """
    w = ring(int(n), weight=1.0)
    u = np.random.default_rng(0).standard_normal((int(n),), dtype=np.float32)
    return dirichlet_energy(w, u)


