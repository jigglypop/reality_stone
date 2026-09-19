"""V13 episode generation: adds a compositional held-out condition split.

This module is strictly additive to the frozen V10/V11 benchmark
(`local_cloud_benchmark.py`, `local_cloud_ood_benchmark.py`). It does not
import private names from those modules and does not modify the factorial
task definition; it only offers a second, compositional way of assigning the
same 32 (context, bit-pattern) latent conditions to train/evaluation so that
combinatorial generalization -- not lookup -- can be measured.

Diagnosed defect being addressed here (V13 scope item 2): the V10/V11
`generate_episodes` "iid" split draws train and evaluation episodes from the
identical pool of 32 latent conditions, so a model can pass by memorizing
condition -> label rather than composing context and bits. `condition_split
="compositional"` holds out 8 of the 32 cells (two bit-patterns per context,
chosen as bitwise complements so every individual bit value remains present
on both sides) for evaluation only.

Diagnosed defect of that compositional split (V13 round 3, math-verified):
holding out an index and its bitwise complement makes the held-out labels
anti-identifiable. With probability 3/4 the pair's labels are undetermined by
the six train cells of that context, and in exactly those cases every
margin-based learner (max-margin, logistic, NN) outputs the *opposite* label
for both held-out cells, so the ideal-learner ceiling on the heldout panel is
0.25. `condition_split="balanced"` is the fair replacement (numerically
confirmed): per context it holds out one +1-labelled and one -1-labelled
cell that are *not* bitwise complements of each other, under which an ideal
linear learner attains held-out accuracy 1.0. The compositional mode is kept
unmodified for reproduction of the earlier (negative) results.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .local_cloud_benchmark import (
    LocalCloudBenchmarkConfig,
    LocalCloudEpisode,
    generate_episodes,
)
from .local_cloud_kernel import LocalCloudObservation


ConditionSplit = Literal["iid", "compositional", "balanced"]
HoldoutSplit = Literal["compositional", "balanced"]
V13Panel = Literal["id", "noise", "horizon", "combined", "heldout"]
V13_PANELS: tuple[V13Panel, ...] = ("id", "noise", "horizon", "combined", "heldout")

_COMPOSITIONAL_TRAIN_CELLS = 24
_COMPOSITIONAL_EVAL_CELLS = 8

# Frozen V10 factorial context weights (same values as the inline table in
# `generate_episodes_v2` below and in the frozen V10 generator).
_CONTEXT_WEIGHTS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
)


def _exact_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _rng(seed: int, tag: int) -> np.random.Generator:
    seed = _exact_int(seed, "seed", 0)
    return np.random.default_rng(np.random.SeedSequence((seed, tag)))


def _pattern_bits(index: int) -> tuple[int, int, int]:
    """Map a 3-bit index (0..7) to the same ±1 bit-pattern encoding used by
    the frozen V10 factorial task."""
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("pattern index must be an exact integer in [0, 8)")
    return tuple(1 if (index >> shift) & 1 else -1 for shift in (2, 1, 0))  # type: ignore[return-value]


def cell_label(context: int, index: int) -> int:
    """Ground-truth label of a (context, bit-pattern-index) cell under the
    frozen V10 factorial rule: sign(w_context . bits). Never zero because the
    +-1 weight/bit dot product over three bits is odd."""
    if type(context) is not int or not 0 <= context < 4:
        raise ValueError("context must be an exact integer in [0, 4)")
    bits = _pattern_bits(index)
    value = sum(weight * bit for weight, bit in zip(_CONTEXT_WEIGHTS[context], bits))
    return 1 if value > 0 else -1


def holdout_cells(seed: int) -> tuple[tuple[int, int], ...]:
    """Deterministically choose 8 held-out (context, bit-pattern-index) cells:
    exactly two per context, chosen as an index and its bitwise complement so
    every individual bit value survives in the remaining 6 train cells for
    that context. Depends only on `seed`, not on split or evaluation tag, so
    train and evaluation calls agree on the same held-out set."""
    rng = _rng(seed, 777)
    cells: list[tuple[int, int]] = []
    for context in range(4):
        index = int(rng.integers(0, 8))
        complement = 7 - index
        cells.append((context, index))
        cells.append((context, complement))
    if len(set(cells)) != 8:
        raise RuntimeError("unreachable: complement pairing must yield eight distinct cells")
    return tuple(cells)


def holdout_cells_balanced(seed: int) -> tuple[tuple[int, int], ...]:
    """Deterministically choose 8 held-out cells for the fair "balanced"
    split: exactly two per context, one with label +1 and one with label -1,
    that are *not* bitwise complements of each other (complement pairs are
    anti-identifiable from the train cells; see module docstring). Depends
    only on `seed`, so train and evaluation calls agree on the same set."""
    rng = _rng(seed, 778)
    cells: list[tuple[int, int]] = []
    for context in range(4):
        positives = [index for index in range(8) if cell_label(context, index) == 1]
        negatives = [index for index in range(8) if cell_label(context, index) == -1]
        if len(positives) != 4 or len(negatives) != 4:
            raise RuntimeError("unreachable: each context has four cells per label")
        positive = positives[int(rng.integers(0, len(positives)))]
        allowed = [index for index in negatives if index != 7 - positive]
        if len(allowed) != 3:
            raise RuntimeError("unreachable: exactly one negative is the complement")
        negative = allowed[int(rng.integers(0, len(allowed)))]
        cells.append((context, positive))
        cells.append((context, negative))
    if len(set(cells)) != 8:
        raise RuntimeError("unreachable: balanced pairing must yield eight distinct cells")
    return tuple(cells)


def generate_episodes_v2(
    seed: int,
    count: int,
    config: LocalCloudBenchmarkConfig,
    *,
    split: Literal["train", "evaluation"],
    condition_split: ConditionSplit = "iid",
) -> tuple[LocalCloudEpisode, ...]:
    """Same factorial task as `generate_episodes`, with an optional
    compositional condition split.

    condition_split="iid": byte-for-byte identical to
    `local_cloud_benchmark.generate_episodes` (delegated, not reimplemented).

    condition_split="compositional": train draws only from the 24 cells that
    are not held out (complement-pair holdout; anti-identifiable, kept for
    reproduction only); evaluation draws only from the 8 held-out cells.

    condition_split="balanced": same 24/8 mechanics, but the held-out cells
    come from `holdout_cells_balanced` (per-context label-balanced,
    non-complement; identifiable by an ideal linear learner).
    """
    if type(condition_split) is not str or condition_split not in {
        "iid",
        "compositional",
        "balanced",
    }:
        raise ValueError("condition_split must be an exact registered string")
    if condition_split == "iid":
        return generate_episodes(seed, count, config, split=split)

    if type(config) is not LocalCloudBenchmarkConfig:
        raise ValueError("config must be an exact LocalCloudBenchmarkConfig")
    if type(split) is not str or split not in {"train", "evaluation"}:
        raise ValueError("split must be an exact registered string")

    if condition_split == "compositional":
        held = set(holdout_cells(seed))
        train_tag, evaluation_tag = 130, 240
    else:
        held = set(holdout_cells_balanced(seed))
        train_tag, evaluation_tag = 131, 241
    all_cells = tuple((context, index) for context in range(4) for index in range(8))
    if split == "train":
        pool = tuple(cell for cell in all_cells if cell not in held)
        tag = train_tag
    else:
        pool = tuple(cell for cell in all_cells if cell in held)
        tag = evaluation_tag
    expected_pool_size = _COMPOSITIONAL_TRAIN_CELLS if split == "train" else _COMPOSITIONAL_EVAL_CELLS
    if len(pool) != expected_pool_size:
        raise RuntimeError("unreachable: compositional pool size drifted from its fixed partition")
    count = _exact_int(count, "count", expected_pool_size)
    if count % expected_pool_size:
        raise ValueError(f"count must be divisible by {expected_pool_size} for this compositional split")

    rng = _rng(seed, tag)
    conditions = list(pool * (count // expected_pool_size))
    rng.shuffle(conditions)
    context_weights = np.asarray(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=np.int64)
    episodes: list[LocalCloudEpisode] = []
    for context_index, pattern_index in conditions:
        first, second, third = _pattern_bits(pattern_index)
        local_bits = np.asarray((first, second, third, rng.choice((-1, 1))), dtype=np.int64)
        local_events = rng.normal(0.0, config.noise_sigma, size=(config.episode_steps, 4, 4))
        shared_events = rng.normal(0.0, config.noise_sigma, size=(config.episode_steps, 4))
        local_events[0] += local_bits[:, None]
        shared_events[1, context_index] += 1.0
        observations = tuple(
            LocalCloudObservation(
                local=tuple(tuple(float(value) for value in row) for row in local_events[tick]),
                shared=tuple(float(value) for value in shared_events[tick]),
            )
            for tick in range(config.episode_steps)
        )
        target = int(np.sign(context_weights[context_index] @ local_bits[:3]))
        episodes.append(
            LocalCloudEpisode(
                observations=observations,
                target=target,
                context_index=context_index,
                local_bits=tuple(int(value) for value in local_bits),
            )
        )
    return tuple(episodes)


def panel_configs_v13(
    *,
    train_episodes: int,
    evaluation_episodes: int,
    heldout_split: HoldoutSplit = "compositional",
) -> dict[V13Panel, tuple[LocalCloudBenchmarkConfig, ConditionSplit]]:
    """Registered V13 evaluation panels: the four frozen V11 OOD panels
    (id/noise/horizon/combined, all condition_split="iid") plus a fifth
    `heldout` panel that evaluates strictly on the held-out cells of the
    requested split (default "compositional", reproducing the original V13
    panels byte-for-byte; "balanced" is the fair, identifiable variant) at
    the V11 `id` operating point (T=4, sigma=0.04)."""
    if type(heldout_split) is not str or heldout_split not in {"compositional", "balanced"}:
        raise ValueError("heldout_split must be an exact registered string")
    base = LocalCloudBenchmarkConfig(
        train_episodes=train_episodes, evaluation_episodes=evaluation_episodes
    )
    return {
        "id": (base, "iid"),
        "noise": (
            LocalCloudBenchmarkConfig(
                train_episodes=train_episodes,
                evaluation_episodes=evaluation_episodes,
                noise_sigma=0.08,
            ),
            "iid",
        ),
        "horizon": (
            LocalCloudBenchmarkConfig(
                train_episodes=train_episodes,
                evaluation_episodes=evaluation_episodes,
                episode_steps=8,
            ),
            "iid",
        ),
        "combined": (
            LocalCloudBenchmarkConfig(
                train_episodes=train_episodes,
                evaluation_episodes=evaluation_episodes,
                episode_steps=8,
                noise_sigma=0.08,
            ),
            "iid",
        ),
        "heldout": (base, heldout_split),
    }


__all__ = [
    "ConditionSplit",
    "HoldoutSplit",
    "V13Panel",
    "V13_PANELS",
    "cell_label",
    "holdout_cells",
    "holdout_cells_balanced",
    "generate_episodes_v2",
    "panel_configs_v13",
]
