from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayEpisode:
    subject: str
    relation: str
    value: str
    count: int
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not self.subject or not self.relation or not self.value:
            raise ValueError("replay episode fields must be non-empty")
        if int(self.count) < 1:
            raise ValueError("replay count must be positive")


@dataclass(frozen=True, slots=True)
class ConsolidatedRecall:
    value: str | None
    abstained: bool
    familiarity: float
    top_score: float
    margin: float
    initial_similarity: float
    final_similarity: float


@dataclass(frozen=True, slots=True)
class AttractorProbe:
    value: str
    initial_similarity: float
    final_similarity: float
    similarity_gain: float
    basin_success: bool


@dataclass(frozen=True, slots=True)
class ConsolidationSnapshot:
    dimension: int
    seed: int
    values: tuple[str, ...]
    association: np.ndarray
    key_projector: np.ndarray
    attractor: np.ndarray
    replay_updates: int
    familiarity_enabled: bool
    attractor_enabled: bool


@dataclass(frozen=True, slots=True)
class SeedBundle:
    seed: int
    current_episodes: tuple[ReplayEpisode, ...]
    schedules: dict[str, tuple[ReplayEpisode, ...]]
    value_vocabulary: tuple[str, ...]
    current_targets: tuple[str, ...]
    histories: dict[tuple[str, str], tuple[object, ...]]
    target_people: tuple[str, ...]


ArmName = Literal[
    "current_prioritized_replay",
    "episodic_online_oracle",
    "no_replay",
    "random_replay",
    "all_events_replay",
    "temporal_version_shuffle",
    "cue_target_shuffle",
    "no_attractor",
    "no_familiarity_gate",
]
