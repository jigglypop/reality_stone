from __future__ import annotations

from dataclasses import dataclass, field

from .replay_consolidation import (
    ReplayConsolidatedAttractorMemory,
    ReplayConsolidationConfig,
)
from .replay_consolidation_models import ConsolidatedRecall, ReplayEpisode


@dataclass(slots=True)
class ReplayCutoffBridge:
    """Opt-in bridge that makes the episodic cutoff explicit.

    Events are staged only before consolidation. ``detach_episodic`` clears the
    staging rows; subsequent recall is served solely from persistent matrices.
    """

    seed: int
    values: tuple[str, ...]
    config: ReplayConsolidationConfig
    _episodes: list[ReplayEpisode] = field(default_factory=list, init=False, repr=False)
    _model: ReplayConsolidatedAttractorMemory | None = field(
        default=None, init=False, repr=False
    )
    episodic_detached: bool = field(default=False, init=False)

    def stage(self, episode: ReplayEpisode) -> None:
        if self.episodic_detached:
            raise RuntimeError("episodic staging has been detached")
        self._episodes.append(episode)

    def consolidate(self) -> None:
        if self.episodic_detached:
            raise RuntimeError("cannot consolidate after episodic cutoff")
        model = ReplayConsolidatedAttractorMemory(
            seed=self.seed,
            values=self.values,
            config=self.config,
        )
        model.fit(tuple(self._episodes))
        self._model = model

    def detach_episodic(self) -> None:
        if self._model is None:
            raise RuntimeError("consolidate before detaching episodic staging")
        self._episodes.clear()
        self.episodic_detached = True

    def recall(self, subject: str, relation: str) -> ConsolidatedRecall:
        if not self.episodic_detached or self._model is None:
            raise RuntimeError("recall requires a consolidated, detached bridge")
        return self._model.recall(subject, relation)

    @property
    def staged_episode_count(self) -> int:
        return len(self._episodes)
