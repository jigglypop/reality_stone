from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .replay_consolidation_encoding import StableBipolarEncoder
from .replay_consolidation_models import AttractorProbe, ConsolidatedRecall, ConsolidationSnapshot, ReplayEpisode


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    norms = np.linalg.norm(values, axis=0)
    valid = norms > 1e-12
    if np.any(valid):
        values[:, valid] /= norms[valid]
    if np.any(~valid):
        values[:, ~valid] = 0.0
    return values


@dataclass(frozen=True, slots=True)
class ReplayConsolidationConfig:
    dimension: int = 256
    ridge: float = 0.03
    familiarity_threshold: float = 0.62
    decode_score_threshold: float = 0.55
    decode_margin_threshold: float = 0.06
    attractor_steps: int = 3
    attractor_mix: float = 0.65
    attractor_success_cosine: float = 0.60
    tombstone_token: str = "__TOMBSTONE__"

    def __post_init__(self) -> None:
        if int(self.dimension) < 8:
            raise ValueError("dimension must be at least 8")
        if float(self.ridge) <= 0.0:
            raise ValueError("ridge must be positive")
        for value in (
            self.familiarity_threshold,
            self.decode_score_threshold,
            self.decode_margin_threshold,
            self.attractor_mix,
            self.attractor_success_cosine,
        ):
            if not np.isfinite(value):
                raise ValueError("configuration values must be finite")
        if not 0.0 <= self.attractor_mix <= 1.0:
            raise ValueError("attractor_mix must lie in [0, 1]")
        if int(self.attractor_steps) < 0:
            raise ValueError("attractor_steps must be non-negative")


class ReplayConsolidatedAttractorMemory:
    """Persistent distributed association and attractor matrices.

    Replay contributes weighted normal equations. After fitting, recall retains
    only a key-familiarity projector, a key-to-value association matrix, and a
    value-state projector; no episodic event row or key/value table is retained.
    """

    def __init__(
        self,
        *,
        seed: int,
        values: tuple[str, ...],
        config: ReplayConsolidationConfig,
        familiarity_enabled: bool = True,
        attractor_enabled: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.config = config
        self.encoder = StableBipolarEncoder(seed=self.seed, dimension=config.dimension)
        self.values = tuple(sorted(set(values)))
        if not self.values:
            raise ValueError("value vocabulary must not be empty")
        self.familiarity_enabled = bool(familiarity_enabled)
        self.attractor_enabled = bool(attractor_enabled)
        dimension = config.dimension
        self.association = np.zeros((dimension, dimension), dtype=np.float64)
        self.key_projector = np.zeros((dimension, dimension), dtype=np.float64)
        self.attractor = np.zeros((dimension, dimension), dtype=np.float64)
        self.replay_updates = 0
        self._value_codes = np.stack([self.encoder.value(value) for value in self.values])

    @staticmethod
    def _aggregate(
        episodes: tuple[ReplayEpisode, ...] | list[ReplayEpisode],
    ) -> list[ReplayEpisode]:
        counts: dict[tuple[str, str, str], int] = {}
        priorities: dict[tuple[str, str, str], float] = {}
        for episode in episodes:
            key = (episode.subject, episode.relation, episode.value)
            counts[key] = counts.get(key, 0) + int(episode.count)
            priorities[key] = max(priorities.get(key, 0.0), float(episode.priority))
        return [
            ReplayEpisode(subject, relation, value, count, priorities[key])
            for key, count in sorted(counts.items())
            for subject, relation, value in [key]
        ]

    def fit(self, episodes: tuple[ReplayEpisode, ...] | list[ReplayEpisode]) -> None:
        rows = self._aggregate(episodes)
        self.replay_updates = sum(row.count for row in rows)
        if not rows:
            return

        keys = np.stack([self.encoder.key(row.subject, row.relation) for row in rows], axis=1)
        targets = np.stack([self.encoder.value(row.value) for row in rows], axis=1)
        sqrt_weight = np.sqrt(np.asarray([row.count for row in rows], dtype=np.float64))
        weighted_keys = keys * sqrt_weight
        weighted_targets = targets * sqrt_weight

        key_gram = weighted_keys.T @ weighted_keys
        key_gram += self.config.ridge * np.eye(key_gram.shape[0])
        key_solution = np.linalg.solve(key_gram, weighted_keys.T)
        self.association = weighted_targets @ key_solution
        self.key_projector = weighted_keys @ key_solution

        if self.attractor_enabled:
            target_counts: dict[str, int] = {}
            for row in rows:
                target_counts[row.value] = target_counts.get(row.value, 0) + row.count
            target_values = sorted(target_counts)
            target_basis = np.stack(
                [self.encoder.value(value) for value in target_values], axis=1
            )
            target_basis *= np.sqrt(
                np.asarray([target_counts[value] for value in target_values], dtype=np.float64)
            )
            target_gram = target_basis.T @ target_basis
            target_gram += self.config.ridge * np.eye(target_gram.shape[0])
            target_solution = np.linalg.solve(target_gram, target_basis.T)
            self.attractor = target_basis @ target_solution
        else:
            self.attractor.fill(0.0)

    def recall_batch(
        self,
        pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        *,
        flip_rate: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[ConsolidatedRecall, ...]:
        if not pairs:
            return ()
        keys = np.stack(
            [self.encoder.key(subject, relation) for subject, relation in pairs], axis=1
        )
        if flip_rate > 0.0:
            if rng is None:
                raise ValueError("rng is required for corrupted-cue recall")
            keys = keys.copy()
            keys[rng.random(keys.shape) < float(flip_rate)] *= -1.0
            keys = _normalize_columns(keys)

        familiarity = np.linalg.norm(self.key_projector @ keys, axis=0)
        drive = self.association @ keys
        initial = _normalize_columns(drive)
        final = initial.copy()
        if self.attractor_enabled and self.config.attractor_steps > 0:
            anchor = initial.copy()
            for _ in range(self.config.attractor_steps):
                projected = self.attractor @ final
                mixed = (
                    (1.0 - self.config.attractor_mix) * anchor
                    + self.config.attractor_mix * projected
                )
                next_state = _normalize_columns(mixed)
                nonzero = np.linalg.norm(next_state, axis=0) > 1e-12
                final[:, nonzero] = next_state[:, nonzero]

        scores = self._value_codes @ final
        top_indices = np.argmax(scores, axis=0)
        columns = np.arange(scores.shape[1])
        top_scores = scores[top_indices, columns]
        if scores.shape[0] > 1:
            second_scores = np.partition(scores, -2, axis=0)[-2]
        else:
            second_scores = np.full(scores.shape[1], -1.0)
        margins = top_scores - second_scores

        results: list[ConsolidatedRecall] = []
        for index, top_index_raw in enumerate(top_indices):
            top_index = int(top_index_raw)
            top_value = self.values[top_index]
            has_drive = bool(np.linalg.norm(drive[:, index]) > 1e-12)
            initial_similarity = (
                float(initial[:, index] @ self._value_codes[top_index]) if has_drive else 0.0
            )
            final_similarity = (
                float(final[:, index] @ self._value_codes[top_index]) if has_drive else 0.0
            )
            passes_gate = has_drive and (
                not self.familiarity_enabled
                or (
                    familiarity[index] >= self.config.familiarity_threshold
                    and top_scores[index] >= self.config.decode_score_threshold
                    and margins[index] >= self.config.decode_margin_threshold
                )
            )
            abstained = (not passes_gate) or top_value == self.config.tombstone_token
            results.append(
                ConsolidatedRecall(
                    None if abstained else top_value,
                    abstained,
                    float(familiarity[index]),
                    float(top_scores[index]) if has_drive else 0.0,
                    float(margins[index]) if has_drive else 0.0,
                    initial_similarity,
                    final_similarity,
                )
            )
        return tuple(results)

    def recall(
        self,
        subject: str,
        relation: str,
        *,
        flip_rate: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> ConsolidatedRecall:
        return self.recall_batch(
            ((subject, relation),), flip_rate=flip_rate, rng=rng
        )[0]

    def probe_attractor_batch(
        self,
        values: tuple[str, ...] | list[str],
        *,
        flip_rate: float,
        rng: np.random.Generator,
        exact_hamming: bool = False,
    ) -> tuple[AttractorProbe, ...]:
        if not values:
            return ()
        targets = np.stack([self.encoder.value(value) for value in values], axis=1)
        states = targets.copy()
        if exact_hamming:
            flip_count = int(round(float(flip_rate) * states.shape[0]))
            flip_count = min(max(flip_count, 0), states.shape[0])
            for column in range(states.shape[1]):
                if flip_count:
                    indices = rng.choice(states.shape[0], size=flip_count, replace=False)
                    states[indices, column] *= -1.0
        else:
            states[rng.random(states.shape) < float(flip_rate)] *= -1.0
        states = _normalize_columns(states)
        initial = np.sum(states * targets, axis=0)
        if self.attractor_enabled:
            for _ in range(self.config.attractor_steps):
                projected = self.attractor @ states
                next_states = _normalize_columns(projected)
                nonzero = np.linalg.norm(next_states, axis=0) > 1e-12
                states[:, nonzero] = next_states[:, nonzero]
        final = np.sum(states * targets, axis=0)
        return tuple(
            AttractorProbe(
                value=value,
                initial_similarity=float(initial[index]),
                final_similarity=float(final[index]),
                similarity_gain=float(final[index] - initial[index]),
                basin_success=bool(final[index] >= self.config.attractor_success_cosine),
            )
            for index, value in enumerate(values)
        )

    def probe_attractor(
        self,
        value: str,
        *,
        flip_rate: float,
        rng: np.random.Generator,
        exact_hamming: bool = False,
    ) -> AttractorProbe:
        return self.probe_attractor_batch(
            (value,),
            flip_rate=flip_rate,
            rng=rng,
            exact_hamming=exact_hamming,
        )[0]

    def clone_with(
        self,
        *,
        familiarity_enabled: bool | None = None,
        attractor_enabled: bool | None = None,
    ) -> "ReplayConsolidatedAttractorMemory":
        use_familiarity = (
            self.familiarity_enabled
            if familiarity_enabled is None
            else bool(familiarity_enabled)
        )
        use_attractor = (
            self.attractor_enabled if attractor_enabled is None else bool(attractor_enabled)
        )
        model = ReplayConsolidatedAttractorMemory(
            seed=self.seed,
            values=self.values,
            config=self.config,
            familiarity_enabled=use_familiarity,
            attractor_enabled=use_attractor,
        )
        model.association = self.association.copy()
        model.key_projector = self.key_projector.copy()
        model.attractor = self.attractor.copy() if use_attractor else np.zeros_like(self.attractor)
        model.replay_updates = self.replay_updates
        return model

    def snapshot(self) -> ConsolidationSnapshot:
        return ConsolidationSnapshot(
            dimension=self.config.dimension,
            seed=self.seed,
            values=self.values,
            association=self.association.copy(),
            key_projector=self.key_projector.copy(),
            attractor=self.attractor.copy(),
            replay_updates=self.replay_updates,
            familiarity_enabled=self.familiarity_enabled,
            attractor_enabled=self.attractor_enabled,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ConsolidationSnapshot,
        *,
        config: ReplayConsolidationConfig,
    ) -> "ReplayConsolidatedAttractorMemory":
        if snapshot.dimension != config.dimension:
            raise ValueError("snapshot dimension does not match config")
        model = cls(
            seed=snapshot.seed,
            values=snapshot.values,
            config=config,
            familiarity_enabled=snapshot.familiarity_enabled,
            attractor_enabled=snapshot.attractor_enabled,
        )
        model.association = np.asarray(snapshot.association, dtype=np.float64).copy()
        model.key_projector = np.asarray(snapshot.key_projector, dtype=np.float64).copy()
        model.attractor = np.asarray(snapshot.attractor, dtype=np.float64).copy()
        model.replay_updates = int(snapshot.replay_updates)
        return model

    @property
    def association_weight_norm(self) -> float:
        return float(np.linalg.norm(self.association))

    @property
    def attractor_weight_norm(self) -> float:
        return float(np.linalg.norm(self.attractor))
