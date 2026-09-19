"""Auditable bounded episodic memory for explicit ADD/UPDATE/DELETE/NOOP."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def _unit(vector: torch.Tensor) -> torch.Tensor:
    value = vector.detach().float().view(-1)
    if not torch.isfinite(value).all() or value.norm().item() <= 0.0:
        raise ValueError("memory key must be finite and nonzero")
    return value / value.norm()


@dataclass(frozen=True)
class MemoryRecall:
    value: int | None
    evidence_id: str | None
    similarity: float
    margin: float
    abstained: bool


@dataclass
class _Entry:
    key: torch.Tensor
    value: int
    evidence_id: str
    priority: float
    timestamp: int


@dataclass
class AuditedEpisodicMemory:
    dim: int
    capacity: int = 12
    update_similarity: float = 0.92
    recall_similarity: float = 0.60
    recall_margin: float = 0.05
    merge_updates: bool = True
    abstention_enabled: bool = True
    _entries: list[_Entry] = field(default_factory=list, init=False, repr=False)
    audit_log: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        self.capacity = int(self.capacity)
        if self.dim < 1 or self.capacity < 1:
            raise ValueError("dim and capacity must be positive")

    def __len__(self) -> int:
        return len(self._entries)

    def _nearest(self, key: torch.Tensor) -> tuple[int | None, float]:
        if not self._entries:
            return None, -1.0
        similarities = torch.tensor([float(entry.key @ key) for entry in self._entries])
        index = int(similarities.argmax().item())
        return index, float(similarities[index].item())

    def _eviction_index(self, now: int) -> int:
        utilities: list[float] = []
        for index, entry in enumerate(self._entries):
            other = [float(entry.key @ candidate.key) for j, candidate in enumerate(self._entries) if j != index]
            novelty = 1.0 - max(other, default=0.0)
            recency = math.exp(-max(now - entry.timestamp, 0) / max(self.capacity, 1))
            utilities.append(math.log1p(entry.priority) + 0.35 * novelty + 0.25 * recency)
        return min(range(len(utilities)), key=utilities.__getitem__)

    def upsert(self, key: torch.Tensor, value: int, evidence_id: str, *, priority: float, timestamp: int) -> str:
        try:
            normalized = _unit(key)
            if normalized.numel() != self.dim or not evidence_id or not math.isfinite(priority) or priority <= 0.0:
                raise ValueError
        except (TypeError, ValueError):
            self.audit_log.append({"operation": "NOOP", "evidence_id": str(evidence_id)})
            return "NOOP"
        index, similarity = self._nearest(normalized)
        if self.merge_updates and index is not None and similarity >= self.update_similarity:
            previous = self._entries[index]
            self._entries[index] = _Entry(
                normalized, int(value), str(evidence_id), max(float(priority), previous.priority), int(timestamp)
            )
            self.audit_log.append({"operation": "UPDATE", "evidence_id": str(evidence_id), "replaced": previous.evidence_id})
            return "UPDATE"
        if len(self._entries) >= self.capacity:
            self._entries.pop(self._eviction_index(int(timestamp)))
        self._entries.append(_Entry(normalized, int(value), str(evidence_id), float(priority), int(timestamp)))
        self.audit_log.append({"operation": "ADD", "evidence_id": str(evidence_id)})
        return "ADD"

    def delete(self, evidence_id: str) -> bool:
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.evidence_id != evidence_id]
        deleted = len(self._entries) < before
        self.audit_log.append({"operation": "DELETE", "evidence_id": str(evidence_id), "deleted": deleted})
        return deleted

    def recall(self, cue: torch.Tensor) -> MemoryRecall:
        try:
            normalized = _unit(cue)
        except ValueError:
            return MemoryRecall(None, None, 0.0, 0.0, True)
        if normalized.numel() != self.dim or not self._entries:
            return MemoryRecall(None, None, 0.0, 0.0, True)
        scores = sorted(
            ((float(entry.key @ normalized), index) for index, entry in enumerate(self._entries)),
            reverse=True,
        )
        top, index = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        margin = top - second
        abstain = self.abstention_enabled and (top < self.recall_similarity or margin < self.recall_margin)
        if abstain:
            return MemoryRecall(None, None, top, margin, True)
        entry = self._entries[index]
        return MemoryRecall(entry.value, entry.evidence_id, top, margin, False)


__all__ = ["AuditedEpisodicMemory", "MemoryRecall"]
