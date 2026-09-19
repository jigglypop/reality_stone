"""Bounded valid-time memory for the optional RuntimeAgent temporal bridge.

This module is intentionally structured and non-semantic.  It stores explicit
(subject, relation, value, valid_session, evidence_id) events, resolves by valid
time rather than arrival order, preserves tombstones, and returns provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TemporalOperation(str, Enum):
    UPSERT = "UPSERT"
    DELETE = "DELETE"


RecallMode = Literal["current", "previous", "as_of"]


@dataclass(frozen=True, slots=True)
class TemporalMemoryEvent:
    subject: str
    relation: str
    value: str | None
    valid_session: int
    sequence: int
    evidence_id: str
    operation: TemporalOperation = TemporalOperation.UPSERT
    priority: float = 1.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.relation)

    @property
    def logical_version(self) -> tuple[int, int, str]:
        return (self.valid_session, self.sequence, self.evidence_id)


@dataclass(frozen=True, slots=True)
class TemporalMemoryRecall:
    value: str | None
    evidence_id: str | None
    valid_session: int | None
    abstained: bool
    cost: int

    @classmethod
    def unknown(cls, *, cost: int = 0) -> "TemporalMemoryRecall":
        return cls(None, None, None, True, cost)


@dataclass(frozen=True, slots=True)
class TemporalAuditEntry:
    action: str
    evidence_id: str
    key: tuple[str, str]
    valid_session: int


@dataclass
class TemporalAuditedMemory:
    capacity: int = 128
    max_versions_per_key: int = 3
    _versions: dict[tuple[str, str], list[TemporalMemoryEvent]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _evidence_ids: set[str] = field(default_factory=set, init=False, repr=False)
    audit_log: list[TemporalAuditEntry] = field(default_factory=list, init=False)
    recall_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.capacity = int(self.capacity)
        self.max_versions_per_key = int(self.max_versions_per_key)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if self.max_versions_per_key < 2:
            raise ValueError("max_versions_per_key must be at least 2")

    def __len__(self) -> int:
        return sum(len(rows) for rows in self._versions.values())

    def ingest(self, event: TemporalMemoryEvent) -> str:
        if not event.subject or not event.relation or not event.evidence_id:
            raise ValueError("subject, relation, and evidence_id are required")
        if event.valid_session < 0 or event.sequence < 0:
            raise ValueError("valid_session and sequence must be non-negative")
        if not math.isfinite(event.priority) or event.priority <= 0.0:
            raise ValueError("priority must be finite and positive")
        if event.evidence_id in self._evidence_ids:
            self.audit_log.append(
                TemporalAuditEntry(
                    "NOOP_DUPLICATE",
                    event.evidence_id,
                    event.key,
                    event.valid_session,
                )
            )
            return "NOOP_DUPLICATE"

        rows = self._versions.setdefault(event.key, [])
        previous_latest = rows[-1] if rows else None
        rows.append(event)
        rows.sort(key=lambda item: item.logical_version)
        self._evidence_ids.add(event.evidence_id)
        if previous_latest is None:
            action = "DELETE" if event.operation is TemporalOperation.DELETE else "ADD"
        elif event.operation is TemporalOperation.DELETE and event.logical_version >= previous_latest.logical_version:
            action = "DELETE"
        elif event.logical_version > previous_latest.logical_version:
            action = "UPDATE"
        else:
            action = "LATE_EVENT"
        self.audit_log.append(
            TemporalAuditEntry(action, event.evidence_id, event.key, event.valid_session)
        )
        self._trim_key(event.key)
        self._enforce_capacity()
        return action

    def _trim_key(self, key: tuple[str, str]) -> None:
        rows = self._versions[key]
        while len(rows) > self.max_versions_per_key:
            removed = rows.pop(0)
            self.audit_log.append(
                TemporalAuditEntry(
                    "COMPACT",
                    removed.evidence_id,
                    removed.key,
                    removed.valid_session,
                )
            )

    @staticmethod
    def _retention_score(
        event: TemporalMemoryEvent,
        *,
        latest: bool,
        newest_session: int,
    ) -> float:
        return (
            event.priority
            + (2.0 if latest else 0.0)
            + (0.25 if latest and event.operation is TemporalOperation.DELETE else 0.0)
            + 0.001 * event.valid_session / max(1, newest_session)
        )

    def _enforce_capacity(self) -> None:
        while len(self) > self.capacity:
            newest = max(
                (event.valid_session for rows in self._versions.values() for event in rows),
                default=1,
            )
            candidates = []
            for key, rows in self._versions.items():
                for index, event in enumerate(rows):
                    candidates.append(
                        (
                            self._retention_score(
                                event,
                                latest=index == len(rows) - 1,
                                newest_session=newest,
                            ),
                            event.logical_version,
                            key,
                            index,
                            event,
                        )
                    )
            _, _, key, index, event = min(candidates, key=lambda row: (row[0], row[1]))
            del self._versions[key][index]
            if not self._versions[key]:
                del self._versions[key]
            self.audit_log.append(
                TemporalAuditEntry(
                    "EVICT",
                    event.evidence_id,
                    event.key,
                    event.valid_session,
                )
            )

    @staticmethod
    def _select(
        rows: list[TemporalMemoryEvent],
        *,
        mode: RecallMode,
        as_of_session: int | None,
    ) -> TemporalMemoryEvent | None:
        if not rows:
            return None
        if mode == "current":
            return rows[-1]
        if mode == "as_of":
            if as_of_session is None:
                raise ValueError("as_of_session is required for as_of recall")
            eligible = [event for event in rows if event.valid_session <= as_of_session]
            return eligible[-1] if eligible else None
        if mode == "previous":
            current = rows[-1]
            current_value = (
                current.value
                if current.operation is TemporalOperation.UPSERT
                else None
            )
            for event in reversed(rows[:-1]):
                if event.operation is TemporalOperation.DELETE or event.value is None:
                    continue
                if current_value is None or event.value != current_value:
                    return event
            return None
        raise ValueError(f"unsupported recall mode: {mode}")

    def recall(
        self,
        subject: str,
        relation: str,
        *,
        mode: RecallMode = "current",
        as_of_session: int | None = None,
    ) -> TemporalMemoryRecall:
        self.recall_count += 1
        rows = self._versions.get((subject, relation), [])
        cost = 1 if mode == "current" else max(1, len(rows))
        event = self._select(rows, mode=mode, as_of_session=as_of_session)
        if (
            event is None
            or event.operation is TemporalOperation.DELETE
            or event.value is None
        ):
            return TemporalMemoryRecall.unknown(cost=cost)
        return TemporalMemoryRecall(
            event.value,
            event.evidence_id,
            event.valid_session,
            False,
            cost,
        )
