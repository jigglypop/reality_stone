"""Fixed synthetic interference benchmark for audited episodic memory."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch

from .episodic_memory import AuditedEpisodicMemory
from .runtime import HippocampusMemory


@dataclass(frozen=True)
class EpisodicMemoryBenchConfig:
    dim: int = 32
    concepts: int = 8
    interference: int = 4
    capacity: int = 12
    seeds: int = 32
    key_noise: float = 0.025


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm().clamp(min=1e-12)


def _lcb(values: list[float], *, seed: int = 20260811, draws: int = 3000) -> float:
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return means[max(0, int(0.025 * draws) - 1)]


class _FifoMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.rows: list[tuple[torch.Tensor, int, str]] = []

    def add(self, key: torch.Tensor, value: int, evidence: str) -> None:
        if len(self.rows) >= self.capacity:
            self.rows.pop(0)
        self.rows.append((key, value, evidence))

    def recall(self, cue: torch.Tensor) -> tuple[int | None, str | None, bool]:
        if not self.rows:
            return None, None, True
        scores = [float(key @ cue) for key, _, _ in self.rows]
        index = max(range(len(scores)), key=scores.__getitem__)
        if scores[index] < 0.60:
            return None, None, True
        return self.rows[index][1], self.rows[index][2], False


def _seed_result(seed: int, cfg: EpisodicMemoryBenchConfig) -> dict[str, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    bases = torch.linalg.qr(torch.randn(cfg.dim, cfg.concepts + cfg.interference, generator=generator)).Q.T
    candidate = AuditedEpisodicMemory(cfg.dim, cfg.capacity)
    merge_off = AuditedEpisodicMemory(cfg.dim, cfg.capacity, merge_updates=False)
    abstention_off = AuditedEpisodicMemory(cfg.dim, cfg.capacity, abstention_enabled=False)
    existing = HippocampusMemory(cfg.dim, capacity=cfg.capacity)
    fifo = _FifoMemory(cfg.capacity)

    latest: list[tuple[torch.Tensor, int, str]] = []
    timestamp = 0
    for index in range(cfg.concepts):
        key = _normalize(bases[index] + cfg.key_noise * torch.randn(cfg.dim, generator=generator))
        evidence = f"s{seed}-c{index}-old"
        for memory in (candidate, merge_off, abstention_off):
            memory.upsert(key, index, evidence, priority=2.0, timestamp=timestamp)
        existing.encode(key, torch.nn.functional.one_hot(torch.tensor(index), cfg.concepts).float(), priority=2.0)
        fifo.add(key, index, evidence)
        timestamp += 1
    for index in range(cfg.concepts):
        key = _normalize(bases[index] + cfg.key_noise * torch.randn(cfg.dim, generator=generator))
        value = cfg.concepts - 1 - index
        evidence = f"s{seed}-c{index}-new"
        latest.append((bases[index], value, evidence))
        for memory in (candidate, merge_off, abstention_off):
            memory.upsert(key, value, evidence, priority=1.0, timestamp=timestamp)
        existing.encode(key, torch.nn.functional.one_hot(torch.tensor(value), cfg.concepts).float(), priority=1.0)
        fifo.add(key, value, evidence)
        timestamp += 1
    for offset in range(cfg.interference):
        key = bases[cfg.concepts + offset]
        evidence = f"s{seed}-d{offset}"
        for memory in (candidate, merge_off, abstention_off):
            memory.upsert(key, 0, evidence, priority=1.5, timestamp=timestamp)
        existing.encode(key, torch.nn.functional.one_hot(torch.tensor(0), cfg.concepts).float(), priority=1.5)
        fifo.add(key, 0, evidence)
        timestamp += 1

    metrics: dict[str, dict[str, float]] = {}
    for name in ("candidate", "merge_off", "abstention_off", "existing", "fifo", "no_memory"):
        correct_value = correct_evidence = 0
        for cue, value, evidence in latest:
            query = _normalize(cue + cfg.key_noise * torch.randn(cfg.dim, generator=generator))
            if name in ("candidate", "merge_off", "abstention_off"):
                recalled = {"candidate": candidate, "merge_off": merge_off, "abstention_off": abstention_off}[name].recall(query)
                got_value, got_evidence = recalled.value, recalled.evidence_id
            elif name == "fifo":
                got_value, got_evidence, _ = fifo.recall(query)
            elif name == "existing":
                vector = existing.recall(query, topk=1)
                got_value = None if vector.numel() != cfg.concepts or vector.norm().item() == 0.0 else int(vector.argmax().item())
                got_evidence = None
            else:
                got_value = got_evidence = None
            correct_value += int(got_value == value)
            correct_evidence += int(got_evidence == evidence)

        unseen = _normalize(torch.randn(cfg.dim, generator=generator))
        unseen = _normalize(unseen - sum(float(unseen @ base) * base for base in bases))
        if name in ("candidate", "merge_off", "abstention_off"):
            abstained = {"candidate": candidate, "merge_off": merge_off, "abstention_off": abstention_off}[name].recall(unseen).abstained
        elif name == "fifo":
            _, _, abstained = fifo.recall(unseen)
        elif name == "existing":
            abstained = existing.recall(unseen, topk=1).norm().item() == 0.0
        else:
            abstained = True

        delete_correct = 0.0
        if name == "candidate":
            target_cue, _, target_evidence = latest[0]
            deleted = candidate.delete(target_evidence)
            delete_correct = float(deleted and candidate.recall(target_cue).abstained)
        latest_acc = correct_value / cfg.concepts
        evidence_acc = correct_evidence / cfg.concepts
        composite = (latest_acc + evidence_acc + float(abstained) + delete_correct) / 4.0
        metrics[name] = {
            "latest_accuracy": latest_acc,
            "evidence_accuracy": evidence_acc,
            "abstention": float(abstained),
            "delete_correct": delete_correct,
            "composite": composite,
        }
    metrics["candidate"]["capacity_ok"] = float(len(candidate) <= cfg.capacity)
    metrics["candidate"]["update_audit_count"] = float(sum(row["operation"] == "UPDATE" for row in candidate.audit_log))
    metrics["candidate"]["delete_audit_count"] = float(sum(row["operation"] == "DELETE" for row in candidate.audit_log))
    return metrics


def evaluate_episodic_memory(config: EpisodicMemoryBenchConfig | None = None) -> dict[str, object]:
    cfg = config or EpisodicMemoryBenchConfig()
    rows = [_seed_result(998000 + index, cfg) for index in range(cfg.seeds)]
    names = ("candidate", "merge_off", "abstention_off", "existing", "fifo", "no_memory")
    means = {
        name: {
            metric: sum(row[name][metric] for row in rows) / len(rows)
            for metric in rows[0][name]
        }
        for name in names
    }
    comparisons = {
        name: _lcb([row["candidate"]["composite"] - row[name]["composite"] for row in rows], seed=20260811 + i)
        for i, name in enumerate(("existing", "merge_off", "fifo"))
    }
    candidate = means["candidate"]
    false_recall_gap = means["candidate"]["abstention"] - means["abstention_off"]["abstention"]
    hard_gate = bool(
        candidate["latest_accuracy"] >= 0.90
        and candidate["evidence_accuracy"] >= 0.90
        and candidate["abstention"] >= 0.95
        and candidate["delete_correct"] >= 0.95
        and all(value > 0.10 for value in comparisons.values())
        and false_recall_gap >= 0.20
        and candidate["capacity_ok"] == 1.0
        and candidate["update_audit_count"] == cfg.concepts
        and candidate["delete_audit_count"] == 1.0
    )
    return {
        "schema": "clarus.episodic-memory.validation.v1",
        "config": asdict(cfg),
        "means": means,
        "lcb_candidate_composite_minus": comparisons,
        "abstention_advantage_vs_off": false_recall_gap,
        "hard_gate": hard_gate,
        "promisingness_score": 90.0 if hard_gate else 0.0,
        "grade": "GO" if hard_gate else "STOP",
        "claim_limit": "synthetic bounded-capacity key/value memory mechanics only",
    }


__all__ = ["EpisodicMemoryBenchConfig", "evaluate_episodic_memory"]
