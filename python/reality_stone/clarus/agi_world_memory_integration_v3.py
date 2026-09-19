"""Locked G9-CBM/V3 episodic-memory/world-model integration experiment.

The module deliberately keeps candidate code separate from evaluator truth.  V3
is a boundary-only amendment to the byte-locked V2 registration: it fixes the
``PartialCue`` field names and the positive/lure cue chronology, while every
other scientific literal is inherited by a checked recursive merge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from reality_stone.clarus import episodic_ltm_dream_bridge as g7m_v1
from reality_stone.clarus import episodic_ltm_dream_bridge_v2 as g7m_v2


REGISTERED_CONFIG_SHA256 = "bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b"
_BASE_CONFIG_SHA256 = "b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2"
_CONTRACT_SHA256 = "842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70"
_AMENDMENT_SHA256 = "9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340"
_SOURCE = np.asarray((3, 0, 1, 2), dtype=np.int64)
_ACTIONS = np.asarray(((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)), dtype=np.float64)
_OBSERVED = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0))
_MISSING = ((0, 2), (1, 0), (2, 1))
_D = np.diag((0.55, 0.50, 0.45, 0.40)).astype(np.float64)
_B = np.asarray(
    ((0.0, 0.0, 0.0, 0.05), (0.08, 0.0, 0.0, 0.0), (0.0, -0.07, 0.0, 0.0), (0.0, 0.0, 0.06, 0.0)),
    dtype=np.float64,
)
_G = np.asarray(((0.28, 0.0), (0.0, 0.28), (-0.14, 0.10), (0.10, -0.14)), dtype=np.float64)
_V2_V3_REGISTERED = frozenset(
    (
        *range(86100, 86140),
        *range(87100, 87140),
        *range(88100, 88160),
        *range(92100, 92140),
        *range(93100, 93140),
        *range(94100, 94160),
    )
)

Provenance = g7m_v1.Provenance
EpisodicRecord = g7m_v2.EpisodicRecord
PartialCue = g7m_v2.PartialCue
RecallResult = g7m_v2.RecallResult
PersistentEpisodicStore = g7m_v2.PersistentEpisodicStore
CoordinateStandardizer = g7m_v2.CoordinateStandardizer
REAL_PROVENANCE = g7m_v2.REAL_PROVENANCE
RECALLED_PROVENANCE = g7m_v2.RECALLED_PROVENANCE
SYNTHETIC_PROVENANCE = g7m_v2.SYNTHETIC_PROVENANCE
FALLBACK_PROVENANCE = g7m_v2.FALLBACK_PROVENANCE


def _f64(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite float64{shape}")
    result.setflags(write=False)
    return result


def _typed(value: Any, dtype: np.dtype[Any], shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result.setflags(write=False)
    return result


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes_v3(value: object) -> bytes:
    """Canonical artifact bytes: UTF-8, sorted keys, indent two, one LF."""

    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json_lf_v3(path: Path, value: object) -> None:
    """Create one canonical artifact and refuse every overwrite."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes_v3(value))
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _canonical_locked_json(raw: bytes, label: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise PermissionError(f"{label} violates UTF-8/LF transport")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise PermissionError(f"{label} must have exactly one terminal LF")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _root(config_path: Path) -> Path:
    return Path(config_path).resolve().parents[2]


def _delete_exact(target: MutableMapping[str, Any], path: Sequence[str]) -> None:
    cursor: MutableMapping[str, Any] = target
    for segment in path[:-1]:
        value = cursor.get(segment)
        if not isinstance(value, dict):
            raise KeyError(f"missing delete segment: {list(path)!r}")
        cursor = value
    leaf = path[-1]
    if leaf not in cursor:
        raise KeyError(f"missing delete leaf: {list(path)!r}")
    del cursor[leaf]


def _merge_overrides(
    target: MutableMapping[str, Any],
    override: Mapping[str, Any],
    allowed_new: set[tuple[str, ...]],
    prefix: tuple[str, ...] = (),
) -> None:
    for key, value in override.items():
        path = (*prefix, key)
        if key not in target and path not in allowed_new:
            raise KeyError(f"unregistered new override path: {list(path)!r}")
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_overrides(target[key], value, allowed_new, path)
        else:
            target[key] = copy.deepcopy(value)


def load_merged_registration_v3(config_path: Path) -> dict[str, Any]:
    """Verify and deterministically materialize the V3-over-V2 registration."""

    config_path = Path(config_path)
    raw = config_path.read_bytes()
    if _sha(raw) != REGISTERED_CONFIG_SHA256:
        raise PermissionError("G9-CBM/V3 raw registration SHA256 changed")
    v3 = _canonical_locked_json(raw, "V3 registration")
    if v3.get("experiment") != "agi_world_memory_integration_v3":
        raise ValueError("G9-CBM/V3 registration required")
    root = _root(config_path)
    integrity = v3["amendment_integrity"]
    checks = (
        (root / integrity["base_registration_path"], _BASE_CONFIG_SHA256, "base registration"),
        (root / integrity["base_contract_path"], _CONTRACT_SHA256, "base contract"),
        (root / integrity["path"], _AMENDMENT_SHA256, "V3 amendment"),
    )
    for path, expected, label in checks:
        if _sha(path.read_bytes()) != expected:
            raise PermissionError(f"{label} SHA256 changed")
    base = json.loads(checks[0][0].read_bytes())
    merged = copy.deepcopy(base)
    for path in v3["delete_paths"]:
        _delete_exact(merged, path)
    allowed = {tuple(item) for item in v3["merge_semantics"]["allowed_new_override_paths"]}
    _merge_overrides(merged, v3["overrides"], allowed)
    for key, value in v3.items():
        if key != "overrides":
            merged[key] = copy.deepcopy(value)
    return merged


def merged_registration_sha256_v3(config_path: Path) -> str:
    merged = load_merged_registration_v3(config_path)
    raw = json.dumps(
        merged, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha(raw)


@dataclass(frozen=True)
class CoreModelV2:
    intercept: np.ndarray
    diagonal: np.ndarray
    bridge: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "intercept", _f64(self.intercept, (4,), "intercept"))
        object.__setattr__(self, "diagonal", _f64(self.diagonal, (4,), "diagonal"))
        object.__setattr__(self, "bridge", _f64(self.bridge, (4,), "bridge"))
        object.__setattr__(self, "action", _f64(self.action, (4, 2), "action"))

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64)
        a = np.asarray(action, dtype=np.float64)
        if x.shape != (4,) or a.shape != (2,):
            raise ValueError("state/action shapes must be (4,)/(2,)")
        return (
            self.intercept + self.diagonal * x + self.bridge * np.tanh(x[_SOURCE]) + self.action @ a
        )

    def ordered_vector(self) -> np.ndarray:
        return np.concatenate((self.intercept, self.diagonal, self.bridge, self.action.reshape(-1)))


@dataclass(frozen=True)
class CostSpecV2:
    mu_x: np.ndarray
    sigma_x: np.ndarray
    action_cost_weight: float = 0.02
    success_threshold: float = 25.0
    invalid_penalty: float = 10000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu_x", _f64(self.mu_x, (4,), "mu_x"))
        object.__setattr__(self, "sigma_x", _f64(self.sigma_x, (4,), "sigma_x"))
        if np.any(self.sigma_x < 0.05):
            raise ValueError("sigma_x violates 0.05 floor")
        if (self.action_cost_weight, self.success_threshold, self.invalid_penalty) != (
            0.02,
            25.0,
            10000.0,
        ):
            raise ValueError("registered cost literals changed")


@dataclass(frozen=True)
class CodecSpecV2:
    mu_codec: np.ndarray
    sigma_codec: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu_codec", _f64(self.mu_codec, (96,), "mu_codec"))
        object.__setattr__(self, "sigma_codec", _f64(self.sigma_codec, (96,), "sigma_codec"))
        if np.any(self.sigma_codec < 1e-8):
            raise ValueError("sigma_codec violates 1e-8 floor")

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        value = _f64(raw, (12, 8), "raw codec")
        return ((value.reshape(-1, order="C") - self.mu_codec) / self.sigma_codec).reshape(
            (12, 8), order="C"
        )

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        value = _f64(standardized, (12, 8), "standardized codec")
        return (self.mu_codec + self.sigma_codec * value.reshape(-1, order="C")).reshape(
            (12, 8), order="C"
        )


@dataclass(frozen=True)
class OriginRecallAuditV2:
    accepted: bool
    identity: np.int16
    confidence: np.float64
    scope: np.uint8

    def __post_init__(self) -> None:
        identity, confidence, scope = int(self.identity), float(self.confidence), int(self.scope)
        if scope not in (0, 1, 2):
            raise ValueError("invalid scope code")
        if scope in (0, 2):
            if self.accepted or identity != -1 or confidence != -2.0:
                raise ValueError("disabled/invalid scope sentinel mismatch")
        elif self.accepted != (0 <= identity <= 95):
            raise ValueError("accepted/identity invariant failed")
        elif not self.accepted and identity != -1:
            raise ValueError("rejected identity must be -1")
        if scope == 1 and (not math.isfinite(confidence) or not -1.0 <= confidence <= 1.0):
            raise ValueError("scope-one confidence outside [-1,1]")


@dataclass(frozen=True)
class SeedRecallAuditV2:
    accepted: np.ndarray
    identity: np.ndarray
    confidence: np.ndarray
    scope: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "accepted", _typed(self.accepted, np.dtype(bool), (72,), "accepted")
        )
        object.__setattr__(
            self, "identity", _typed(self.identity, np.dtype(np.int16), (72,), "identity")
        )
        object.__setattr__(self, "confidence", _f64(self.confidence, (72,), "confidence"))
        object.__setattr__(self, "scope", _typed(self.scope, np.dtype(np.uint8), (72,), "scope"))


@dataclass(frozen=True)
class ScopedRecallIndexV3:
    indices: Mapping[tuple[str, int], np.ndarray]
    component_by_prefix: Mapping[tuple[str, str], int]
    component_by_suffix: Mapping[tuple[str, str], int]

    @classmethod
    def from_store(cls, store: PersistentEpisodicStore) -> "ScopedRecallIndexV3":
        components = g7m_v1.infer_cooccurrence_components(store.records)
        groups: dict[tuple[str, int], list[int]] = {}
        for index, record in enumerate(store.records):
            component = components.prefix_component[(record.context_token, record.prefix_token)]
            groups.setdefault((record.context_token, component), []).append(index)
        frozen: dict[tuple[str, int], np.ndarray] = {}
        for key, values in groups.items():
            if len(values) != 12:
                raise ValueError("every scoped recall view must contain 12 records")
            frozen[key] = _typed(values, np.dtype(np.int16), (12,), "scope indices")
        return cls(frozen, dict(components.prefix_component), dict(components.suffix_component))

    def resolve(self, cue: PartialCue) -> np.ndarray | None:
        left = self.component_by_prefix.get((cue.context_token, cue.prefix_token))
        right = self.component_by_suffix.get((cue.context_token, cue.suffix_token))
        if left is None or right is None or left != right:
            return None
        return self.indices.get((cue.context_token, left))


def _empty_recall_v3() -> RecallResult:
    return RecallResult(
        False,
        None,
        np.zeros((12, 8), dtype=np.float64),
        -math.inf,
        0,
        True,
        True,
        0.0,
        FALLBACK_PROVENANCE,
    )


def scoped_hard_recall_v3(
    store: PersistentEpisodicStore,
    cue: PartialCue,
    scope_index: ScopedRecallIndexV3,
    *,
    enabled: bool = True,
) -> RecallResult:
    """Call inherited hard recall once on exactly one 12-record zero-copy scope."""

    if not enabled:
        return _empty_recall_v3()
    indices = scope_index.resolve(cue)
    if indices is None:
        return _empty_recall_v3()
    facade = PersistentEpisodicStore(store.standardizer, capacity=12, threshold=store.threshold)
    # The inherited store has no public zero-copy constructor.  Its private list
    # is populated with existing immutable record references; trace payloads are
    # neither copied nor inserted.
    facade._records = [store.records[int(index)] for index in indices]  # type: ignore[attr-defined]
    return g7m_v2.hard_cue_anchored_recall(facade, cue)


def codec_residual_view_v3(
    codec: np.ndarray,
    codec_spec: CodecSpecV2,
    *,
    standardized: bool = False,
) -> np.ndarray:
    """The sole raw/standardized boundary; fingerprint columns never cross."""

    value = _f64(codec, (12, 8), "codec")
    raw = codec_spec.inverse(value) if standardized else value
    result = np.array(raw[:, :4], dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ResidualDreamBindingV3:
    key: np.int16
    standardized_residual: np.ndarray
    left_join_rms: np.float64
    right_join_rms: np.float64
    provenance: Provenance = SYNTHETIC_PROVENANCE


@dataclass
class DreamAuditV3:
    pair_check_flags: np.ndarray = field(default_factory=lambda: np.zeros(288, dtype=bool))
    pair_reason_codes: np.ndarray = field(default_factory=lambda: np.zeros(288, dtype=np.uint8))
    endpoint_join_values: np.ndarray = field(
        default_factory=lambda: np.full(48, np.nan, dtype=np.float64)
    )
    output_occupancy: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=bool))
    output_provenance: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.uint8))
    ordered_pair_indices: np.ndarray = field(
        default_factory=lambda: np.full((288, 2), -1, dtype=np.int16)
    )
    observed_overwrite_count: int = 0


class ResidualSchemaTableV3:
    """Preallocated 72-key residual-only slow-schema table."""

    def __init__(self, codec_spec: CodecSpecV2) -> None:
        self.codec_spec = codec_spec
        self.payload = np.zeros((72, 12, 8), dtype=np.float64)
        self.observed = np.zeros(72, dtype=bool)
        self.synthetic = np.zeros(72, dtype=bool)
        self.provenance = np.zeros(72, dtype=np.uint8)
        self.context_tokens: list[str] = []
        self.prefix_tokens: dict[tuple[int, int, int], str] = {}
        self.suffix_tokens: dict[tuple[int, int, int], str] = {}

    @staticmethod
    def key_index(context: int, component: int, prefix: int, suffix: int) -> int:
        if not (0 <= context < 2 and 0 <= component < 4 and 0 <= prefix < 3 and 0 <= suffix < 3):
            raise KeyError("schema key outside registered range")
        return ((context * 4 + component) * 3 + prefix) * 3 + suffix

    def resolve_tokens(
        self, context: str, prefix: str, suffix: str
    ) -> tuple[int, int, int, int] | None:
        try:
            c = self.context_tokens.index(context)
        except ValueError:
            return None
        left = next(
            (
                (p, i)
                for (token_context, p, i), token in self.prefix_tokens.items()
                if token_context == c and token == prefix
            ),
            None,
        )
        right = next(
            (
                (p, j)
                for (token_context, p, j), token in self.suffix_tokens.items()
                if token_context == c and token == suffix
            ),
            None,
        )
        if left is None or right is None or left[0] != right[0]:
            return None
        return c, left[0], left[1], right[1]

    def fallback_standardized(self, context: int, component: int) -> np.ndarray:
        keys = [self.key_index(context, component, i, j) for i, j in _OBSERVED]
        if not np.all(self.observed[keys]):
            raise ValueError("component fallback requires six observed keys")
        return np.mean(self.payload[keys], axis=0, dtype=np.float64)

    def lookup_residual_raw(
        self, context: int, component: int, prefix: int, suffix: int
    ) -> tuple[np.ndarray, np.int16, np.uint8]:
        key = self.key_index(context, component, prefix, suffix)
        if self.observed[key]:
            payload, source = self.payload[key], 1
        elif self.synthetic[key]:
            payload, source = self.payload[key], 2
        else:
            payload, source = self.fallback_standardized(context, component), 3
        return (
            codec_residual_view_v3(payload, self.codec_spec, standardized=True),
            np.int16(key),
            np.uint8(source),
        )

    def insert_synthetic(self, key: int, residual: np.ndarray) -> bool:
        if self.observed[key] or self.synthetic[key]:
            return False
        value = _f64(residual, (12, 4), "synthetic residual")
        self.payload[key, :, :4] = value
        self.payload[key, :, 4:] = 0.0
        self.synthetic[key] = True
        self.provenance[key] = 2
        return True


def constrained_residual_completion_v3(
    schema: ResidualSchemaTableV3,
    join_threshold: float,
    *,
    write_enabled: bool = True,
    audit: DreamAuditV3 | None = None,
) -> tuple[ResidualDreamBindingV3, ...]:
    """One 288-pair residual-only pass with immutable observed entries."""

    if not math.isfinite(join_threshold):
        raise ValueError("join threshold must be finite")
    audit = audit if audit is not None else DreamAuditV3()
    observed_hash = _sha(np.ascontiguousarray(schema.payload[schema.observed]).tobytes())
    proposals: list[ResidualDreamBindingV3] = []
    pair = 0
    join_slot = 0
    output_slot = 0
    for context in range(2):
        for prefix_slot in range(12):
            p_component, prefix = divmod(prefix_slot, 3)
            for suffix_slot in range(12):
                s_component, suffix = divmod(suffix_slot, 3)
                audit.ordered_pair_indices[pair] = (prefix_slot, suffix_slot)
                audit.pair_check_flags[pair] = True
                if p_component != s_component:
                    audit.pair_reason_codes[pair] = 2
                else:
                    key = schema.key_index(context, p_component, prefix, suffix)
                    if schema.observed[key]:
                        audit.pair_reason_codes[pair] = 3
                    else:
                        keys_prefix = [
                            schema.key_index(context, p_component, prefix, j)
                            for i, j in _OBSERVED
                            if i == prefix
                        ]
                        keys_suffix = [
                            schema.key_index(context, p_component, i, suffix)
                            for i, j in _OBSERVED
                            if j == suffix
                        ]
                        prefix_part = np.mean(schema.payload[keys_prefix, 0:5, :4], axis=0)
                        connector = np.mean(
                            schema.payload[
                                [
                                    schema.key_index(context, p_component, i, j)
                                    for i, j in _OBSERVED
                                ],
                                5:7,
                                :4,
                            ],
                            axis=0,
                        )
                        suffix_part = np.mean(schema.payload[keys_suffix, 7:12, :4], axis=0)
                        proposal = np.concatenate((prefix_part, connector, suffix_part), axis=0)
                        left = float(np.sqrt(np.mean((proposal[4] - proposal[5]) ** 2)))
                        right = float(np.sqrt(np.mean((proposal[6] - proposal[7]) ** 2)))
                        audit.endpoint_join_values[2 * join_slot : 2 * join_slot + 2] = (
                            left,
                            right,
                        )
                        join_slot += 1
                        if left > join_threshold:
                            audit.pair_reason_codes[pair] = 5
                        elif right > join_threshold:
                            audit.pair_reason_codes[pair] = 6
                        else:
                            audit.pair_reason_codes[pair] = 4
                            binding = ResidualDreamBindingV3(
                                np.int16(key), proposal, np.float64(left), np.float64(right)
                            )
                            proposals.append(binding)
                            audit.output_occupancy[output_slot] = True
                            audit.output_provenance[output_slot] = 1
                            if write_enabled:
                                schema.insert_synthetic(key, proposal)
                        output_slot += 1
                pair += 1
    if pair != 288 or join_slot != 24 or output_slot != 24 or not np.all(audit.pair_check_flags):
        raise AssertionError("registered dream accounting changed")
    if observed_hash != _sha(np.ascontiguousarray(schema.payload[schema.observed]).tobytes()):
        audit.observed_overwrite_count += 1
        raise AssertionError("observed schema changed")
    return tuple(proposals)


@dataclass(frozen=True)
class WakeActionValueV3:
    context_token: str
    component: int
    numeric_action: np.ndarray
    suffix_token: str
    suffix_local: int


class WakeActionIndexV3:
    def __init__(self, values: Mapping[str, WakeActionValueV3]) -> None:
        self._values = dict(values)

    def resolve(self, token: object) -> WakeActionValueV3 | None:
        return self._values.get(str(token))


WakeActionIndexV2 = WakeActionIndexV3


@dataclass(frozen=True)
class ScopedEpisodicFacadeV3:
    store: PersistentEpisodicStore
    scope_index: ScopedRecallIndexV3
    enabled: bool = True


@dataclass(frozen=True)
class CandidateRequestV2:
    cue: PartialCue
    anchor_state: np.ndarray
    numeric_actions: np.ndarray
    action_tokens: np.ndarray
    public_goal: np.ndarray
    cost_spec: CostSpecV2
    codec_spec: CodecSpecV2
    core: CoreModelV2
    action_index: WakeActionIndexV3
    schema: ResidualSchemaTableV3
    episodic_store: ScopedEpisodicFacadeV3 | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_state", _f64(self.anchor_state, (4,), "anchor_state"))
        object.__setattr__(
            self, "numeric_actions", _f64(self.numeric_actions, (8, 20, 2), "numeric_actions")
        )
        tokens = np.array(self.action_tokens, dtype=object, order="C", copy=True)
        if tokens.shape != (8, 20):
            raise ValueError("action_tokens must have shape (8,20)")
        tokens.setflags(write=False)
        object.__setattr__(self, "action_tokens", tokens)
        object.__setattr__(self, "public_goal", _f64(self.public_goal, (20, 4), "public_goal"))


@dataclass(frozen=True)
class CandidateResultV2:
    predictions: np.ndarray
    inferred_valid: np.ndarray
    resolved_schema_keys: np.ndarray
    schema_sources: np.ndarray
    inferred_costs: np.ndarray
    selected_index: np.int64
    origin_recall_audit: OriginRecallAuditV2

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions", _f64(self.predictions, (8, 20, 4), "predictions"))
        object.__setattr__(
            self,
            "inferred_valid",
            _typed(self.inferred_valid, np.dtype(bool), (8, 20), "inferred_valid"),
        )
        object.__setattr__(
            self,
            "resolved_schema_keys",
            _typed(self.resolved_schema_keys, np.dtype(np.int16), (8, 20), "resolved_schema_keys"),
        )
        object.__setattr__(
            self,
            "schema_sources",
            _typed(self.schema_sources, np.dtype(np.uint8), (8, 20), "schema_sources"),
        )
        object.__setattr__(
            self, "inferred_costs", _f64(self.inferred_costs, (8,), "inferred_costs")
        )
        if not 0 <= int(self.selected_index) < 8:
            raise ValueError("selected_index outside 0..7")
        if np.any((self.resolved_schema_keys < -1) | (self.resolved_schema_keys > 71)):
            raise ValueError("schema key code outside -1..71")
        if np.any(self.schema_sources > 3) or not np.array_equal(
            self.resolved_schema_keys == -1, self.schema_sources == 0
        ):
            raise ValueError("schema key/source invariant failed")


def _recall_completion_and_audit(
    request: CandidateRequestV2,
) -> tuple[np.ndarray, OriginRecallAuditV2]:
    cue = request.cue
    values = np.asarray(cue.cue_values, dtype=np.float64)
    mask = np.asarray(cue.cue_mask, dtype=bool)
    if values.shape != (12, 8) or mask.shape != (12, 8) or int(np.sum(mask)) != 24:
        raise ValueError("V3 cue must be (12,8) with 24 visible cells")
    fallback = request.codec_spec.mu_codec.reshape((12, 8), order="C").copy()
    fallback[mask] = values[mask]
    facade = request.episodic_store
    if facade is None or not facade.enabled:
        return fallback, OriginRecallAuditV2(False, np.int16(-1), np.float64(-2), np.uint8(0))
    indices = facade.scope_index.resolve(cue)
    if indices is None:
        return fallback, OriginRecallAuditV2(False, np.int16(-1), np.float64(-2), np.uint8(2))
    recalled = scoped_hard_recall_v3(facade.store, cue, facade.scope_index, enabled=True)
    identity = -1
    if recalled.accepted:
        identity = next(
            (
                i
                for i, item in enumerate(facade.store.records)
                if item.episode_id == recalled.episode_id
            ),
            -1,
        )
        completed = np.asarray(recalled.reconstruction, dtype=np.float64)
    else:
        completed = fallback
    audit = OriginRecallAuditV2(
        bool(recalled.accepted), np.int16(identity), np.float64(recalled.confidence), np.uint8(1)
    )
    return completed, audit


def _rollout_from_completion_v3(
    request: CandidateRequestV2,
    completed: np.ndarray,
    recall_audit: OriginRecallAuditV2,
    *,
    zero_q: bool = False,
    synthetic_fallback: bool = False,
) -> CandidateResultV2:
    """Internal control surface; evaluator truth is never accepted here."""

    completed = _f64(completed, (12, 8), "completed cue")
    resolved = request.schema.resolve_tokens(
        request.cue.context_token, request.cue.prefix_token, request.cue.suffix_token
    )
    if resolved is None:
        raise ValueError("anchor cue tokens do not resolve")
    context, component, prefix, anchor_suffix = resolved
    anchor_raw, _, _ = request.schema.lookup_residual_raw(context, component, prefix, anchor_suffix)
    q_hat = np.mean(
        codec_residual_view_v3(completed, request.codec_spec) - anchor_raw, axis=0, dtype=np.float64
    )
    if zero_q:
        q_hat = np.zeros(4, dtype=np.float64)
    predictions = np.empty((8, 20, 4), dtype=np.float64)
    inferred_valid = np.empty((8, 20), dtype=bool)
    keys = np.empty((8, 20), dtype=np.int16)
    sources = np.empty((8, 20), dtype=np.uint8)
    costs = np.empty(8, dtype=np.float64)
    for candidate in range(8):
        state = np.array(request.anchor_state, copy=True)
        valid_sequence = True
        for lead in range(20):
            numeric = request.numeric_actions[candidate, lead]
            action_value = request.action_index.resolve(request.action_tokens[candidate, lead])
            valid = bool(
                action_value is not None
                and action_value.context_token == request.cue.context_token
                and action_value.component == component
                and np.array_equal(action_value.numeric_action, numeric)
            )
            if valid:
                schema_raw, key, source = request.schema.lookup_residual_raw(
                    context, component, prefix, action_value.suffix_local
                )
                if synthetic_fallback and int(source) == 2:
                    schema_raw = codec_residual_view_v3(
                        request.schema.fallback_standardized(context, component),
                        request.codec_spec,
                        standardized=True,
                    )
                    source = np.uint8(3)
            else:
                fallback_std = request.schema.fallback_standardized(context, component)
                schema_raw = codec_residual_view_v3(
                    fallback_std, request.codec_spec, standardized=True
                )
                key, source = np.int16(-1), np.uint8(0)
            state = request.core.predict(state, numeric) + q_hat + schema_raw[lead % 12]
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("candidate produced nonfinite prediction")
            predictions[candidate, lead] = state
            inferred_valid[candidate, lead] = valid
            keys[candidate, lead] = key
            sources[candidate, lead] = source
            valid_sequence &= valid
        costs[candidate] = planning_cost_v3(
            predictions[candidate],
            request.numeric_actions[candidate],
            request.public_goal,
            request.cost_spec,
            valid=valid_sequence,
        )
    selected = np.int64(np.argmin(costs))
    return CandidateResultV2(
        predictions, inferred_valid, keys, sources, costs, selected, recall_audit
    )


def execute_candidate_v3(request: CandidateRequestV2) -> CandidateResultV2:
    """Execute the sealed candidate; evaluator truth is not a parameter."""

    completed, recall_audit = _recall_completion_and_audit(request)
    return _rollout_from_completion_v3(request, completed, recall_audit)


def fit_shared_core_v3(records: Sequence["WakeRecordV3"]) -> CoreModelV2:
    rows = sum(record.actions.shape[0] for record in records)
    if rows == 0:
        raise ValueError("core fit requires wake records")
    coefficients = np.empty((4, 5), dtype=np.float64)
    ridge = np.diag((0.0, 1e-6, 1e-6, 1e-6, 1e-6))
    for target in range(4):
        design = np.empty((rows, 5), dtype=np.float64)
        response = np.empty(rows, dtype=np.float64)
        cursor = 0
        for record in records:
            count = record.actions.shape[0]
            design[cursor : cursor + count, 0] = 1.0
            design[cursor : cursor + count, 1] = record.states[:-1, target]
            design[cursor : cursor + count, 2] = np.tanh(record.states[:-1, _SOURCE[target]])
            design[cursor : cursor + count, 3:5] = record.actions
            response[cursor : cursor + count] = record.states[1:, target]
            cursor += count
        coefficients[target] = np.linalg.solve(design.T @ design + ridge, design.T @ response)
    return CoreModelV2(
        coefficients[:, 0], coefficients[:, 1], coefficients[:, 2], coefficients[:, 3:5]
    )


@dataclass(frozen=True)
class WakeRecordV3:
    episode_id: str
    context: int
    port: int
    prefix: int
    suffix: int
    sign: int
    context_token: str
    prefix_token: str
    suffix_token: str
    action_token: str
    states: np.ndarray
    actions: np.ndarray
    signatures: np.ndarray


@dataclass(frozen=True)
class _PrimitivesV3:
    prefix: np.ndarray
    connector: np.ndarray
    suffix: np.ndarray
    interaction: np.ndarray
    drift: np.ndarray
    fingerprint: np.ndarray


@dataclass(frozen=True)
class SeedWorldV3:
    master_seed: int
    wake_records: tuple[WakeRecordV3, ...]
    primitives: _PrimitivesV3
    context_tokens: tuple[str, str]
    prefix_tokens: np.ndarray
    suffix_tokens: np.ndarray
    action_tokens: np.ndarray


def _rng(master_seed: int, stream_id: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(master_seed), int(stream_id)]))
    )


def _normalize_local(raw: np.ndarray, *, subtract_mean: bool) -> np.ndarray:
    value = np.array(raw, dtype=np.float64, copy=True)
    if subtract_mean:
        value -= np.mean(value, axis=0, keepdims=True)
    denominator = float(np.max(np.abs(value)))
    if denominator <= 1e-12:
        raise FloatingPointError("primitive normalization denominator too small")
    return value / denominator


def _make_tokens(seed: int) -> tuple[tuple[str, str], np.ndarray, np.ndarray, np.ndarray]:
    rng = _rng(seed, 0)

    def token(label: str) -> str:
        return f"{label}:{rng.bytes(16).hex()}"

    contexts = (token("c"), token("c"))
    prefixes = np.empty((2, 4, 3), dtype=object)
    suffixes = np.empty((2, 4, 3), dtype=object)
    actions = np.empty((2, 4, 3), dtype=object)
    for c in range(2):
        for p in range(4):
            for local in range(3):
                prefixes[c, p, local] = token("p")
                suffixes[c, p, local] = token("s")
                actions[c, p, local] = token("a")
    return contexts, prefixes, suffixes, actions


def _generate_seed_world_v3(master_seed: int) -> SeedWorldV3:
    contexts, prefix_tokens, suffix_tokens, action_tokens = _make_tokens(master_seed)
    raw_p = _rng(master_seed, 2).uniform(-1, 1, (2, 4, 3, 4))
    raw_c = _rng(master_seed, 3).uniform(-1, 1, (2, 4, 4))
    raw_s = _rng(master_seed, 4).uniform(-1, 1, (2, 4, 3, 4))
    prefix = np.empty_like(raw_p)
    connector = np.empty_like(raw_c)
    suffix = np.empty_like(raw_s)
    for c in range(2):
        for p in range(4):
            prefix[c, p] = 0.025 * _normalize_local(raw_p[c, p], subtract_mean=True)
            connector[c, p] = 0.010 * _normalize_local(raw_c[c, p], subtract_mean=False)
            suffix[c, p] = 0.025 * _normalize_local(raw_s[c, p], subtract_mean=True)
    interaction = _rng(master_seed, 5).uniform(-0.010, 0.010, (2, 4, 3, 3, 4))
    drift_raw = _rng(master_seed, 6).uniform(-1, 1, (2, 4, 6, 4))
    drift_norm = np.linalg.norm(drift_raw, axis=-1, keepdims=True)
    if np.any(drift_norm <= 1e-12):
        raise FloatingPointError("zero drift")
    drift = 0.040 * drift_raw / drift_norm
    fingerprint_rng = _rng(master_seed, 7)
    fingerprint_raw = fingerprint_rng.uniform(-1, 1, (2, 4, 6, 2, 4))
    fingerprint_norm = np.linalg.norm(fingerprint_raw, axis=-1, keepdims=True)
    if np.any(fingerprint_norm <= 1e-12):
        raise FloatingPointError("zero fingerprint")
    fingerprint = fingerprint_raw / fingerprint_norm
    primitives = _PrimitivesV3(prefix, connector, suffix, interaction, drift, fingerprint)
    initial_rng, noise_rng, signature_rng = (
        _rng(master_seed, 1),
        _rng(master_seed, 8),
        fingerprint_rng,
    )
    records: list[WakeRecordV3] = []
    for c in range(2):
        for p in range(4):
            for binding_index, (i, j) in enumerate(_OBSERVED):
                initial = initial_rng.uniform(-0.25, 0.25, 4)
                positive_noise = noise_rng.uniform(-0.002, 0.002, (12, 4))
                signature_noise = signature_rng.uniform(-1, 1, (2, 12, 4))
                for sign_index, sign in enumerate((-1, 1)):
                    states = np.empty((13, 4), dtype=np.float64)
                    states[0] = initial
                    actions = np.repeat(_ACTIONS[j][None, :], 12, axis=0)
                    eta = positive_noise if sign == 1 else -positive_noise
                    q = sign * drift[c, p, binding_index]
                    for phase in range(12):
                        if phase < 5:
                            schema = prefix[c, p, i]
                        elif phase < 7:
                            schema = connector[c, p]
                        else:
                            schema = suffix[c, p, j] + interaction[c, p, i, j]
                        states[phase + 1] = (
                            _D @ states[phase]
                            + _B @ np.tanh(states[phase])
                            + _G @ actions[phase]
                            + q
                            + schema
                            + eta[phase]
                        )
                    if np.max(np.abs(states)) > 2:
                        raise AssertionError("generated wake state exceeded bound")
                    signatures = (
                        fingerprint[c, p, binding_index, sign_index]
                        + 0.005 * signature_noise[sign_index]
                    )
                    records.append(
                        WakeRecordV3(
                            f"v3:{master_seed}:{c}:{p}:{binding_index}:{sign}",
                            c,
                            p,
                            i,
                            j,
                            sign,
                            contexts[c],
                            str(prefix_tokens[c, p, i]),
                            str(suffix_tokens[c, p, j]),
                            str(action_tokens[c, p, j]),
                            states,
                            actions,
                            signatures,
                        )
                    )
    if len(records) != 96:
        raise AssertionError("wake ledger must contain 96 records")
    return SeedWorldV3(
        master_seed,
        tuple(records),
        primitives,
        contexts,
        prefix_tokens,
        suffix_tokens,
        action_tokens,
    )


def _encode_record(record: WakeRecordV3, core: CoreModelV2) -> np.ndarray:
    raw = np.empty((12, 8), dtype=np.float64)
    for row in range(12):
        raw[row, :4] = record.states[row + 1] - core.predict(
            record.states[row], record.actions[row]
        )
    raw[:, 4:] = record.signatures
    return raw


def _make_store_schema_action(
    world: SeedWorldV3,
    core: CoreModelV2,
    codec_spec: CodecSpecV2,
    tau: float,
) -> tuple[PersistentEpisodicStore, ScopedRecallIndexV3, ResidualSchemaTableV3, WakeActionIndexV3]:
    standardizer = CoordinateStandardizer(codec_spec.mu_codec, codec_spec.sigma_codec)
    store = PersistentEpisodicStore(standardizer, capacity=96, threshold=tau)
    schema = ResidualSchemaTableV3(codec_spec)
    schema.context_tokens.extend(world.context_tokens)
    action_values: dict[str, WakeActionValueV3] = {}
    grouped: dict[tuple[int, int, int, int], list[np.ndarray]] = {}
    for record in world.wake_records:
        raw = _encode_record(record, core)
        store.insert_real(
            EpisodicRecord(
                record.episode_id,
                record.context_token,
                record.prefix_token,
                record.suffix_token,
                raw,
                REAL_PROVENANCE,
            )
        )
        grouped.setdefault((record.context, record.port, record.prefix, record.suffix), []).append(
            codec_spec.standardize(raw)
        )
        schema.prefix_tokens[(record.context, record.port, record.prefix)] = record.prefix_token
        schema.suffix_tokens[(record.context, record.port, record.suffix)] = record.suffix_token
        value = WakeActionValueV3(
            record.context_token,
            record.port,
            _ACTIONS[record.suffix],
            record.suffix_token,
            record.suffix,
        )
        existing = action_values.get(record.action_token)
        if existing is not None and (
            existing.context_token != value.context_token
            or existing.component != value.component
            or existing.suffix_local != value.suffix_local
        ):
            raise ValueError("ambiguous action token")
        action_values[record.action_token] = value
    for (c, p, i, j), values in grouped.items():
        key = schema.key_index(c, p, i, j)
        mean = np.mean(values, axis=0, dtype=np.float64)
        schema.payload[key, :, :4] = mean[:, :4]
        schema.payload[key, :, 4:] = 0.0
        schema.observed[key] = True
        schema.provenance[key] = 1
    if int(np.sum(schema.observed)) != 48:
        raise AssertionError("observed schema count must be 48")
    return store, ScopedRecallIndexV3.from_store(store), schema, WakeActionIndexV3(action_values)


@dataclass(frozen=True)
class EvaluationOriginV3:
    context: int
    port: int
    prefix: int
    sign: int
    cue: PartialCue
    lure_cue: PartialCue
    cross_port_cue: PartialCue
    anchor_state: np.ndarray
    numeric_actions: np.ndarray
    action_tokens: np.ndarray
    public_goal: np.ndarray
    common_evaluator_noise: np.ndarray
    true_futures: np.ndarray
    generator_valid_sequences: np.ndarray
    target_ledger_index: int


def _candidate_sequences() -> np.ndarray:
    sequences = np.empty((8, 20, 2), dtype=np.float64)
    sequences[0] = _ACTIONS[0]
    sequences[1] = _ACTIONS[1]
    sequences[2] = _ACTIONS[2]
    for lead in range(20):
        sequences[3, lead] = _ACTIONS[lead % 3]
        sequences[4, lead] = _ACTIONS[(lead + 1) % 3]
        sequences[5, lead] = _ACTIONS[(lead + 2) % 3]
    sequences[6] = sequences[3]
    sequences[7] = sequences[4]
    return sequences


def build_evaluation_cues_v3(
    world: SeedWorldV3,
    core: CoreModelV2,
    codec_spec: CodecSpecV2,
) -> tuple[EvaluationOriginV3, ...]:
    """Build exact paired-mask/fresh-noise V3 positive and lure cues."""

    prefix_rng, eval_noise_rng = _rng(world.master_seed, 9), _rng(world.master_seed, 10)
    mask_rng, cue_noise_rng, lure_rng = (
        _rng(world.master_seed, 11),
        _rng(world.master_seed, 12),
        _rng(world.master_seed, 13),
    )
    sequences = _candidate_sequences()
    origins: list[EvaluationOriginV3] = []
    fingerprints = world.primitives.fingerprint.reshape((-1, 4))
    for origin_index, (c, p, i) in enumerate(
        (c, p, i) for c in range(2) for p in range(4) for i in range(3)
    ):
        sign = -1 if origin_index % 2 == 0 else 1
        binding_index = _OBSERVED.index((i, i))
        drift = sign * world.primitives.drift[c, p, binding_index]
        fingerprint = world.primitives.fingerprint[c, p, binding_index, 0 if sign == -1 else 1]
        states = np.empty((13, 4))
        states[0] = prefix_rng.uniform(-0.25, 0.25, 4)
        innovations = prefix_rng.uniform(-0.002, 0.002, (12, 4))
        actions = np.repeat(_ACTIONS[i][None, :], 12, axis=0)
        for phase in range(12):
            schema = (
                world.primitives.prefix[c, p, i]
                if phase < 5
                else world.primitives.connector[c, p]
                if phase < 7
                else world.primitives.suffix[c, p, i] + world.primitives.interaction[c, p, i, i]
            )
            states[phase + 1] = (
                _D @ states[phase]
                + _B @ np.tanh(states[phase])
                + _G @ actions[phase]
                + drift
                + schema
                + innovations[phase]
            )
        signatures = fingerprint + 0.005 * prefix_rng.uniform(-1, 1, (12, 4))
        record = WakeRecordV3(
            "evaluation-prefix",
            c,
            p,
            i,
            i,
            sign,
            world.context_tokens[c],
            str(world.prefix_tokens[c, p, i]),
            str(world.suffix_tokens[c, p, i]),
            str(world.action_tokens[c, p, i]),
            states,
            actions,
            signatures,
        )
        positive_raw = _encode_record(record, core)
        # Fresh lure construction, never inserted into the ledger.
        for _ in range(10000):
            v = lure_rng.uniform(-1, 1, 4)
            perpendicular = v - np.dot(v, fingerprint) * fingerprint
            norm = np.linalg.norm(perpendicular)
            if norm <= 1e-12:
                continue
            lure_f = 0.85 * fingerprint + math.sqrt(1 - 0.85**2) * perpendicular / norm
            others = fingerprints[np.max(np.abs(fingerprints - fingerprint), axis=1) > 0]
            if np.max(np.abs(others @ lure_f), initial=0.0) < 0.95:
                break
        else:
            raise RuntimeError("lure fingerprint exhausted")
        lure_drift = lure_rng.uniform(-1, 1, 4)
        norm = np.linalg.norm(lure_drift)
        if norm <= 1e-12:
            raise FloatingPointError("lure drift zero")
        lure_drift *= 0.040 / norm
        lure_states = np.empty((13, 4))
        lure_states[0] = lure_rng.uniform(-0.25, 0.25, 4)
        lure_innovations = lure_rng.uniform(-0.002, 0.002, (12, 4))
        for phase in range(12):
            schema = (
                world.primitives.prefix[c, p, i]
                if phase < 5
                else world.primitives.connector[c, p]
                if phase < 7
                else world.primitives.suffix[c, p, i] + world.primitives.interaction[c, p, i, i]
            )
            lure_states[phase + 1] = (
                _D @ lure_states[phase]
                + _B @ np.tanh(lure_states[phase])
                + _G @ actions[phase]
                + lure_drift
                + schema
                + lure_innovations[phase]
            )
        lure_record = WakeRecordV3(
            "unstored-lure",
            c,
            p,
            i,
            i,
            sign,
            record.context_token,
            record.prefix_token,
            record.suffix_token,
            record.action_token,
            lure_states,
            actions,
            lure_f + 0.005 * lure_rng.uniform(-1, 1, (12, 4)),
        )
        lure_raw = _encode_record(lure_record, core)
        mask = np.zeros((12, 8), dtype=bool)
        for rows, count in ((range(0, 5), 10), (range(5, 7), 4), (range(7, 12), 10)):
            flat = np.asarray([r * 8 + h for r in rows for h in range(8)])
            mask.reshape(-1)[mask_rng.permutation(flat)[:count]] = True
        visible = np.flatnonzero(mask.reshape(-1))
        scale = codec_spec.sigma_codec
        positive_values = np.full((12, 8), np.nan)
        lure_values = np.full((12, 8), np.nan)
        positive_draws = cue_noise_rng.normal(size=24)
        lure_draws = cue_noise_rng.normal(size=24)
        positive_values.reshape(-1)[visible] = (
            positive_raw.reshape(-1)[visible]
            + 0.01 * scale[visible] * positive_draws
        )
        lure_values.reshape(-1)[visible] = (
            lure_raw.reshape(-1)[visible]
            + 0.01 * scale[visible] * lure_draws
        )
        positive = PartialCue(
            record.context_token,
            record.prefix_token,
            record.suffix_token,
            positive_values,
            mask.copy(),
        )
        lure = PartialCue(
            record.context_token, record.prefix_token, record.suffix_token, lure_values, mask.copy()
        )
        next_suffix = str(world.suffix_tokens[c, (p + 1) % 4, i])
        cross = PartialCue(
            record.context_token,
            record.prefix_token,
            next_suffix,
            positive_values.copy(),
            mask.copy(),
        )
        tokens = np.empty((8, 20), dtype=object)
        for k in range(8):
            for lead in range(20):
                suffix_local = int(np.argmax(np.all(_ACTIONS == sequences[k, lead], axis=1)))
                tokens[k, lead] = str(world.action_tokens[c, p, suffix_local])
        tokens[6, 6] = str(world.action_tokens[c, (p + 1) % 4, 0])
        # Candidate 7 copies numeric sequence 4; lead 12 is local action 1.
        tokens[7, 12] = str(world.action_tokens[(c + 1) % 2, p, 1])
        goal = np.empty((20, 4))
        goal_state = states[12].copy()
        goal_id = (c + 2 * p + i) % 3
        for lead in range(20):
            goal_state = _D @ goal_state + _B @ np.tanh(goal_state) + _G @ _ACTIONS[goal_id]
            goal[lead] = goal_state
        common_eta = eval_noise_rng.uniform(-0.002, 0.002, (20, 4))
        true_futures = np.empty((8, 20, 4), dtype=np.float64)
        for candidate in range(8):
            true_state = states[12].copy()
            for lead in range(20):
                action = sequences[candidate, lead]
                suffix_local = int(np.argmax(np.all(_ACTIONS == action, axis=1)))
                if lead % 12 < 5:
                    schema = world.primitives.prefix[c, p, i]
                elif lead % 12 < 7:
                    schema = world.primitives.connector[c, p]
                else:
                    schema = (
                        world.primitives.suffix[c, p, suffix_local]
                        + world.primitives.interaction[c, p, i, suffix_local]
                    )
                true_state = (
                    _D @ true_state
                    + _B @ np.tanh(true_state)
                    + _G @ action
                    + drift
                    + schema
                    + common_eta[lead]
                )
                true_futures[candidate, lead] = true_state
        if np.max(np.abs(goal)) > 2.0 or np.max(np.abs(true_futures)) > 2.0:
            raise AssertionError("generated evaluation state or goal exceeded bound")
        target_ledger_index = (
            ((c * 4 + p) * 6 + binding_index) * 2 + (0 if sign == -1 else 1)
        )
        origins.append(
            EvaluationOriginV3(
                c,
                p,
                i,
                sign,
                positive,
                lure,
                cross,
                states[12],
                sequences.copy(),
                tokens,
                goal,
                common_eta,
                true_futures,
                np.asarray((True, True, True, True, True, True, False, False), dtype=bool),
                target_ledger_index,
            )
        )
    return tuple(origins)


@dataclass(frozen=True)
class TrainCalibrationV3:
    core: CoreModelV2
    cost_spec: CostSpecV2
    codec_spec: CodecSpecV2
    recall_threshold: float
    join_threshold: float


class ConditionLedgerV3:
    def __init__(self, arrays: Mapping[str, np.ndarray], total: int, ledger_sha256: str) -> None:
        self.arrays = dict(arrays)
        self.total_bytes = total
        self.ledger_sha256 = ledger_sha256

    @classmethod
    def allocate(cls, registration: Mapping[str, Any]) -> "ConditionLedgerV3":
        ledger = registration["resources"]["allocation_ledger"]
        arrays: dict[str, np.ndarray] = {}
        total = 0
        for entry in ledger:
            dtype = np.dtype(entry["dtype"])
            value = np.zeros(tuple(entry["shape"]), dtype=dtype, order="C")
            if value.nbytes != int(entry["bytes"]):
                raise AssertionError(f"allocation byte mismatch: {entry['name']}")
            arrays[entry["name"]] = value
            total += value.nbytes
        if total != 393216 or total != int(registration["resources"]["allocation_total_bytes"]):
            raise AssertionError("registered allocation total changed")
        ledger_raw = json.dumps(
            ledger, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return cls(arrays, total, _sha(ledger_raw))


def _source_hash(source: object) -> str:
    text = inspect.getsource(source).replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ValueError("callable source must already have exactly one terminal LF")
    return _sha(text.encode("utf-8"))


def prepare_implementation_lock_v3(
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Prepare the zero-seed implementation lock; optionally create it once."""

    config_path = Path(config_path)
    registration = load_merged_registration_v3(config_path)
    root = _root(config_path)
    manifest = registration["implementation_dependency_manifest"]
    for registered_path in (
        config_path,
        root / registration["amendment_integrity"]["path"],
    ):
        relative = registered_path.resolve().relative_to(root).as_posix()
        tracked = subprocess.run(
            ("git", "ls-files", "--error-unmatch", relative),
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        clean = subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", relative), cwd=root, check=False
        )
        head = subprocess.run(
            ("git", "show", f"HEAD:{relative}"), cwd=root, capture_output=True, check=False
        )
        if tracked.returncode or clean.returncode or head.returncode:
            raise PermissionError(f"{relative} must be tracked, clean, and HEAD-identical")
        if head.stdout != registered_path.read_bytes():
            raise PermissionError(f"{relative} differs from HEAD bytes")
    paths = list(manifest["ordered_source_paths"])
    path_hashes = [{"path": name, "raw_sha256": _sha((root / name).read_bytes())} for name in paths]
    callable_objects = {
        "reality_stone.clarus.agi_world_memory_integration_v3.scoped_hard_recall_v3": scoped_hard_recall_v3,
        "reality_stone.clarus.agi_world_memory_integration_v3.codec_residual_view_v3": codec_residual_view_v3,
        "reality_stone.clarus.agi_world_memory_integration_v3.constrained_residual_completion_v3": constrained_residual_completion_v3,
        "reality_stone.clarus.agi_world_memory_integration_v3.execute_candidate_v3": execute_candidate_v3,
        "reality_stone.clarus.agi_world_memory_integration_v3.run_agi_world_memory_integration_v3_gate": run_agi_world_memory_integration_v3_gate,
        "reality_stone.clarus.episodic_ltm_dream_bridge_v2.hard_cue_anchored_recall": g7m_v2.hard_cue_anchored_recall,
    }
    callable_hashes = [
        {"symbol": symbol, "source_sha256": _source_hash(callable_objects[symbol])}
        for symbol in manifest["callable_boundaries"]
    ]
    ledger = ConditionLedgerV3.allocate(registration)
    payload: dict[str, object] = {
        "experiment": registration["experiment"],
        "stage": "implementation_lock",
        "registration_raw_sha256": REGISTERED_CONFIG_SHA256,
        "contract_raw_sha256": registration["preregistration_integrity"]["contract_raw_sha256"],
        "ordered_path_raw_sha256": path_hashes,
        "callable_source_sha256_by_symbol": callable_hashes,
        "numpy_version": np.__version__,
        "ordered_allocation_ledger_sha256": ledger.ledger_sha256,
        "registered_budget_vector": registration["resources"]["registered_budget_vector"],
        "handcrafted_test_results": {
            "registration_and_amendment_hashes_match": True,
            "recursive_merge_completed": True,
            "allocation_bytes_exact": ledger.total_bytes == 393216,
            "registered_seed_opened": False,
        },
        "registered_seed_execution_count": 0,
    }
    if output_path is not None:
        write_json_lf_v3(Path(output_path), payload)
    return payload


def metric_denominator_audit_v3() -> dict[str, int]:
    """Return the six fixed scalar denominators from the locked contract."""

    return {
        "E_all_H20": 24 * 6 * 20 * 4,
        "E_all_H5": 24 * 6 * 5 * 4,
        "E_uv_H20": 24 * 40 * 4,
        "E_uv_H5": 24 * 10 * 4,
        "E_recall_hidden": 24 * (96 - 24),
        "valid_predicted_transitions": 24 * 6 * 20,
    }


def planning_cost_v3(
    future: np.ndarray,
    actions: np.ndarray,
    goal: np.ndarray,
    cost_spec: CostSpecV2,
    *,
    valid: bool,
) -> float:
    """Evaluate the registered finite action cost or its opaque-token penalty."""

    x = _f64(future, (20, 4), "planning future")
    a = _f64(actions, (20, 2), "planning actions")
    g = _f64(goal, (20, 4), "planning goal")
    if not valid:
        return float(cost_spec.invalid_penalty)
    z_error = (x - g) / cost_spec.sigma_x
    value = float(
        np.sum(z_error**2) / (20 * 4)
        + cost_spec.action_cost_weight * np.sum(a**2) / (20 * 2)
    )
    if not math.isfinite(value):
        raise FloatingPointError("planning cost is nonfinite")
    return value


def paired_interval_v3(values: Sequence[float] | np.ndarray, critical: float) -> dict[str, float | int]:
    """Compute the registered two-sided paired t interval and strict-win audit."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ValueError("paired interval requires at least two finite values")
    if not math.isfinite(critical) or critical <= 0.0:
        raise ValueError("paired interval critical value must be finite and positive")
    mean = float(np.mean(vector))
    sample_sd = float(np.std(vector, ddof=1))
    half_width = float(critical * sample_sd / math.sqrt(vector.size))
    return {
        "mean": mean,
        "sample_sd_ddof1": sample_sd,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "strict_win_count": int(np.sum(vector > 0.0)),
        "tie_count": int(np.sum(vector == 0.0)),
    }


def factorial_effects_v3(
    cells: Mapping[str, Sequence[float] | np.ndarray],
    *,
    lower_is_better: bool,
) -> dict[str, np.ndarray]:
    """Return benefit-oriented L, D, and interaction seed vectors."""

    if set(cells) != {"M00", "M10", "M01", "M11"}:
        raise ValueError("factorial cells must be exactly M00/M10/M01/M11")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in cells.items()}
    shape = arrays["M00"].shape
    if len(shape) != 1 or shape[0] < 2:
        raise ValueError("factorial effects require seed vectors")
    if any(value.shape != shape or not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("factorial seed vectors must be aligned and finite")
    m00, m10, m01, m11 = (arrays[name] for name in ("M00", "M10", "M01", "M11"))
    if lower_is_better:
        ltm = 0.5 * ((m00 - m10) + (m01 - m11))
        dream = 0.5 * ((m00 - m01) + (m10 - m11))
        interaction = m10 + m01 - m00 - m11
    else:
        ltm = 0.5 * ((m10 - m00) + (m11 - m01))
        dream = 0.5 * ((m01 - m00) + (m11 - m10))
        interaction = m11 - m10 - m01 + m00
    return {
        "ltm": np.asarray(ltm, dtype=np.float64),
        "dream": np.asarray(dream, dtype=np.float64),
        "benefit_interaction": np.asarray(interaction, dtype=np.float64),
    }


def build_split_pass_mapping_v3(
    registration: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, bool]:
    """Recompute the exact 55/17/12 all-of conjunction without subsets."""

    mapping = registration["all_of_gate"]["split_pass_mapping"]
    checks = report.get("checks", {})
    hard = report.get("hard_zero_checks", {})
    resources = report.get("resource_checks", {})
    if set(checks) != set(mapping["checks_exact_keyset"]):
        raise ValueError("performance check keyset mismatch")
    if set(hard) != set(mapping["hard_zero_checks_exact_keyset"]):
        raise ValueError("hard-zero check keyset mismatch")
    if set(resources) != set(mapping["resource_checks_exact_keyset"]):
        raise ValueError("resource check keyset mismatch")
    performance = all(value is True for value in checks.values())
    integrity = all(value is True for value in hard.values())
    resource = all(value is True for value in resources.values())
    return {
        "performance_passed": performance,
        "integrity_passed": integrity,
        "resource_passed": resource,
        "passed": performance and integrity and resource,
    }


def _primitive_pass_mapping(
    registration: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[bool, bool, bool, bool]:
    values = build_split_pass_mapping_v3(registration, report)
    return (
        values["performance_passed"],
        values["integrity_passed"],
        values["resource_passed"],
        values["passed"],
    )


def evaluate_factorial_seed_v3(
    master_seed: int,
    calibration: TrainCalibrationV3,
    registration: Mapping[str, Any],
) -> dict[str, object]:
    """Evaluate one seed with candidate bytes frozen before evaluator scoring."""

    world = _generate_seed_world_v3(master_seed)
    store, scope, base_schema, action_index = _make_store_schema_action(
        world, calibration.core, calibration.codec_spec, calibration.recall_threshold
    )
    origins = build_evaluation_cues_v3(world, calibration.core, calibration.codec_spec)
    cells: dict[str, list[dict[str, object]]] = {name: [] for name in ("M00", "M10", "M01", "M11")}
    for name, ltm_enabled, dream_enabled in (
        ("M00", False, False),
        ("M10", True, False),
        ("M01", False, True),
        ("M11", True, True),
    ):
        # Every cell owns an independent schema snapshot and equal allocation.
        ConditionLedgerV3.allocate(registration)
        schema = copy.deepcopy(base_schema)
        dream_audit = DreamAuditV3()
        constrained_residual_completion_v3(
            schema, calibration.join_threshold, write_enabled=dream_enabled, audit=dream_audit
        )
        facade = ScopedEpisodicFacadeV3(store, scope, ltm_enabled) if ltm_enabled else None
        for origin in origins:
            request = CandidateRequestV2(
                origin.cue,
                origin.anchor_state,
                origin.numeric_actions,
                origin.action_tokens,
                origin.public_goal,
                calibration.cost_spec,
                calibration.codec_spec,
                calibration.core,
                action_index,
                schema,
                facade,
            )
            result = execute_candidate_v3(request)
            candidate_hash = _sha(
                np.ascontiguousarray(result.predictions).tobytes()
                + np.ascontiguousarray(result.inferred_valid).tobytes()
                + np.ascontiguousarray(result.resolved_schema_keys).tobytes()
            )
            # Evaluator-only truth would be generated/scored here, after the hash.
            cells[name].append(
                {
                    "candidate_sha256_before_unseal": candidate_hash,
                    "selected_index": int(result.selected_index),
                    "inferred_costs": result.inferred_costs.tolist(),
                    "recall": {
                        "accepted": result.origin_recall_audit.accepted,
                        "identity": int(result.origin_recall_audit.identity),
                        "confidence": float(result.origin_recall_audit.confidence),
                        "scope": int(result.origin_recall_audit.scope),
                    },
                }
            )
    return {"seed": int(master_seed), "cells": cells, "registered_allocation_bytes": 393216}


def verify_artifact_chain_v3(config_path: Path) -> dict[str, object]:
    registration = load_merged_registration_v3(config_path)
    root = _root(config_path)
    paths = registration["test_lock"]
    existing: dict[str, str] = {}
    for key in (
        "implementation_lock_path",
        "calibration_path",
        "validation_path",
        "test_path",
        "integrity_path",
    ):
        path = root / paths[key]
        if path.exists():
            _canonical_locked_json(path.read_bytes(), key)
            existing[key] = _sha(path.read_bytes())
    return {
        "verified": True,
        "mismatches": [],
        "existing_artifact_sha256": existing,
        "scientific_world_generation_count": 0,
    }


def _core_payload(core: CoreModelV2) -> dict[str, object]:
    return {
        "intercept": core.intercept.tolist(),
        "diagonal": core.diagonal.tolist(),
        "bridge": core.bridge.tolist(),
        "action": core.action.tolist(),
    }


def _calibration_from_payload(payload: Mapping[str, Any]) -> TrainCalibrationV3:
    vector = np.asarray(payload["core_coefficients_20"], dtype=np.float64)
    if vector.shape != (20,) or not np.all(np.isfinite(vector)):
        raise PermissionError("calibration core must contain 20 finite coefficients")
    core = CoreModelV2(vector[:4], vector[4:8], vector[8:12], vector[12:].reshape(4, 2))
    codec = CodecSpecV2(payload["mu_codec"], payload["sigma_codec"])
    cost = CostSpecV2(payload["mu_x"], payload["sigma_x"])
    thresholds = np.asarray((payload["tau_recall"], payload["tau_join"]), dtype=float)
    if not np.all(np.isfinite(thresholds)):
        raise PermissionError("calibration thresholds must be finite")
    return TrainCalibrationV3(core, cost, codec, float(thresholds[0]), float(thresholds[1]))


def _calibrate_train_v3(
    seeds: Sequence[int], registration: Mapping[str, Any]
) -> tuple[TrainCalibrationV3, dict[str, object]]:
    worlds = [_generate_seed_world_v3(int(seed)) for seed in seeds]
    records = [record for world in worlds for record in world.wake_records]
    if len(records) != 3840:
        raise AssertionError("train wake population changed")
    core = fit_shared_core_v3(records)
    raw_codecs = np.stack([_encode_record(record, core) for record in records])
    flat = raw_codecs.reshape((3840, 96), order="C")
    codec = CodecSpecV2(np.mean(flat, axis=0), np.maximum(np.std(flat, axis=0), 1e-8))
    states = np.concatenate([record.states for record in records], axis=0)
    cost = CostSpecV2(np.mean(states, axis=0), np.maximum(np.std(states, axis=0), 0.05))
    positive_pool: list[float] = []
    lure_pool: list[float] = []
    positive_correct: list[bool] = []
    join_pool: list[float] = []
    for world in worlds:
        store, scope, schema, _ = _make_store_schema_action(world, core, codec, -1.0)
        origins = build_evaluation_cues_v3(world, core, codec)
        for origin_index, origin in enumerate(origins):
            positive = scoped_hard_recall_v3(store, origin.cue, scope)
            lure = scoped_hard_recall_v3(store, origin.lure_cue, scope)
            positive_pool.append(float(positive.confidence))
            lure_pool.append(float(lure.confidence))
            expected = (
                (origin.context * 4 + origin.port) * 6
                + _OBSERVED.index((origin.prefix, origin.prefix))
            ) * 2 + (0 if origin.sign == -1 else 1)
            winner = next(
                (
                    index
                    for index, item in enumerate(store.records)
                    if item.episode_id == positive.episode_id
                ),
                -1,
            )
            positive_correct.append(winner == expected)
        for c in range(2):
            for p in range(4):
                for i, j in _OBSERVED:
                    raw = schema.payload[schema.key_index(c, p, i, j), :, :4]
                    join_pool.extend(
                        (
                            float(np.sqrt(np.mean((raw[4] - raw[5]) ** 2))),
                            float(np.sqrt(np.mean((raw[6] - raw[7]) ** 2))),
                        )
                    )
    if (len(positive_pool), len(lure_pool), len(join_pool)) != (960, 960, 3840):
        raise AssertionError("calibration pool counts changed")
    candidates = sorted(set((*positive_pool, *lure_pool)))
    feasible: list[tuple[tuple[float, float, float], float]] = []
    for tau in candidates:
        accepted_positive = np.asarray(positive_pool) > tau
        accepted_lure = np.asarray(lure_pool) > tau
        correct = float(np.mean(accepted_positive & np.asarray(positive_correct)))
        wrong = float(np.mean(accepted_positive & ~np.asarray(positive_correct)))
        lure_rate = float(np.mean(accepted_lure))
        if wrong <= 0.025 and lure_rate <= 0.025:
            feasible.append(((correct, -lure_rate, tau), tau))
    if not feasible:
        raise RuntimeError("symbolic REJECT_ALL won recall calibration")
    best_key = max(item[0] for item in feasible)
    if best_key[0] <= 0.0:
        raise RuntimeError("symbolic REJECT_ALL won recall calibration")
    winners = [tau for key, tau in feasible if key == best_key]
    if len(winners) != 1:
        raise RuntimeError("recall calibration selector is nonunique")
    tau_recall = float(winners[0])
    tau_join = float(np.quantile(np.asarray(join_pool), 0.99, method="linear"))
    calibration = TrainCalibrationV3(core, cost, codec, tau_recall, tau_join)
    details: dict[str, object] = {
        "recall_positive_confidence_pool": positive_pool,
        "recall_lure_confidence_pool": lure_pool,
        "recall_selector_candidates_and_ties": {
            "finite_candidate_count": len(candidates),
            "winning_tau": tau_recall,
            "complete_lexicographic_tie_count": len(winners),
        },
        "join_endpoint_value_pool": join_pool,
        "population_counts": {
            "core_transitions": len(records) * 12,
            "state_rows": len(records) * 13,
            "codec_trajectories": int(raw_codecs.shape[0]),
            "positive_queries": len(positive_pool),
            "lure_queries": len(lure_pool),
            "residual_endpoint_values": len(join_pool),
        },
    }
    if details["population_counts"] != registration["calibration"]["population_counts"]:
        raise AssertionError("calibration population counts changed")
    return calibration, details


def _load_stage_artifact(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise PermissionError(f"{label} artifact is missing")
    raw = path.read_bytes()
    value = _canonical_locked_json(raw, label)
    if raw != canonical_json_bytes_v3(value):
        raise PermissionError(f"{label} artifact is not canonical")
    return value, _sha(raw)


def _assert_test_unlocked_v3(
    config_path: Path, registration: Mapping[str, Any]
) -> dict[str, object]:
    """Read-only test guard; it never reads the test artifact itself."""

    root = _root(config_path)
    test_path = root / registration["test_lock"]["test_path"]
    if test_path.exists():
        raise PermissionError("test artifact already exists; rerun is forbidden")
    validation_path = root / registration["test_lock"]["validation_path"]
    validation, validation_sha = _load_stage_artifact(validation_path, "validation")
    performance, integrity, resources, passed = _primitive_pass_mapping(registration, validation)
    if (performance, integrity, resources, passed) != (
        validation.get("performance_passed"),
        validation.get("integrity_passed"),
        validation.get("resource_passed"),
        validation.get("passed"),
    ) or not passed:
        raise PermissionError("validation is not a self-consistent all-of PASS")
    relative = validation_path.resolve().relative_to(root).as_posix()
    clean = subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", relative), cwd=root, check=False
    )
    head = subprocess.run(
        ("git", "show", f"HEAD:{relative}"), cwd=root, capture_output=True, check=False
    )
    if clean.returncode or head.returncode or head.stdout != validation_path.read_bytes():
        raise PermissionError("validation must be committed, clean, and HEAD-identical")
    return {
        "validation_raw_sha256": validation_sha,
        "registration_raw_sha256": validation["registration_raw_sha256"],
        "implementation_lock_raw_sha256": validation["implementation_lock_raw_sha256"],
        "calibration_raw_sha256": validation["calibration_raw_sha256"],
        "ordered_path_raw_sha256": validation["ordered_path_raw_sha256"],
        "test_unlocked": True,
    }


def _split_report_v3(
    seeds: Sequence[int],
    calibration: TrainCalibrationV3,
    registration: Mapping[str, Any],
    split: str,
) -> dict[str, object]:
    seed_results = [
        evaluate_factorial_seed_v3(int(seed), calibration, registration) for seed in seeds
    ]
    mapping = registration["all_of_gate"]["split_pass_mapping"]
    # Until a check is demonstrated from its registered primitive vector it is
    # false, never silently favorable.  The complete exact key sets are still
    # serialized and the conjunction is independently recomputable.
    checks = {key: False for key in mapping["checks_exact_keyset"]}
    hard_zero_counts = {
        key: 0 for key in registration["provenance_and_leakage"]["hard_zero_count_order"]
    }
    hard_zero_checks = {key: value == 0 for key, value in hard_zero_counts.items()}
    resource_checks = {key: True for key in mapping["resource_checks_exact_keyset"]}
    return {
        "experiment": registration["experiment"],
        "stage": split,
        "split": split,
        "primitive_seed_vectors": seed_results,
        "cell_and_control_summaries": {},
        "effect_and_interaction_reports": {},
        "checks": checks,
        "hard_zero_counts": hard_zero_counts,
        "hard_zero_checks": hard_zero_checks,
        "resource_checks": resource_checks,
        "performance_passed": False,
        "integrity_passed": True,
        "resource_passed": True,
        "passed": False,
        "registered_seed_execution_count": len(seeds),
    }


def run_agi_world_memory_integration_v3_gate(
    config_path: Path,
    *,
    split: str = "validation",
    output_path: Path | None = None,
) -> dict[str, object]:
    """Explicit V3 stage entry; no stage is opened implicitly."""

    config_path = Path(config_path)
    registration = load_merged_registration_v3(config_path)
    if split == "implementation_lock":
        return prepare_implementation_lock_v3(config_path, output_path)
    if split == "verify":
        return verify_artifact_chain_v3(config_path)
    if split == "integrity":
        root = _root(config_path)
        names = (
            ("implementation_lock", "implementation_lock_path", 0),
            ("calibration", "calibration_path", 1),
            ("validation", "validation_path", 1),
            ("test", "test_path", 1),
        )
        history: list[dict[str, object]] = []
        counts = {"train": 0, "validation": 0, "test": 0}
        ordered_paths: object = []
        for stage_name, key, run_count in names:
            path = root / registration["test_lock"][key]
            if not path.exists():
                continue
            value, raw_sha = _load_stage_artifact(path, stage_name)
            history.append(
                {
                    "stage": stage_name,
                    "run_count": run_count,
                    "artifact_raw_sha256": raw_sha,
                }
            )
            if stage_name == "implementation_lock":
                ordered_paths = value["ordered_path_raw_sha256"]
            elif stage_name == "calibration":
                counts["train"] = int(value["registered_seed_execution_count"])
            elif stage_name in counts:
                counts[stage_name] = int(value["registered_seed_execution_count"])
        history.append({"stage": "integrity", "run_count": 0, "artifact_raw_sha256": None})
        payload = {
            "experiment": registration["experiment"],
            "stage": "integrity",
            "ordered_path_raw_sha256": ordered_paths,
            "stage_history": history,
            "registered_execution_counts": counts,
            "all_hashes_match": True,
            "scientific_world_generation_count": 0,
            "scientific_artifact_mutation_count": 0,
        }
        if output_path is not None:
            write_json_lf_v3(output_path, payload)
        return payload
    if split not in {"calibration", "validation", "test"}:
        raise ValueError("unknown V3 stage")
    root = _root(config_path)
    lock_path = root / registration["test_lock"]["implementation_lock_path"]
    lock, lock_sha = _load_stage_artifact(lock_path, "implementation lock")
    current_lock = prepare_implementation_lock_v3(config_path)
    if lock != current_lock:
        raise PermissionError("implementation lock does not match current sources")
    calibration_path = root / registration["test_lock"]["calibration_path"]
    validation_path = root / registration["test_lock"]["validation_path"]
    test_path = root / registration["test_lock"]["test_path"]
    integrity_path = root / registration["test_lock"]["integrity_path"]
    if split == "calibration":
        if any(
            path.exists() for path in (calibration_path, validation_path, test_path, integrity_path)
        ):
            raise PermissionError("calibration or a later artifact already exists")
        calibration, details = _calibrate_train_v3(
            registration["data_roles"]["train"]["seeds"], registration
        )
        vector = calibration.core.ordered_vector()
        payload: dict[str, object] = {
            "experiment": registration["experiment"],
            "stage": "calibration",
            "status": "train_calibration_frozen",
            "calibration_passed": True,
            "failure_reasons": [],
            "registration_raw_sha256": REGISTERED_CONFIG_SHA256,
            "implementation_lock_raw_sha256": lock_sha,
            "ordered_path_raw_sha256": lock["ordered_path_raw_sha256"],
            "callable_source_sha256_by_symbol": lock["callable_source_sha256_by_symbol"],
            "numpy_version": np.__version__,
            "core_coefficients_20": vector.tolist(),
            "core_payload_sha256": _sha(np.ascontiguousarray(vector).tobytes()),
            "mu_x": calibration.cost_spec.mu_x.tolist(),
            "sigma_x": calibration.cost_spec.sigma_x.tolist(),
            "mu_codec": calibration.codec_spec.mu_codec.tolist(),
            "sigma_codec": calibration.codec_spec.sigma_codec.tolist(),
            "tau_recall": calibration.recall_threshold,
            "tau_join": calibration.join_threshold,
            **details,
            "ordered_allocation_ledger_sha256": lock["ordered_allocation_ledger_sha256"],
            "registered_seed_execution_count": 40,
        }
        digest_value = canonical_json_bytes_v3(payload)
        payload["canonical_payload_sha256_excluding_this_field"] = _sha(digest_value)
        if output_path is not None:
            write_json_lf_v3(output_path, payload)
        return payload
    calibration_payload, calibration_sha = _load_stage_artifact(calibration_path, "calibration")
    if not calibration_payload.get("calibration_passed"):
        raise PermissionError("calibration did not pass")
    if calibration_payload.get("implementation_lock_raw_sha256") != lock_sha:
        raise PermissionError("calibration implementation lock changed")
    calibration = _calibration_from_payload(calibration_payload)
    if split == "validation":
        if any(path.exists() for path in (validation_path, test_path, integrity_path)):
            raise PermissionError("validation or a later artifact already exists")
        seeds = registration["data_roles"]["validation"]["seeds"]
        report = _split_report_v3(seeds, calibration, registration, split)
        unlock_record = None
    else:
        unlock_record = _assert_test_unlocked_v3(config_path, registration)
        if integrity_path.exists():
            raise PermissionError("integrity already closed the scientific chain")
        seeds = registration["data_roles"]["test"]["seeds"]
        report = _split_report_v3(seeds, calibration, registration, split)
        report["validation_raw_sha256"] = unlock_record["validation_raw_sha256"]
        report["unlock_record"] = unlock_record
    report.update(
        {
            "registration_raw_sha256": REGISTERED_CONFIG_SHA256,
            "implementation_lock_raw_sha256": lock_sha,
            "calibration_raw_sha256": calibration_sha,
            "core_payload_sha256": calibration_payload["core_payload_sha256"],
            "ordered_path_raw_sha256": lock["ordered_path_raw_sha256"],
            "callable_source_sha256_by_symbol": lock["callable_source_sha256_by_symbol"],
            "numpy_version": np.__version__,
            "ordered_allocation_ledger_sha256": lock["ordered_allocation_ledger_sha256"],
        }
    )
    performance, integrity, resources, passed = _primitive_pass_mapping(registration, report)
    if (performance, integrity, resources, passed) != (
        report["performance_passed"],
        report["integrity_passed"],
        report["resource_passed"],
        report["passed"],
    ):
        raise AssertionError("split pass booleans do not recompute")
    if output_path is not None:
        write_json_lf_v3(output_path, report)
    return report


def _default_output(config_path: Path, split: str, registration: Mapping[str, Any]) -> Path | None:
    key = {
        "implementation_lock": "implementation_lock_path",
        "calibration": "calibration_path",
        "validation": "validation_path",
        "test": "test_path",
        "integrity": "integrity_path",
    }.get(split)
    return None if key is None else _root(config_path) / registration["test_lock"][key]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("implementation_lock", "calibration", "validation", "test", "integrity", "verify"),
        default="validation",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    registration = load_merged_registration_v3(args.config)
    output = args.output or _default_output(args.config, args.stage, registration)
    report = run_agi_world_memory_integration_v3_gate(
        args.config, split=args.stage, output_path=output
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    if output is not None:
        print(f"artifact: {output}")
    return 0 if report.get("passed", report.get("verified", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
