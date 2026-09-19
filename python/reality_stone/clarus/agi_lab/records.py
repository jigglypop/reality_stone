"""Immutable records for the physics-independent AGI Core V0 scaffold.

The records deliberately accept only a small canonical value grammar.  They do
not carry evaluator truth, callbacks, arrays, or mutable containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
import unicodedata


CORE_SCHEMA = "ce.agi_lab.physics-independent-core.v0"
PERMIT_SCHEMA = "ce.agi_lab.action-permit.v0"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def exact_int(name: str, value: object, *, nonnegative: bool = False) -> int:
    """Return an exact int64, rejecting bool and numeric coercions."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < _INT64_MIN or value > _INT64_MAX:
        raise ValueError(f"{name} must fit signed int64")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def finite_float(name: str, value: object) -> float:
    """Return a finite built-in float without accepting bool or custom numerics."""

    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a built-in int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def normalized_text(name: str, value: object) -> str:
    """Return a nonempty NFC string."""

    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    result = unicodedata.normalize("NFC", value)
    result.encode("utf-8", errors="strict")
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def canonical_value(value: object) -> object:
    """Copy a value into the recursively immutable V0 canonical grammar."""

    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return exact_int("canonical integer", value)
    if type(value) is float:
        return finite_float("canonical float", value)
    if type(value) is str:
        result = unicodedata.normalize("NFC", value)
        result.encode("utf-8", errors="strict")
        return result
    if type(value) is tuple:
        return tuple(canonical_value(item) for item in value)
    raise TypeError(
        "canonical values permit only null, bool, int64, finite float, string, and tuple"
    )


def canonical_map(
    name: str, value: object, *, scalar_values_only: bool = False
) -> tuple[tuple[str, object], ...]:
    """Validate and byte-sort an immutable tuple of unique key/value pairs."""

    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of pairs")
    result: list[tuple[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{name}[{index}] must be an exact pair")
        key = normalized_text(f"{name}[{index}].key", item[0])
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        seen.add(key)
        item_value = canonical_value(item[1])
        if scalar_values_only and type(item_value) is tuple:
            raise TypeError(f"{name}[{index}] must contain a scalar value")
        result.append((key, item_value))
    result.sort(key=lambda pair: pair[0].encode("utf-8"))
    return tuple(result)


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        payload: dict[str, object] = {"__type__": type(value).__name__}
        for record_field in fields(value):
            if record_field.metadata.get("private"):
                continue
            payload[record_field.name] = _jsonable(getattr(value, record_field.name))
        return payload
    if type(value) is tuple:
        return [_jsonable(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return canonical_value(value)
    raise TypeError(f"{type(value).__name__} is not canonically serializable")


def canonical_bytes(value: object) -> bytes:
    """Serialize public canonical data with stable UTF-8 JSON bytes."""

    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class CoreAction:
    name: str
    arguments: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalized_text("action.name", self.name))
        object.__setattr__(
            self,
            "arguments",
            canonical_map("action.arguments", self.arguments, scalar_values_only=True),
        )

    def argument(self, name: str) -> object:
        normalized = normalized_text("argument name", name)
        for key, value in self.arguments:
            if key == normalized:
                return value
        raise KeyError(normalized)


@dataclass(frozen=True)
class ActionSpace:
    actions: tuple[CoreAction, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.actions) is not tuple or not self.actions:
            raise TypeError("actions must be a nonempty tuple")
        if any(type(action) is not CoreAction for action in self.actions):
            raise TypeError("actions must contain only CoreAction values")
        action_digests = tuple(canonical_digest(action) for action in self.actions)
        if len(set(action_digests)) != len(action_digests):
            raise ValueError("action space contains duplicate actions")
        object.__setattr__(self, "digest", canonical_digest((CORE_SCHEMA, action_digests)))

    def contains(self, action: CoreAction) -> bool:
        return any(action == candidate for candidate in self.actions)


@dataclass(frozen=True)
class CoreGoal:
    target_state: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_state", exact_int("goal.target_state", self.target_state))


@dataclass(frozen=True)
class CoreObservation:
    episode_id: str
    tick: int
    state: int
    goal: CoreGoal

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", normalized_text("episode_id", self.episode_id))
        object.__setattr__(self, "tick", exact_int("tick", self.tick, nonnegative=True))
        object.__setattr__(self, "state", exact_int("state", self.state))
        if type(self.goal) is not CoreGoal:
            raise TypeError("goal must be a CoreGoal")


@dataclass(frozen=True)
class GenesisRequest:
    episode_id: str
    initial_state: int
    goal: CoreGoal

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", normalized_text("episode_id", self.episode_id))
        object.__setattr__(self, "initial_state", exact_int("initial_state", self.initial_state))
        if type(self.goal) is not CoreGoal:
            raise TypeError("goal must be a CoreGoal")


@dataclass(frozen=True)
class WorldSession:
    episode_id: str
    tick: int
    state: int
    goal: CoreGoal
    world_commitment: str
    action_space: ActionSpace
    policy_digest: str
    used_nonces: tuple[str, ...]
    terminated: bool
    transition_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", normalized_text("episode_id", self.episode_id))
        object.__setattr__(self, "tick", exact_int("tick", self.tick, nonnegative=True))
        object.__setattr__(self, "state", exact_int("state", self.state))
        if type(self.goal) is not CoreGoal:
            raise TypeError("goal must be a CoreGoal")
        object.__setattr__(
            self,
            "world_commitment",
            normalized_text("world_commitment", self.world_commitment),
        )
        if type(self.action_space) is not ActionSpace:
            raise TypeError("action_space must be an ActionSpace")
        object.__setattr__(
            self, "policy_digest", normalized_text("policy_digest", self.policy_digest)
        )
        if type(self.used_nonces) is not tuple or any(
            type(nonce) is not str for nonce in self.used_nonces
        ):
            raise TypeError("used_nonces must be a tuple of strings")
        nonces = tuple(sorted(normalized_text("nonce", nonce) for nonce in self.used_nonces))
        if len(set(nonces)) != len(nonces):
            raise ValueError("used_nonces contains duplicates")
        object.__setattr__(self, "used_nonces", nonces)
        if type(self.terminated) is not bool:
            raise TypeError("terminated must be a bool")
        object.__setattr__(
            self,
            "transition_count",
            exact_int("transition_count", self.transition_count, nonnegative=True),
        )


@dataclass(frozen=True)
class WorldStart:
    observation: CoreObservation
    action_space: ActionSpace
    genesis_digest: str

    def __post_init__(self) -> None:
        if type(self.observation) is not CoreObservation:
            raise TypeError("observation must be a CoreObservation")
        if type(self.action_space) is not ActionSpace:
            raise TypeError("action_space must be an ActionSpace")
        object.__setattr__(
            self, "genesis_digest", normalized_text("genesis_digest", self.genesis_digest)
        )


@dataclass(frozen=True)
class WorldStep:
    observation: CoreObservation
    goal_reached: bool
    terminated: bool
    transition_count: int
    public_reason: str

    def __post_init__(self) -> None:
        if type(self.observation) is not CoreObservation:
            raise TypeError("observation must be a CoreObservation")
        if type(self.goal_reached) is not bool or type(self.terminated) is not bool:
            raise TypeError("goal_reached and terminated must be bool")
        object.__setattr__(
            self,
            "transition_count",
            exact_int("transition_count", self.transition_count, nonnegative=True),
        )
        object.__setattr__(
            self, "public_reason", normalized_text("public_reason", self.public_reason)
        )


@dataclass(frozen=True)
class BeliefState:
    transitions: tuple[tuple[int, str, int], ...]
    model_version: str

    def __post_init__(self) -> None:
        if type(self.transitions) is not tuple:
            raise TypeError("transitions must be a tuple")
        checked: list[tuple[int, str, int]] = []
        for index, item in enumerate(self.transitions):
            if type(item) is not tuple or len(item) != 3:
                raise TypeError(f"transitions[{index}] must be an exact triple")
            checked.append(
                (
                    exact_int(f"transitions[{index}].state", item[0]),
                    normalized_text(f"transitions[{index}].action", item[1]),
                    exact_int(f"transitions[{index}].next_state", item[2]),
                )
            )
        checked.sort(key=lambda item: (item[0], item[1].encode("utf-8"), item[2]))
        keys = [(state, action) for state, action, _ in checked]
        if len(set(keys)) != len(keys):
            raise ValueError("belief contains conflicting transition keys")
        object.__setattr__(self, "transitions", tuple(checked))
        object.__setattr__(
            self, "model_version", normalized_text("model_version", self.model_version)
        )


@dataclass(frozen=True)
class PredictedOutcome:
    next_state: int
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "next_state", exact_int("next_state", self.next_state))
        confidence = finite_float("confidence", self.confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class ActionProposal:
    action: CoreAction
    expected_score: float
    predicted_risk: float
    model_version: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.action) is not CoreAction:
            raise TypeError("action must be a CoreAction")
        object.__setattr__(
            self, "expected_score", finite_float("expected_score", self.expected_score)
        )
        risk = finite_float("predicted_risk", self.predicted_risk)
        if risk < 0.0 or risk > 1.0:
            raise ValueError("predicted_risk must be in [0, 1]")
        object.__setattr__(self, "predicted_risk", risk)
        object.__setattr__(
            self, "model_version", normalized_text("model_version", self.model_version)
        )
        if type(self.evidence_refs) is not tuple or any(
            type(reference) is not str for reference in self.evidence_refs
        ):
            raise TypeError("evidence_refs must be a tuple of strings")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(normalized_text("evidence_ref", item) for item in self.evidence_refs),
        )


@dataclass(frozen=True)
class ActionPermit:
    schema: str
    episode_id: str
    tick: int
    world_commitment: str
    session_digest: str
    action_space_digest: str
    policy_digest: str
    proposal_digest: str
    action: CoreAction
    nonce: str
    authentication_tag: str = field(repr=False, metadata={"private": True})

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", normalized_text("permit.schema", self.schema))
        object.__setattr__(
            self, "episode_id", normalized_text("permit.episode_id", self.episode_id)
        )
        object.__setattr__(self, "tick", exact_int("permit.tick", self.tick, nonnegative=True))
        for name in (
            "world_commitment",
            "session_digest",
            "action_space_digest",
            "policy_digest",
            "proposal_digest",
            "nonce",
            "authentication_tag",
        ):
            object.__setattr__(self, name, normalized_text(f"permit.{name}", getattr(self, name)))
        if type(self.action) is not CoreAction:
            raise TypeError("permit.action must be a CoreAction")

    def public_claim(self) -> tuple[tuple[str, object], ...]:
        return canonical_map(
            "permit claim",
            (
                ("action", canonical_digest(self.action)),
                ("action_space_digest", self.action_space_digest),
                ("episode_id", self.episode_id),
                ("nonce", self.nonce),
                ("policy_digest", self.policy_digest),
                ("proposal_digest", self.proposal_digest),
                ("schema", self.schema),
                ("session_digest", self.session_digest),
                ("tick", self.tick),
                ("world_commitment", self.world_commitment),
            ),
        )

    def public_claim_digest(self) -> str:
        return canonical_digest(self.public_claim())


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason_codes: tuple[str, ...]
    selected_action: CoreAction | None
    permit: ActionPermit | None

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be a bool")
        if type(self.reason_codes) is not tuple or not self.reason_codes:
            raise TypeError("reason_codes must be a nonempty tuple")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(normalized_text("reason_code", reason) for reason in self.reason_codes),
        )
        if self.allowed:
            if type(self.selected_action) is not CoreAction or type(self.permit) is not ActionPermit:
                raise ValueError("an allowed decision requires an action and permit")
            if self.selected_action != self.permit.action:
                raise ValueError("selected action and permit action differ")
        elif self.selected_action is not None or self.permit is not None:
            raise ValueError("a denied decision cannot carry an action or permit")


@dataclass(frozen=True)
class ExperienceRecord:
    before: CoreObservation
    belief: BeliefState
    action: CoreAction
    after: CoreObservation
    terminated: bool

    def __post_init__(self) -> None:
        if type(self.before) is not CoreObservation or type(self.after) is not CoreObservation:
            raise TypeError("experience observations must be CoreObservation")
        if self.before.episode_id != self.after.episode_id:
            raise ValueError("experience cannot cross episodes")
        if self.after.tick != self.before.tick + 1:
            raise ValueError("experience ticks must be consecutive")
        if type(self.belief) is not BeliefState or type(self.action) is not CoreAction:
            raise TypeError("experience belief/action types are invalid")
        if type(self.terminated) is not bool:
            raise TypeError("terminated must be bool")


@dataclass(frozen=True)
class UpdateReceipt:
    previous_model_version: str
    new_model_version: str
    experience_digest: str

    def __post_init__(self) -> None:
        for name in (
            "previous_model_version",
            "new_model_version",
            "experience_digest",
        ):
            object.__setattr__(self, name, normalized_text(name, getattr(self, name)))


@dataclass(frozen=True)
class LedgerEntry:
    index: int
    event_type: str
    payload: tuple[tuple[str, object], ...]
    previous_digest: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", exact_int("ledger.index", self.index, nonnegative=True))
        object.__setattr__(
            self, "event_type", normalized_text("ledger.event_type", self.event_type)
        )
        object.__setattr__(self, "payload", canonical_map("ledger.payload", self.payload))
        object.__setattr__(
            self,
            "previous_digest",
            normalized_text("ledger.previous_digest", self.previous_digest),
        )
        object.__setattr__(self, "digest", normalized_text("ledger.digest", self.digest))


@dataclass(frozen=True)
class CoreRuntimeState:
    session: WorldSession
    observation: CoreObservation
    belief: BeliefState
    ledger: tuple[LedgerEntry, ...]

    def __post_init__(self) -> None:
        if type(self.session) is not WorldSession:
            raise TypeError("session must be a WorldSession")
        if type(self.observation) is not CoreObservation:
            raise TypeError("observation must be a CoreObservation")
        if type(self.belief) is not BeliefState:
            raise TypeError("belief must be a BeliefState")
        if type(self.ledger) is not tuple or not self.ledger or any(
            type(entry) is not LedgerEntry for entry in self.ledger
        ):
            raise TypeError("ledger must be a nonempty tuple of LedgerEntry")
        if self.session.episode_id != self.observation.episode_id:
            raise ValueError("session and observation episode differ")
        if self.session.tick != self.observation.tick:
            raise ValueError("session and observation tick differ")
        if self.session.state != self.observation.state:
            raise ValueError("session and observation state differ")
        if self.session.goal != self.observation.goal:
            raise ValueError("session and observation goal differ")


@dataclass(frozen=True)
class DecisionDraft:
    proposals: tuple[ActionProposal, ...]
    safety_decision: SafetyDecision

    def __post_init__(self) -> None:
        if type(self.proposals) is not tuple or not self.proposals or any(
            type(proposal) is not ActionProposal for proposal in self.proposals
        ):
            raise TypeError("proposals must be a nonempty tuple of ActionProposal")
        if type(self.safety_decision) is not SafetyDecision:
            raise TypeError("safety_decision must be SafetyDecision")


@dataclass(frozen=True)
class CoreStepResult:
    world_step: WorldStep
    update_receipt: UpdateReceipt
    public_entry: LedgerEntry

    def __post_init__(self) -> None:
        if type(self.world_step) is not WorldStep:
            raise TypeError("world_step must be WorldStep")
        if type(self.update_receipt) is not UpdateReceipt:
            raise TypeError("update_receipt must be UpdateReceipt")
        if type(self.public_entry) is not LedgerEntry:
            raise TypeError("public_entry must be LedgerEntry")


def public_ledger_bytes(entries: tuple[LedgerEntry, ...]) -> bytes:
    if type(entries) is not tuple or any(type(entry) is not LedgerEntry for entry in entries):
        raise TypeError("entries must be a tuple of LedgerEntry")
    return canonical_bytes(entries)


def dataclass_public_fields(value: object) -> tuple[str, ...]:
    """Return public field names for boundary tests without reading field values."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be a dataclass instance")
    return tuple(item.name for item in fields(value) if not item.metadata.get("private"))


__all__ = [
    "CORE_SCHEMA",
    "PERMIT_SCHEMA",
    "ActionPermit",
    "ActionProposal",
    "ActionSpace",
    "BeliefState",
    "CoreAction",
    "CoreGoal",
    "CoreObservation",
    "CoreRuntimeState",
    "CoreStepResult",
    "DecisionDraft",
    "ExperienceRecord",
    "GenesisRequest",
    "LedgerEntry",
    "PredictedOutcome",
    "SafetyDecision",
    "UpdateReceipt",
    "WorldSession",
    "WorldStart",
    "WorldStep",
    "canonical_bytes",
    "canonical_digest",
    "canonical_map",
    "canonical_value",
    "dataclass_public_fields",
    "exact_int",
    "finite_float",
    "normalized_text",
    "public_ledger_bytes",
]
