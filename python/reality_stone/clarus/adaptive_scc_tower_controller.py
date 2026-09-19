"""Finite controller for the isolated nested-SCC tower fixture.

The controller consumes causal observations, updates only finite normalized
state, and emits immutable tokens.  Forecast and policy APIs accept those
tokens rather than raw events or parent predictions.  Token hashes and snapshot
HMACs are process-local provenance/integrity checks for this research fixture;
they are neither external authentication nor cross-process persistence claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import hmac
from itertools import count
import json
import math
from numbers import Real
import secrets
from typing import Literal, Sequence, TypeAlias

import numpy as np

from .nested_scc_tower import NestedTowerGenerator


_CONTROLLER_ORDINALS = count()
_PROCESS_SNAPSHOT_KEY = secrets.token_bytes(32)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def permutation_hash(permutation: Sequence[int]) -> str:
    try:
        exact = tuple(permutation)
    except TypeError as error:
        raise ValueError("permutation must be an exact integer sequence") from error
    if any(type(value) is not int for value in exact):
        raise ValueError("permutation entries must be exact integers")
    return _hash_payload({"permutation": exact})


def _finite_real_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a finite real-valued sequence") from error
    result: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a real number, not bool or text")
        canonical = float(value)
        if not math.isfinite(canonical):
            raise ValueError(f"{name} must be finite")
        result.append(canonical)
    return tuple(result)


@dataclass(frozen=True)
class CausalEvent:
    tick: int
    observation: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        observation = _finite_real_tuple(self.observation, "observation")
        object.__setattr__(self, "observation", observation)
        CausalEvent._validate_schema(self)

    def _validate_schema(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("event tick must be a nonnegative exact integer")
        if type(self.observation) is not tuple or any(
            type(value) is not float or not math.isfinite(value) for value in self.observation
        ):
            raise ValueError("event observation must be a canonical tuple of finite floats")


@dataclass(frozen=True)
class TowerStateToken:
    controller_identity: str
    episode_generation: int
    tick: int
    active_depth: int
    state_hash: str
    parameter_hash: str

    def __post_init__(self) -> None:
        TowerStateToken._validate_schema(self)

    def _validate_schema(self) -> None:
        for name in ("episode_generation", "tick", "active_depth"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"token {name} must be a nonnegative exact integer")
        if type(self.controller_identity) is not str or not self.controller_identity:
            raise ValueError("token controller_identity must be a nonempty string")
        for name in ("state_hash", "parameter_hash"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"token {name} must be an exact lowercase SHA-256 digest")


@dataclass(frozen=True)
class PolicyDecision:
    probabilities: tuple[float, ...]
    selected_action: int


@dataclass(frozen=True)
class DepthDecision:
    previous_depth: int
    active_depth: int
    extended: bool
    exact_generic_compatibility: bool
    exhausted: bool
    reason: str


@dataclass(frozen=True)
class LevelReset:
    level: int


@dataclass(frozen=True)
class UpperReset:
    """Reset every active upper level and isolate it for the consumed update."""


@dataclass(frozen=True)
class CrossScaleCut:
    """Cut every upward and downward message for the consumed update."""


@dataclass(frozen=True)
class CutUp:
    bridge_level: int


@dataclass(frozen=True)
class CutDown:
    bridge_level: int


@dataclass(frozen=True)
class TimeShift:
    bridge_level: int
    ticks: int = 1
    direction: Literal["both", "up", "down"] = "both"

    def __post_init__(self) -> None:
        if type(self.ticks) is not int:
            raise ValueError("ticks must be the integer one")
        if self.ticks != 1:
            raise ValueError("the isolated fixture registers exactly a one-tick shift")
        if type(self.direction) is not str or self.direction not in ("both", "up", "down"):
            raise ValueError("direction must be both, up, or down")


@dataclass(frozen=True)
class SignFlip:
    bridge_level: int
    direction: Literal["both", "up", "down"] = "both"

    def __post_init__(self) -> None:
        if type(self.direction) is not str or self.direction not in ("both", "up", "down"):
            raise ValueError("direction must be both, up, or down")


@dataclass(frozen=True)
class StateShuffle:
    level: int
    permutation: tuple[int, ...]
    permutation_manifest_hash: str = ""

    def __post_init__(self) -> None:
        permutation = tuple(self.permutation)
        if any(type(value) is not int for value in permutation):
            raise ValueError("state shuffle entries must be exact integers")
        expected_hash = permutation_hash(permutation)
        if type(self.permutation_manifest_hash) is not str:
            raise ValueError("permutation hash must be empty or an exact SHA-256 string")
        if self.permutation_manifest_hash and self.permutation_manifest_hash != expected_hash:
            raise ValueError("permutation hash does not match the frozen permutation")
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "permutation_manifest_hash", expected_hash)


Intervention: TypeAlias = (
    LevelReset | UpperReset | CrossScaleCut | CutUp | CutDown | TimeShift | SignFlip | StateShuffle
)


@dataclass(frozen=True)
class UpdateTrace:
    tick: int
    active_depth: int
    intervention: str | None
    state_before: tuple[tuple[float, ...], ...]
    raw_upward_messages: tuple[tuple[float, ...], ...]
    raw_downward_messages: tuple[tuple[float, ...], ...]
    consumed_upward_messages: tuple[tuple[float, ...], ...]
    consumed_downward_messages: tuple[tuple[float, ...], ...]
    state_after: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ControllerSnapshot:
    """Exact same-process continuation payload sealed by ``state_dict``."""

    parameter_hash: str
    controller_identity: str
    episode_generation: int
    tick: int
    active_depth: int
    states: tuple[tuple[float, ...], ...]
    previous_upward_messages: tuple[tuple[float, ...], ...]
    previous_downward_messages: tuple[tuple[float, ...], ...]
    latest_token: TowerStateToken | None
    last_depth_decision: DepthDecision | None
    integrity_tag: str
    pending_intervention: Intervention | None = None


def _snapshot_payload(snapshot: ControllerSnapshot) -> dict[str, object]:
    payload = asdict(snapshot)
    payload.pop("integrity_tag", None)
    pending = snapshot.pending_intervention
    if pending is None:
        payload["pending_intervention"] = None
    elif is_dataclass(pending):
        payload["pending_intervention"] = {
            "type": type(pending).__name__,
            "fields": asdict(pending),
        }
    else:
        payload["pending_intervention"] = {
            "type": type(pending).__name__,
            "invalid_repr": repr(pending),
        }
    return payload


def _snapshot_integrity_tag(snapshot: ControllerSnapshot) -> str:
    return hmac.new(
        _PROCESS_SNAPSHOT_KEY,
        _canonical_bytes(_snapshot_payload(snapshot)),
        hashlib.sha256,
    ).hexdigest()


class InvalidTowerStateToken(ValueError):
    """Raised when a stale, foreign, or inconsistent token is supplied."""


class TowerCertificateError(ValueError):
    """Raised when the actual scheduled controller lacks its strict certificate."""


def _tuples(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in array) for array in arrays)


class AdaptiveTowerController:
    """A finite grow-only controller with real one-step interventions.

    Depth growth is conservative: while generic append-zero compatibility is
    refused, one generated level is added until ``D_max``.  No level is removed
    from a sampled point defect, and no truncation or infinite-horizon error
    claim is made by this unit controller.
    """

    def __init__(self, generator: NestedTowerGenerator) -> None:
        if type(generator) is not NestedTowerGenerator:
            raise TypeError("generator must be an exact NestedTowerGenerator")
        generator.assert_integrity()
        certificate = generator.certify_prefix(generator.spec.maximum_depth)
        if not certificate.certified:
            raise TowerCertificateError(
                "the actual previous_tick_jacobi controller requires a strict "
                "level-independent global-coordinate-sup certificate: "
                f"{certificate.reason}"
            )
        self._generator = generator
        self._sealed_generator_reference = generator
        self._sealed_generator_spec_hash = generator.manifest.spec_hash
        self._sealed_generator_parameter_hash = generator.manifest.parameter_hash
        self._controller_identity = self._new_identity()
        self._episode_generation = 0
        self._tick = -1
        self._active_depth = 0
        self._states = self._zero_levels(generator.spec.maximum_depth + 1)
        self._previous_upward_messages = self._zero_levels(generator.spec.maximum_depth)
        self._previous_downward_messages = self._zero_levels(generator.spec.maximum_depth)
        self._latest_token: TowerStateToken | None = None
        self._pending_intervention: Intervention | None = None
        self._last_trace: UpdateTrace | None = None
        self._last_depth_decision: DepthDecision | None = None

    @property
    def generator(self) -> NestedTowerGenerator:
        """Return the constructor-bound generator after live seal verification."""

        generator = self._generator
        if generator is not self._sealed_generator_reference:
            raise ValueError("controller generator identity seal mismatch")
        generator.assert_integrity()
        manifest = generator.manifest
        if (
            manifest.spec_hash != self._sealed_generator_spec_hash
            or manifest.parameter_hash != self._sealed_generator_parameter_hash
        ):
            raise ValueError("controller generator operator seal mismatch")
        return generator

    def _new_identity(self) -> str:
        return _hash_payload(
            {
                "parameter_hash": self.generator.manifest.parameter_hash,
                "controller_ordinal": next(_CONTROLLER_ORDINALS),
                "fixture": "adaptive_nested_scc_tower_v1",
            }
        )

    def _zero_levels(self, count_: int) -> list[np.ndarray]:
        return [np.zeros(self.generator.spec.shell_width, dtype=np.float64) for _ in range(count_)]

    @property
    def active_depth(self) -> int:
        return self._active_depth

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def latest_token(self) -> TowerStateToken | None:
        return self._latest_token

    @property
    def last_trace(self) -> UpdateTrace | None:
        return self._last_trace

    @property
    def last_depth_decision(self) -> DepthDecision | None:
        return self._last_depth_decision

    def reset_episode(self) -> None:
        generator = self.generator
        maximum_depth = generator.spec.maximum_depth
        next_generation = self._episode_generation + 1
        next_states = self._zero_levels(maximum_depth + 1)
        next_upward = self._zero_levels(maximum_depth)
        next_downward = self._zero_levels(maximum_depth)

        # Commit only after the generator seal and every allocation succeed.
        self._episode_generation = next_generation
        self._tick = -1
        self._active_depth = 0
        self._states = next_states
        self._previous_upward_messages = next_upward
        self._previous_downward_messages = next_downward
        self._latest_token = None
        self._pending_intervention = None
        self._last_trace = None
        self._last_depth_decision = None

    def _choose_depth(self) -> DepthDecision:
        previous = self._active_depth
        maximum = self.generator.spec.maximum_depth
        if previous >= maximum:
            compatible = maximum == 0 and self.generator.spec.upward_gain == 0.0
            return DepthDecision(
                previous_depth=previous,
                active_depth=previous,
                extended=False,
                exact_generic_compatibility=compatible,
                exhausted=not compatible,
                reason=(
                    "single registered level with structurally zero boundary coupling"
                    if compatible
                    else "maximum finite depth reached without a generic inclusion certificate"
                ),
            )
        compatible = not self.generator.requires_extension(previous)
        if compatible:
            return DepthDecision(
                previous_depth=previous,
                active_depth=previous,
                extended=False,
                exact_generic_compatibility=True,
                exhausted=False,
                reason="append-zero image is invariant on the declared unit domain",
            )
        return DepthDecision(
            previous_depth=previous,
            active_depth=previous + 1,
            extended=True,
            exact_generic_compatibility=False,
            exhausted=False,
            reason="grew conservatively; active boundary coupling refuses exact inclusion",
        )

    def _state_hash(
        self,
        *,
        identity: str | None = None,
        tick: int | None = None,
        active_depth: int | None = None,
        states: Sequence[Sequence[float]] | None = None,
        upward_messages: Sequence[Sequence[float]] | None = None,
        downward_messages: Sequence[Sequence[float]] | None = None,
        episode_generation: int | None = None,
    ) -> str:
        chosen_identity = self._controller_identity if identity is None else identity
        chosen_tick = self._tick if tick is None else tick
        chosen_depth = self._active_depth if active_depth is None else active_depth
        chosen_states = self._states if states is None else states
        chosen_upward = (
            self._previous_upward_messages if upward_messages is None else upward_messages
        )
        chosen_downward = (
            self._previous_downward_messages if downward_messages is None else downward_messages
        )
        chosen_generation = (
            self._episode_generation if episode_generation is None else episode_generation
        )
        payload = {
            "controller_identity": chosen_identity,
            "episode_generation": chosen_generation,
            "tick": chosen_tick,
            "active_depth": chosen_depth,
            "parameter_hash": self.generator.manifest.parameter_hash,
            "states": [
                [float(value).hex() for value in chosen_states[level]]
                for level in range(len(chosen_states))
            ],
            "previous_upward_messages": [
                [float(value).hex() for value in message] for message in chosen_upward
            ],
            "previous_downward_messages": [
                [float(value).hex() for value in message] for message in chosen_downward
            ],
        }
        return _hash_payload(payload)

    def _build_token(
        self,
        *,
        tick: int,
        active_depth: int,
        states: Sequence[Sequence[float]],
        upward_messages: Sequence[Sequence[float]],
        downward_messages: Sequence[Sequence[float]],
    ) -> TowerStateToken:
        return TowerStateToken(
            controller_identity=self._controller_identity,
            episode_generation=self._episode_generation,
            tick=tick,
            active_depth=active_depth,
            state_hash=self._state_hash(
                tick=tick,
                active_depth=active_depth,
                states=states,
                upward_messages=upward_messages,
                downward_messages=downward_messages,
            ),
            parameter_hash=self.generator.manifest.parameter_hash,
        )

    def _validate_intervention(self, intervention: Intervention, depth: int) -> None:
        if type(intervention) not in (
            LevelReset,
            UpperReset,
            CrossScaleCut,
            CutUp,
            CutDown,
            TimeShift,
            SignFlip,
            StateShuffle,
        ):
            raise ValueError("pending intervention has an unknown type")
        if isinstance(intervention, (UpperReset, CrossScaleCut)):
            return
        if isinstance(intervention, (LevelReset, StateShuffle)):
            level = intervention.level
            if type(level) is not int or not 0 <= level <= depth:
                raise ValueError("intervention level must be active")
            if isinstance(intervention, StateShuffle):
                permutation = intervention.permutation
                if type(permutation) is not tuple or any(
                    type(value) is not int for value in permutation
                ):
                    raise ValueError("state shuffle must contain exact integer tuple entries")
                expected = tuple(range(self.generator.spec.shell_width))
                if len(permutation) != len(expected) or tuple(sorted(permutation)) != expected:
                    raise ValueError("state shuffle must be a full shell permutation")
                manifest_hash = intervention.permutation_manifest_hash
                if (
                    type(manifest_hash) is not str
                    or len(manifest_hash) != 64
                    or any(character not in "0123456789abcdef" for character in manifest_hash)
                    or manifest_hash != permutation_hash(permutation)
                ):
                    raise ValueError("state shuffle permutation hash mismatch")
            return
        bridge = intervention.bridge_level
        if type(bridge) is not int or not 0 <= bridge < depth:
            raise ValueError("intervention bridge must connect two active levels")
        if isinstance(intervention, TimeShift):
            if type(intervention.ticks) is not int or intervention.ticks != 1:
                raise ValueError("pending time shift must be exactly one integer tick")
            if type(intervention.direction) is not str or intervention.direction not in (
                "both",
                "up",
                "down",
            ):
                raise ValueError("pending time shift direction is invalid")
        if isinstance(intervention, SignFlip):
            if type(intervention.direction) is not str or intervention.direction not in (
                "both",
                "up",
                "down",
            ):
                raise ValueError("pending sign-flip direction is invalid")

    def with_intervention(self, intervention: Intervention) -> AdaptiveTowerController:
        """Return an independent arm whose next update consumes the lesion."""

        self._validate_intervention(intervention, self._active_depth)
        clone = AdaptiveTowerController(self.generator)
        snapshot = self.state_dict()
        clone._tick = snapshot.tick
        clone._episode_generation = snapshot.episode_generation
        clone._active_depth = snapshot.active_depth
        clone._states = [np.asarray(values, dtype=np.float64).copy() for values in snapshot.states]
        clone._previous_upward_messages = [
            np.asarray(values, dtype=np.float64).copy()
            for values in snapshot.previous_upward_messages
        ]
        clone._previous_downward_messages = [
            np.asarray(values, dtype=np.float64).copy()
            for values in snapshot.previous_downward_messages
        ]
        clone._latest_token = None
        clone._last_depth_decision = snapshot.last_depth_decision
        clone._pending_intervention = intervention
        return clone

    def observe(self, event: CausalEvent) -> TowerStateToken:
        generator = self.generator
        if type(event) is not CausalEvent:
            raise ValueError("observe requires an exact CausalEvent")
        try:
            CausalEvent._validate_schema(event)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("causal event schema is invalid") from error
        expected_tick = self._tick + 1
        if event.tick != expected_tick:
            raise ValueError(
                f"causal tick must be exactly {expected_tick}; stale, duplicate, and future ticks fail"
            )
        normalized = generator.normalize_observation(event.observation)
        decision = self._choose_depth()
        depth = decision.active_depth
        old = [self._states[level].copy() for level in range(depth + 1)]
        intervention = self._pending_intervention
        if intervention is not None:
            self._validate_intervention(intervention, depth)
            if isinstance(intervention, LevelReset):
                old[intervention.level].fill(0.0)
            elif isinstance(intervention, UpperReset):
                for level in range(1, len(old)):
                    old[level].fill(0.0)
            elif isinstance(intervention, StateShuffle):
                old[intervention.level] = old[intervention.level][
                    list(intervention.permutation)
                ].copy()

        raw_up, raw_down = self.generator.bridge_messages(old)
        consumed_up = [message.copy() for message in raw_up]
        consumed_down = [message.copy() for message in raw_down]
        if isinstance(intervention, (UpperReset, CrossScaleCut)):
            for message in consumed_up:
                message.fill(0.0)
            for message in consumed_down:
                message.fill(0.0)
        elif isinstance(intervention, CutUp):
            consumed_up[intervention.bridge_level].fill(0.0)
        elif isinstance(intervention, CutDown):
            consumed_down[intervention.bridge_level].fill(0.0)
        elif isinstance(intervention, TimeShift):
            bridge = intervention.bridge_level
            if intervention.direction in ("both", "up"):
                consumed_up[bridge] = self._previous_upward_messages[bridge].copy()
            if intervention.direction in ("both", "down"):
                consumed_down[bridge] = self._previous_downward_messages[bridge].copy()
        elif isinstance(intervention, SignFlip):
            bridge = intervention.bridge_level
            if intervention.direction in ("both", "up"):
                consumed_up[bridge] = -consumed_up[bridge]
            if intervention.direction in ("both", "down"):
                consumed_down[bridge] = -consumed_down[bridge]

        updated = self.generator.step_with_messages(old, normalized, consumed_up, consumed_down)
        next_states = [state.copy() for state in self._states]
        for level, state in enumerate(updated):
            next_states[level] = state.copy()
        for level in range(depth + 1, len(next_states)):
            next_states[level].fill(0.0)
        next_upward = [message.copy() for message in self._previous_upward_messages]
        next_downward = [message.copy() for message in self._previous_downward_messages]
        for bridge, message in enumerate(raw_up):
            next_upward[bridge] = message.copy()
        for bridge, message in enumerate(raw_down):
            next_downward[bridge] = message.copy()
        trace = UpdateTrace(
            tick=event.tick,
            active_depth=depth,
            intervention=None if intervention is None else type(intervention).__name__,
            state_before=_tuples(old),
            raw_upward_messages=_tuples(raw_up),
            raw_downward_messages=_tuples(raw_down),
            consumed_upward_messages=_tuples(consumed_up),
            consumed_downward_messages=_tuples(consumed_down),
            state_after=_tuples(updated),
        )
        next_token = self._build_token(
            tick=event.tick,
            active_depth=depth,
            states=next_states,
            upward_messages=next_upward,
            downward_messages=next_downward,
        )

        # Commit only after normalization, updates, trace, and token construction succeed.
        self._active_depth = depth
        self._states = next_states
        self._previous_upward_messages = next_upward
        self._previous_downward_messages = next_downward
        self._tick = event.tick
        self._last_depth_decision = decision
        self._pending_intervention = None
        self._last_trace = trace
        self._latest_token = next_token
        return next_token

    def _validate_token(self, token: TowerStateToken) -> None:
        self.generator.assert_integrity()
        if type(token) is not TowerStateToken:
            raise InvalidTowerStateToken("readout requires an immutable TowerStateToken")
        try:
            TowerStateToken._validate_schema(token)
        except ValueError as error:
            raise InvalidTowerStateToken("tower token schema is invalid") from error
        if token.controller_identity != self._controller_identity:
            raise InvalidTowerStateToken("foreign controller token")
        if token.episode_generation != self._episode_generation:
            raise InvalidTowerStateToken("stale episode token")
        if token.parameter_hash != self.generator.manifest.parameter_hash:
            raise InvalidTowerStateToken("token parameter manifest mismatch")
        if self._latest_token is None or token != self._latest_token:
            raise InvalidTowerStateToken("stale or unissued token")
        if token.tick != self._tick or token.active_depth != self._active_depth:
            raise InvalidTowerStateToken("stale token metadata")
        if token.state_hash != self._state_hash():
            raise InvalidTowerStateToken("state changed after token issuance")

    def _state_readout(self) -> np.ndarray:
        weights = np.asarray(
            [self.generator.spec.level_decay**level for level in range(self._active_depth + 1)],
            dtype=np.float64,
        )
        weights /= float(np.sum(weights))
        stacked = np.stack(self._states[: self._active_depth + 1], axis=0)
        return np.sum(weights[:, None] * stacked, axis=0)

    def read_forecast(self, token: TowerStateToken) -> tuple[float, ...]:
        """Read only the current registered tower state named by ``token``."""

        self._validate_token(token)
        return tuple(float(value) for value in self._state_readout())

    def read_policy(self, token: TowerStateToken, action_mask: Sequence[bool]) -> PolicyDecision:
        self._validate_token(token)
        mask = tuple(action_mask)
        if len(mask) != self.generator.spec.shell_width or not all(
            isinstance(value, (bool, np.bool_)) for value in mask
        ):
            raise ValueError("action_mask must be a shell-width boolean sequence")
        if not any(mask):
            raise ValueError("action_mask must allow at least one action")
        logits = self._state_readout()
        allowed = np.asarray(mask, dtype=bool)
        shifted = logits[allowed] - float(np.max(logits[allowed]))
        exp_values = np.exp(shifted)
        exp_values /= float(np.sum(exp_values))
        probabilities = np.zeros_like(logits)
        probabilities[allowed] = exp_values
        selected = int(np.argmax(probabilities))
        return PolicyDecision(
            probabilities=tuple(float(value) for value in probabilities),
            selected_action=selected,
        )

    def state_copy(self) -> tuple[np.ndarray, ...]:
        self.generator.assert_integrity()
        return tuple(state.copy() for state in self._states)

    def state_dict(self) -> ControllerSnapshot:
        self.generator.assert_integrity()
        unsigned = ControllerSnapshot(
            parameter_hash=self.generator.manifest.parameter_hash,
            controller_identity=self._controller_identity,
            episode_generation=self._episode_generation,
            tick=self._tick,
            active_depth=self._active_depth,
            states=_tuples(self._states),
            previous_upward_messages=_tuples(self._previous_upward_messages),
            previous_downward_messages=_tuples(self._previous_downward_messages),
            latest_token=self._latest_token,
            last_depth_decision=self._last_depth_decision,
            integrity_tag="",
            pending_intervention=self._pending_intervention,
        )
        return replace(unsigned, integrity_tag=_snapshot_integrity_tag(unsigned))

    def load_state_dict(self, snapshot: ControllerSnapshot) -> None:
        self.generator.assert_integrity()
        if not isinstance(snapshot, ControllerSnapshot):
            raise TypeError("snapshot must be a ControllerSnapshot")
        if type(snapshot.integrity_tag) is not str or not hmac.compare_digest(
            snapshot.integrity_tag, _snapshot_integrity_tag(snapshot)
        ):
            raise ValueError("snapshot process-local integrity tag mismatch")
        if (
            type(snapshot.parameter_hash) is not str
            or len(snapshot.parameter_hash) != 64
            or any(character not in "0123456789abcdef" for character in snapshot.parameter_hash)
            or snapshot.parameter_hash != self.generator.manifest.parameter_hash
        ):
            raise ValueError("snapshot parameter manifest mismatch")
        if type(snapshot.controller_identity) is not str or not snapshot.controller_identity:
            raise ValueError("snapshot controller identity must be nonempty")
        if type(snapshot.episode_generation) is not int or snapshot.episode_generation < 0:
            raise ValueError("snapshot episode generation must be a nonnegative integer")
        maximum = self.generator.spec.maximum_depth
        if (
            type(snapshot.tick) is not int
            or snapshot.tick < -1
            or type(snapshot.active_depth) is not int
            or not 0 <= snapshot.active_depth <= maximum
        ):
            raise ValueError("snapshot tick or active depth is invalid")
        try:
            raw_states = tuple(snapshot.states)
            raw_upward = tuple(snapshot.previous_upward_messages)
            raw_downward = tuple(snapshot.previous_downward_messages)
        except TypeError as error:
            raise ValueError("snapshot arrays must use finite outer sequences") from error
        if len(raw_states) != maximum + 1:
            raise ValueError("snapshot state level count mismatch")
        if len(raw_upward) != maximum or len(raw_downward) != maximum:
            raise ValueError("snapshot message history count mismatch")

        states = [
            np.asarray(
                _finite_real_tuple(values, f"snapshot.states[{index}]"),
                dtype=np.float64,
            )
            for index, values in enumerate(raw_states)
        ]
        upward = [
            np.asarray(
                _finite_real_tuple(values, f"snapshot.previous_upward_messages[{index}]"),
                dtype=np.float64,
            )
            for index, values in enumerate(raw_upward)
        ]
        downward = [
            np.asarray(
                _finite_real_tuple(values, f"snapshot.previous_downward_messages[{index}]"),
                dtype=np.float64,
            )
            for index, values in enumerate(raw_downward)
        ]
        width = self.generator.spec.shell_width
        for values in (*states, *upward, *downward):
            if values.shape != (width,) or not np.all(np.isfinite(values)):
                raise ValueError("snapshot arrays must be finite shell-width vectors")
        if any(np.any(np.abs(values) > 1.0) for values in states):
            raise ValueError("snapshot states must lie in the normalized state domain")
        if any(np.any(values != 0.0) for values in states[snapshot.active_depth + 1 :]):
            raise ValueError("inactive snapshot states must be exactly zero")
        if any(np.any(values != 0.0) for values in upward[snapshot.active_depth :]) or any(
            np.any(values != 0.0) for values in downward[snapshot.active_depth :]
        ):
            raise ValueError("inactive snapshot message histories must be exactly zero")
        if snapshot.latest_token is not None:
            token = snapshot.latest_token
            if type(token) is not TowerStateToken:
                raise InvalidTowerStateToken("snapshot token has an invalid type")
            try:
                TowerStateToken._validate_schema(token)
            except ValueError as error:
                raise InvalidTowerStateToken("snapshot token schema is invalid") from error
            expected_hash = self._state_hash(
                identity=snapshot.controller_identity,
                tick=snapshot.tick,
                active_depth=snapshot.active_depth,
                states=states,
                upward_messages=upward,
                downward_messages=downward,
                episode_generation=snapshot.episode_generation,
            )
            if (
                token.controller_identity != snapshot.controller_identity
                or token.episode_generation != snapshot.episode_generation
                or token.tick != snapshot.tick
                or token.active_depth != snapshot.active_depth
                or token.parameter_hash != snapshot.parameter_hash
                or token.state_hash != expected_hash
            ):
                raise InvalidTowerStateToken("snapshot contains an inconsistent pending token")
        pending = snapshot.pending_intervention
        if pending is not None:
            if snapshot.latest_token is not None:
                raise ValueError("a pending intervention snapshot cannot carry a state token")
            self._validate_intervention(pending, snapshot.active_depth)

        self._controller_identity = snapshot.controller_identity
        self._episode_generation = snapshot.episode_generation
        self._tick = snapshot.tick
        self._active_depth = snapshot.active_depth
        self._states = states
        self._previous_upward_messages = upward
        self._previous_downward_messages = downward
        self._latest_token = snapshot.latest_token
        self._last_depth_decision = snapshot.last_depth_decision
        self._pending_intervention = pending
        self._last_trace = None


__all__ = [
    "AdaptiveTowerController",
    "CausalEvent",
    "ControllerSnapshot",
    "CrossScaleCut",
    "CutDown",
    "CutUp",
    "DepthDecision",
    "InvalidTowerStateToken",
    "Intervention",
    "LevelReset",
    "PolicyDecision",
    "SignFlip",
    "StateShuffle",
    "TimeShift",
    "TowerCertificateError",
    "TowerStateToken",
    "UpdateTrace",
    "UpperReset",
    "permutation_hash",
]
