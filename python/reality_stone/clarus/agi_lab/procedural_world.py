"""Two exact software-world families and V0 reference components.

The worlds are categorical, deterministic, and dimensionless.  Their hidden
transition rules remain inside the adapter and are never included in an
agent-facing record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac

from .contracts import DynamicsPredictor
from .records import (
    CORE_SCHEMA,
    PERMIT_SCHEMA,
    ActionPermit,
    ActionProposal,
    ActionSpace,
    BeliefState,
    CoreAction,
    CoreGoal,
    CoreObservation,
    ExperienceRecord,
    GenesisRequest,
    PredictedOutcome,
    SafetyDecision,
    UpdateReceipt,
    WorldSession,
    WorldStart,
    WorldStep,
    canonical_bytes,
    canonical_digest,
    exact_int,
    normalized_text,
)


APPLY_ZERO = CoreAction("apply", (("value", 0),))
APPLY_ONE = CoreAction("apply", (("value", 1),))
FORBIDDEN = CoreAction("forbidden")
ABSTAIN = CoreAction("core.abstain")
DEFAULT_ACTION_SPACE = ActionSpace((APPLY_ZERO, APPLY_ONE, FORBIDDEN, ABSTAIN))


@dataclass(frozen=True)
class SafetyPolicy:
    forbidden_action_names: tuple[str, ...]
    fallback_action: CoreAction
    digest: str = ""

    def __post_init__(self) -> None:
        if type(self.forbidden_action_names) is not tuple:
            raise TypeError("forbidden_action_names must be a tuple")
        forbidden = tuple(
            sorted(normalized_text("forbidden action", name) for name in self.forbidden_action_names)
        )
        if len(set(forbidden)) != len(forbidden):
            raise ValueError("forbidden_action_names contains duplicates")
        if type(self.fallback_action) is not CoreAction:
            raise TypeError("fallback_action must be CoreAction")
        if self.fallback_action.name in forbidden:
            raise ValueError("fallback action cannot be forbidden")
        object.__setattr__(self, "forbidden_action_names", forbidden)
        material = (CORE_SCHEMA, "safety-policy-v0", forbidden, self.fallback_action)
        computed = canonical_digest(material)
        if self.digest and self.digest != computed:
            raise ValueError("safety policy digest mismatch")
        object.__setattr__(self, "digest", computed)


DEFAULT_SAFETY_POLICY = SafetyPolicy(("forbidden",), ABSTAIN)


class PermitAuthority:
    """Issue and verify authenticated permits without exposing the session key."""

    def __init__(self, secret_key: bytes) -> None:
        if type(secret_key) is not bytes or len(secret_key) < 16:
            raise ValueError("secret_key must be at least 16 bytes")
        self.__secret_key = bytes(secret_key)

    def _authentication_tag(self, public_claim: object) -> str:
        return hmac.new(
            self.__secret_key,
            canonical_bytes(public_claim),
            hashlib.sha256,
        ).hexdigest()

    def _issue(
        self,
        *,
        session: WorldSession,
        proposal: ActionProposal,
        policy_digest: str,
    ) -> ActionPermit:
        proposal_digest = canonical_digest(proposal)
        nonce = canonical_digest(
            (
                PERMIT_SCHEMA,
                session.episode_id,
                session.tick,
                session.world_commitment,
                canonical_digest(session),
                session.action_space.digest,
                policy_digest,
                proposal_digest,
                proposal.action,
            )
        )
        unsigned = ActionPermit(
            schema=PERMIT_SCHEMA,
            episode_id=session.episode_id,
            tick=session.tick,
            world_commitment=session.world_commitment,
            session_digest=canonical_digest(session),
            action_space_digest=session.action_space.digest,
            policy_digest=policy_digest,
            proposal_digest=proposal_digest,
            action=proposal.action,
            nonce=nonce,
            authentication_tag="unsigned",
        )
        return replace(
            unsigned,
            authentication_tag=self._authentication_tag(unsigned.public_claim()),
        )

    def verify(self, permit: ActionPermit) -> bool:
        if type(permit) is not ActionPermit:
            return False
        expected = self._authentication_tag(permit.public_claim())
        return hmac.compare_digest(expected, permit.authentication_tag)


class SafetyKernel:
    """Immutable policy boundary; it has no update or learner API."""

    def __init__(self, policy: SafetyPolicy, authority: PermitAuthority) -> None:
        if type(policy) is not SafetyPolicy or type(authority) is not PermitAuthority:
            raise TypeError("invalid safety kernel dependency")
        self._policy = policy
        self._authority = authority

    @property
    def policy_digest(self) -> str:
        return self._policy.digest

    def _is_allowed(self, action: CoreAction, action_space: ActionSpace) -> bool:
        return action_space.contains(action) and action.name not in self._policy.forbidden_action_names

    def authorize(
        self,
        proposals: tuple[ActionProposal, ...],
        observation: CoreObservation,
        session: WorldSession,
    ) -> SafetyDecision:
        if type(proposals) is not tuple or not proposals:
            raise TypeError("proposals must be a nonempty tuple")
        if observation.episode_id != session.episode_id or observation.tick != session.tick:
            raise ValueError("observation and session are not aligned")
        selected: ActionProposal | None = None
        rejected = False
        for proposal in proposals:
            if type(proposal) is not ActionProposal:
                raise TypeError("proposals must contain ActionProposal")
            if self._is_allowed(proposal.action, session.action_space):
                selected = proposal
                break
            rejected = True
        if selected is None:
            if not self._is_allowed(self._policy.fallback_action, session.action_space):
                return SafetyDecision(False, ("NO_SAFE_FALLBACK",), None, None)
            selected = ActionProposal(
                action=self._policy.fallback_action,
                expected_score=0.0,
                predicted_risk=0.0,
                model_version=proposals[0].model_version,
                evidence_refs=("safety-fallback",),
            )
            reason_codes = ("FALLBACK_AFTER_REJECTION",)
        else:
            reason_codes = ("ALLOW_AFTER_REJECTION",) if rejected else ("ALLOW",)
        permit = self._authority._issue(
            session=session,
            proposal=selected,
            policy_digest=self._policy.digest,
        )
        return SafetyDecision(True, reason_codes, selected.action, permit)


class _XorRule:
    def reduce(self, state: int, action_value: int) -> int:
        return state ^ action_value


class _SetRule:
    def reduce(self, state: int, action_value: int) -> int:
        del state
        return action_value


class ProceduralWorldAdapter:
    """Pure reducer wrapper with a hidden evaluator-owned transition rule."""

    def __init__(
        self,
        *,
        world_instance_id: str,
        transition_rule: object,
        authority: PermitAuthority,
        policy_digest: str,
        horizon: int = 8,
    ) -> None:
        self.__world_commitment = canonical_digest(
            (CORE_SCHEMA, "world-instance-v0", normalized_text("world_instance_id", world_instance_id))
        )
        if not callable(getattr(transition_rule, "reduce", None)):
            raise TypeError("transition_rule must provide reduce")
        self.__transition_rule = transition_rule
        if type(authority) is not PermitAuthority:
            raise TypeError("authority must be PermitAuthority")
        self.__authority = authority
        self.__policy_digest = normalized_text("policy_digest", policy_digest)
        self.__horizon = exact_int("horizon", horizon, nonnegative=True)
        if self.__horizon == 0:
            raise ValueError("horizon must be positive")

    def genesis(self, request: GenesisRequest) -> tuple[WorldSession, WorldStart]:
        if type(request) is not GenesisRequest:
            raise TypeError("request must be GenesisRequest")
        if request.initial_state not in {0, 1}:
            raise ValueError("V0 initial_state must be binary")
        observation = CoreObservation(
            episode_id=request.episode_id,
            tick=0,
            state=request.initial_state,
            goal=request.goal,
        )
        session = WorldSession(
            episode_id=request.episode_id,
            tick=0,
            state=request.initial_state,
            goal=request.goal,
            world_commitment=self.__world_commitment,
            action_space=DEFAULT_ACTION_SPACE,
            policy_digest=self.__policy_digest,
            used_nonces=(),
            terminated=False,
            transition_count=0,
        )
        genesis_digest = canonical_digest(
            (
                CORE_SCHEMA,
                "genesis",
                observation,
                DEFAULT_ACTION_SPACE.digest,
                self.__world_commitment,
                self.__policy_digest,
            )
        )
        return session, WorldStart(observation, DEFAULT_ACTION_SPACE, genesis_digest)

    def _verify(self, session: WorldSession, permit: ActionPermit) -> None:
        if session.terminated:
            raise PermissionError("episode is not live")
        checks = (
            permit.schema == PERMIT_SCHEMA,
            permit.episode_id == session.episode_id,
            permit.tick == session.tick,
            permit.world_commitment
            == session.world_commitment
            == self.__world_commitment,
            permit.session_digest == canonical_digest(session),
            permit.action_space_digest == session.action_space.digest,
            permit.policy_digest == session.policy_digest == self.__policy_digest,
            session.action_space.contains(permit.action),
            permit.nonce not in session.used_nonces,
            self.__authority.verify(permit),
        )
        if not all(checks):
            raise PermissionError("permit verification failed")

    @staticmethod
    def _action_value(action: CoreAction) -> int | None:
        if action == ABSTAIN:
            return None
        if action.name != "apply":
            raise PermissionError("verified event contains no reducible action")
        value = action.argument("value")
        value = exact_int("action value", value)
        if value not in {0, 1}:
            raise ValueError("V0 action value must be binary")
        return value

    def execute(
        self, session: WorldSession, permit: ActionPermit
    ) -> tuple[WorldSession, WorldStep]:
        if type(session) is not WorldSession or type(permit) is not ActionPermit:
            raise TypeError("execute requires WorldSession and ActionPermit")
        self._verify(session, permit)
        action_value = self._action_value(permit.action)
        next_state = (
            session.state
            if action_value is None
            else exact_int(
                "reducer result",
                self.__transition_rule.reduce(session.state, action_value),
            )
        )
        if next_state not in {0, 1}:
            raise RuntimeError("transition rule left the V0 binary state space")
        next_tick = session.tick + 1
        goal_reached = next_state == session.goal.target_state
        terminated = goal_reached or next_tick >= self.__horizon
        next_session = WorldSession(
            episode_id=session.episode_id,
            tick=next_tick,
            state=next_state,
            goal=session.goal,
            world_commitment=session.world_commitment,
            action_space=session.action_space,
            policy_digest=session.policy_digest,
            used_nonces=session.used_nonces + (permit.nonce,),
            terminated=terminated,
            transition_count=session.transition_count + 1,
        )
        observation = CoreObservation(
            episode_id=session.episode_id,
            tick=next_tick,
            state=next_state,
            goal=session.goal,
        )
        if goal_reached:
            reason = "goal_reached"
        elif terminated:
            reason = "horizon_reached"
        else:
            reason = "transition"
        return next_session, WorldStep(
            observation=observation,
            goal_reached=goal_reached,
            terminated=terminated,
            transition_count=next_session.transition_count,
            public_reason=reason,
        )


def make_xor_world(
    *,
    world_instance_id: str,
    authority: PermitAuthority,
    policy_digest: str,
    horizon: int = 8,
) -> ProceduralWorldAdapter:
    return ProceduralWorldAdapter(
        world_instance_id=world_instance_id,
        transition_rule=_XorRule(),
        authority=authority,
        policy_digest=policy_digest,
        horizon=horizon,
    )


def make_set_world(
    *,
    world_instance_id: str,
    authority: PermitAuthority,
    policy_digest: str,
    horizon: int = 8,
) -> ProceduralWorldAdapter:
    return ProceduralWorldAdapter(
        world_instance_id=world_instance_id,
        transition_rule=_SetRule(),
        authority=authority,
        policy_digest=policy_digest,
        horizon=horizon,
    )


class TabularWorldModel:
    """Small pure learner used only to exercise the V0 closed loop."""

    @staticmethod
    def initialize(observation: CoreObservation) -> BeliefState:
        if type(observation) is not CoreObservation:
            raise TypeError("observation must be CoreObservation")
        return BeliefState((), "tabular-v0-empty")

    @staticmethod
    def infer(observation: CoreObservation, previous: BeliefState) -> BeliefState:
        if type(observation) is not CoreObservation or type(previous) is not BeliefState:
            raise TypeError("invalid infer inputs")
        return previous

    @staticmethod
    def predict(
        belief: BeliefState,
        observation: CoreObservation,
        action: CoreAction,
    ) -> PredictedOutcome:
        if type(belief) is not BeliefState or type(observation) is not CoreObservation:
            raise TypeError("invalid prediction inputs")
        if type(action) is not CoreAction:
            raise TypeError("action must be CoreAction")
        key = canonical_digest(action)
        for state, action_digest, next_state in belief.transitions:
            if state == observation.state and action_digest == key:
                return PredictedOutcome(next_state, 1.0)
        return PredictedOutcome(observation.state, 0.0)

    @classmethod
    def rollout(
        cls,
        belief: BeliefState,
        observation: CoreObservation,
        actions: tuple[CoreAction, ...],
    ) -> tuple[PredictedOutcome, ...]:
        if type(actions) is not tuple or not actions:
            raise TypeError("actions must be a nonempty tuple")
        if any(type(action) is not CoreAction for action in actions):
            raise TypeError("actions must contain only CoreAction values")
        current = observation
        outcomes: list[PredictedOutcome] = []
        for action in actions:
            outcome = cls.predict(belief, current, action)
            outcomes.append(outcome)
            current = CoreObservation(
                episode_id=current.episode_id,
                tick=current.tick + 1,
                state=outcome.next_state,
                goal=current.goal,
            )
        return tuple(outcomes)

    @staticmethod
    def update(
        belief: BeliefState, experience: ExperienceRecord
    ) -> tuple[BeliefState, UpdateReceipt]:
        if type(belief) is not BeliefState or type(experience) is not ExperienceRecord:
            raise TypeError("invalid update inputs")
        key = (experience.before.state, canonical_digest(experience.action))
        transitions = [
            item for item in belief.transitions if (item[0], item[1]) != key
        ]
        transitions.append((key[0], key[1], experience.after.state))
        transition_tuple = tuple(transitions)
        version = canonical_digest(("tabular-v0", transition_tuple))
        next_belief = BeliefState(transition_tuple, version)
        receipt = UpdateReceipt(
            previous_model_version=belief.model_version,
            new_model_version=version,
            experience_digest=canonical_digest(experience),
        )
        return next_belief, receipt


class GreedyOneStepPlanner:
    """Rank bounded one-step predictions without world or evaluator access."""

    @staticmethod
    def rank(
        goal: CoreGoal,
        observation: CoreObservation,
        belief: BeliefState,
        predictor: DynamicsPredictor,
        action_space: ActionSpace,
        budget: int,
    ) -> tuple[ActionProposal, ...]:
        budget = exact_int("planning budget", budget, nonnegative=True)
        if budget == 0:
            raise ValueError("planning budget must be positive")
        proposals: list[ActionProposal] = []
        for action in action_space.actions[:budget]:
            prediction = predictor.rollout(belief, observation, (action,))[0]
            proposals.append(
                ActionProposal(
                    action=action,
                    expected_score=1.0 if prediction.next_state == goal.target_state else 0.0,
                    predicted_risk=0.0,
                    model_version=belief.model_version,
                    evidence_refs=(canonical_digest(prediction),),
                )
            )
        proposals.sort(
            key=lambda proposal: (
                -proposal.expected_score,
                canonical_digest(proposal.action),
            )
        )
        return tuple(proposals)


__all__ = [
    "ABSTAIN",
    "APPLY_ONE",
    "APPLY_ZERO",
    "DEFAULT_ACTION_SPACE",
    "DEFAULT_SAFETY_POLICY",
    "FORBIDDEN",
    "GreedyOneStepPlanner",
    "PermitAuthority",
    "ProceduralWorldAdapter",
    "SafetyKernel",
    "SafetyPolicy",
    "TabularWorldModel",
    "make_set_world",
    "make_xor_world",
]
