"""Protocol-parametric event-sourced orchestrator for AGI Core V0."""

from __future__ import annotations

import hashlib

from .contracts import (
    ActionExecutor,
    CorePlanner,
    MemoryStore,
    SafetyBoundary,
    WorldAdapter,
    WorldModel,
)
from .records import (
    CORE_SCHEMA,
    ActionProposal,
    BeliefState,
    CoreRuntimeState,
    CoreStepResult,
    DecisionDraft,
    ExperienceRecord,
    GenesisRequest,
    LedgerEntry,
    SafetyDecision,
    UpdateReceipt,
    WorldSession,
    WorldStart,
    WorldStep,
    canonical_bytes,
    canonical_digest,
    canonical_map,
    exact_int,
)


_GENESIS_PREVIOUS_DIGEST = "0" * 64


def _validate_genesis(
    request: GenesisRequest,
    session: object,
    start: object,
    policy_digest: str,
) -> tuple[WorldSession, WorldStart]:
    if type(session) is not WorldSession or type(start) is not WorldStart:
        raise TypeError("world genesis returned invalid record types")
    expected = request_to_observation(request)
    checks = (
        start.observation == expected,
        session.episode_id == expected.episode_id,
        session.tick == expected.tick == 0,
        session.state == expected.state,
        session.goal == expected.goal,
        session.action_space == start.action_space,
        session.policy_digest == policy_digest,
        session.used_nonces == (),
        not session.terminated,
        session.transition_count == 0,
    )
    if not all(checks):
        raise RuntimeError("world genesis records are inconsistent")
    return session, start


def _validate_transition(
    before: CoreRuntimeState,
    permit_nonce: str,
    next_session: object,
    world_step: object,
) -> tuple[WorldSession, WorldStep]:
    if type(next_session) is not WorldSession or type(world_step) is not WorldStep:
        raise TypeError("executor returned invalid record types")
    expected_nonces = tuple(sorted(before.session.used_nonces + (permit_nonce,)))
    observation = world_step.observation
    checks = (
        next_session.episode_id == before.session.episode_id == observation.episode_id,
        next_session.tick == before.session.tick + 1 == observation.tick,
        next_session.state == observation.state,
        next_session.goal == before.session.goal == observation.goal,
        next_session.world_commitment == before.session.world_commitment,
        next_session.action_space == before.session.action_space,
        next_session.policy_digest == before.session.policy_digest,
        next_session.used_nonces == expected_nonces,
        next_session.transition_count == before.session.transition_count + 1,
        world_step.transition_count == next_session.transition_count,
        world_step.terminated == next_session.terminated,
        world_step.goal_reached == (observation.state == observation.goal.target_state),
        not world_step.goal_reached or world_step.terminated,
    )
    if not all(checks):
        raise RuntimeError("executor returned inconsistent transition records")
    return next_session, world_step


def _validate_update(
    belief: BeliefState,
    experience: ExperienceRecord,
    next_belief: object,
    receipt: object,
) -> tuple[BeliefState, UpdateReceipt]:
    if type(next_belief) is not BeliefState or type(receipt) is not UpdateReceipt:
        raise TypeError("memory returned invalid update record types")
    checks = (
        receipt.previous_model_version == belief.model_version,
        receipt.new_model_version == next_belief.model_version,
        receipt.experience_digest == canonical_digest(experience),
    )
    if not all(checks):
        raise RuntimeError("memory returned an inconsistent update receipt")
    return next_belief, receipt


def _entry_digest(
    *,
    index: int,
    event_type: str,
    payload: tuple[tuple[str, object], ...],
    previous_digest: str,
) -> str:
    material = (
        CORE_SCHEMA,
        index,
        event_type,
        payload,
        previous_digest,
    )
    return hashlib.sha256(canonical_bytes(material)).hexdigest()


def append_ledger_entry(
    ledger: tuple[LedgerEntry, ...],
    *,
    event_type: str,
    payload: tuple[tuple[str, object], ...],
) -> tuple[LedgerEntry, ...]:
    """Return a new hash-chained ledger; never mutate the input tuple."""

    if type(ledger) is not tuple or any(type(entry) is not LedgerEntry for entry in ledger):
        raise TypeError("ledger must be a tuple of LedgerEntry")
    index = len(ledger)
    previous_digest = ledger[-1].digest if ledger else _GENESIS_PREVIOUS_DIGEST
    checked_payload = canonical_map("ledger payload", payload)
    entry = LedgerEntry(
        index=index,
        event_type=event_type,
        payload=checked_payload,
        previous_digest=previous_digest,
        digest=_entry_digest(
            index=index,
            event_type=event_type,
            payload=checked_payload,
            previous_digest=previous_digest,
        ),
    )
    return ledger + (entry,)


def verify_ledger(ledger: tuple[LedgerEntry, ...]) -> bool:
    """Verify indices, predecessor links, and every public digest."""

    if type(ledger) is not tuple or not ledger:
        return False
    previous_digest = _GENESIS_PREVIOUS_DIGEST
    for index, entry in enumerate(ledger):
        if type(entry) is not LedgerEntry:
            return False
        if entry.index != index or entry.previous_digest != previous_digest:
            return False
        expected = _entry_digest(
            index=index,
            event_type=entry.event_type,
            payload=entry.payload,
            previous_digest=previous_digest,
        )
        if entry.digest != expected:
            return False
        previous_digest = entry.digest
    return True


class CoreOrchestrator:
    """Compose protocols without importing or inspecting a concrete world family."""

    def __init__(
        self,
        *,
        world: WorldAdapter,
        executor: ActionExecutor,
        memory: MemoryStore,
        world_model: WorldModel,
        planner: CorePlanner,
        safety: SafetyBoundary,
        planning_budget: int,
    ) -> None:
        self._world = world
        self._executor = executor
        self._memory = memory
        self._world_model = world_model
        self._planner = planner
        self._safety = safety
        self._planning_budget = exact_int(
            "planning_budget", planning_budget, nonnegative=True
        )
        if self._planning_budget == 0:
            raise ValueError("planning_budget must be positive")

    def genesis(self, request: GenesisRequest) -> CoreRuntimeState:
        """Start one new episode with an explicit lifecycle event."""

        if type(request) is not GenesisRequest:
            raise TypeError("request must be a GenesisRequest")
        session, start = self._world.genesis(request)
        session, start = _validate_genesis(
            request, session, start, self._safety.policy_digest
        )
        belief = self._memory.initialize(start.observation)
        payload = canonical_map(
            "genesis payload",
            (
                ("action_space_digest", start.action_space.digest),
                ("episode_id", start.observation.episode_id),
                ("genesis_digest", start.genesis_digest),
                ("goal", canonical_digest(start.observation.goal)),
                ("observation", canonical_digest(start.observation)),
                ("schema", CORE_SCHEMA),
            ),
        )
        ledger = append_ledger_entry((), event_type="genesis", payload=payload)
        return CoreRuntimeState(
            session=session,
            observation=start.observation,
            belief=belief,
            ledger=ledger,
        )

    def _decision_components(
        self, state: CoreRuntimeState
    ) -> tuple[BeliefState, tuple[ActionProposal, ...], SafetyDecision]:
        if state.session.terminated:
            raise RuntimeError("cannot decide after episode termination")
        belief = self._memory.infer(state.observation, state.belief)
        proposals = self._planner.rank(
            state.observation.goal,
            state.observation,
            belief,
            self._world_model,
            state.session.action_space,
            self._planning_budget,
        )
        if type(proposals) is not tuple or not proposals:
            raise TypeError("planner must return a nonempty tuple")
        if any(type(proposal) is not ActionProposal for proposal in proposals):
            raise TypeError("planner returned a non-ActionProposal")
        if any(proposal.model_version != belief.model_version for proposal in proposals):
            raise RuntimeError("planner proposal model version is stale or inconsistent")
        decision = self._safety.authorize(proposals, state.observation, state.session)
        if type(decision) is not SafetyDecision:
            raise TypeError("safety boundary returned an invalid decision")
        return belief, proposals, decision

    def decide(self, state: CoreRuntimeState) -> DecisionDraft:
        """Return the pre-transition decision for noninterference tests and audit."""

        _, proposals, decision = self._decision_components(state)
        return DecisionDraft(proposals=proposals, safety_decision=decision)

    def step(self, state: CoreRuntimeState) -> tuple[CoreRuntimeState, CoreStepResult]:
        """Apply exactly one verified event and return a new immutable state."""

        before_policy = self._safety.policy_digest
        belief, proposals, decision = self._decision_components(state)
        if not decision.allowed or decision.permit is None or decision.selected_action is None:
            raise PermissionError("safety boundary denied all actions")
        next_session, world_step = self._executor.execute(
            state.session, decision.permit
        )
        next_session, world_step = _validate_transition(
            state,
            decision.permit.nonce,
            next_session,
            world_step,
        )
        experience = ExperienceRecord(
            before=state.observation,
            belief=belief,
            action=decision.selected_action,
            after=world_step.observation,
            terminated=world_step.terminated,
        )
        next_belief, receipt = self._memory.update(belief, experience)
        next_belief, receipt = _validate_update(
            belief, experience, next_belief, receipt
        )
        if self._safety.policy_digest != before_policy:
            raise RuntimeError("safety policy changed during an agent tick")
        proposal_digests = tuple(canonical_digest(proposal) for proposal in proposals)
        payload = canonical_map(
            "step payload",
            (
                ("action", canonical_digest(decision.selected_action)),
                ("after_observation", canonical_digest(world_step.observation)),
                ("before_observation", canonical_digest(state.observation)),
                ("experience_digest", receipt.experience_digest),
                ("model_version", receipt.new_model_version),
                ("permit_claim_digest", decision.permit.public_claim_digest()),
                ("proposal_digests", proposal_digests),
                ("safety_reason_codes", decision.reason_codes),
                ("transition_verified", True),
            ),
        )
        ledger = append_ledger_entry(state.ledger, event_type="agent_transition", payload=payload)
        next_state = CoreRuntimeState(
            session=next_session,
            observation=world_step.observation,
            belief=next_belief,
            ledger=ledger,
        )
        result = CoreStepResult(
            world_step=world_step,
            update_receipt=receipt,
            public_entry=ledger[-1],
        )
        return next_state, result


def request_to_observation(request: GenesisRequest):
    """Build the only learner-visible projection required by the V0 genesis."""

    from .records import CoreObservation

    return CoreObservation(
        episode_id=request.episode_id,
        tick=0,
        state=request.initial_state,
        goal=request.goal,
    )


__all__ = [
    "CoreOrchestrator",
    "append_ledger_entry",
    "request_to_observation",
    "verify_ledger",
]
