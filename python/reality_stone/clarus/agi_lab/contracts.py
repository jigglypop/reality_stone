"""Protocol-only boundaries for the physics-independent AGI Core V0."""

from __future__ import annotations

from typing import Protocol

from .records import (
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
)


class BeliefEngine(Protocol):
    def initialize(self, observation: CoreObservation) -> BeliefState: ...

    def infer(
        self, observation: CoreObservation, previous: BeliefState
    ) -> BeliefState: ...


class DynamicsPredictor(Protocol):
    def predict(
        self,
        belief: BeliefState,
        observation: CoreObservation,
        action: CoreAction,
    ) -> PredictedOutcome: ...

    def rollout(
        self,
        belief: BeliefState,
        observation: CoreObservation,
        actions: tuple[CoreAction, ...],
    ) -> tuple[PredictedOutcome, ...]: ...


class WorldModel(DynamicsPredictor, Protocol):
    """Replaceable predictive-model role exposed to the planner."""


class OnlineLearner(Protocol):
    def update(
        self, belief: BeliefState, experience: ExperienceRecord
    ) -> tuple[BeliefState, UpdateReceipt]: ...


class MemoryStore(BeliefEngine, OnlineLearner, Protocol):
    """Replaceable immutable-memory role for inference and experience commits."""


class CorePlanner(Protocol):
    def rank(
        self,
        goal: CoreGoal,
        observation: CoreObservation,
        belief: BeliefState,
        predictor: DynamicsPredictor,
        action_space: ActionSpace,
        budget: int,
    ) -> tuple[ActionProposal, ...]: ...


class SafetyBoundary(Protocol):
    @property
    def policy_digest(self) -> str: ...

    def authorize(
        self,
        proposals: tuple[ActionProposal, ...],
        observation: CoreObservation,
        session: WorldSession,
    ) -> SafetyDecision: ...


class ActionExecutor(Protocol):
    def execute(
        self, session: WorldSession, permit: ActionPermit
    ) -> tuple[WorldSession, WorldStep]: ...


class WorldAdapter(Protocol):
    def genesis(self, request: GenesisRequest) -> tuple[WorldSession, WorldStart]: ...


__all__ = [
    "ActionExecutor",
    "BeliefEngine",
    "CorePlanner",
    "DynamicsPredictor",
    "MemoryStore",
    "OnlineLearner",
    "SafetyBoundary",
    "WorldAdapter",
    "WorldModel",
]
