"""Isolated physics-independent AGI Core V0 research scaffold.

This package is intentionally not re-exported from :mod:`reality_stone.clarus`.
It provides mechanism-level contracts only; it is not evidence of AGI,
consciousness, a brain algorithm, SCC necessity, or Riemannian necessity.
"""

from .contracts import (
    ActionExecutor,
    BeliefEngine,
    CorePlanner,
    DynamicsPredictor,
    MemoryStore,
    OnlineLearner,
    SafetyBoundary,
    WorldAdapter,
    WorldModel,
)
from .orchestrator import CoreOrchestrator, append_ledger_entry, verify_ledger
from .procedural_world import (
    ABSTAIN,
    APPLY_ONE,
    APPLY_ZERO,
    DEFAULT_SAFETY_POLICY,
    FORBIDDEN,
    GreedyOneStepPlanner,
    PermitAuthority,
    SafetyKernel,
    TabularWorldModel,
    make_set_world,
    make_xor_world,
)
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
    CoreRuntimeState,
    CoreStepResult,
    DecisionDraft,
    GenesisRequest,
    LedgerEntry,
    SafetyDecision,
    WorldSession,
    canonical_bytes,
    canonical_digest,
    dataclass_public_fields,
    public_ledger_bytes,
)


IMPLEMENTATION_STATUS = "PHYSICS_INDEPENDENT_CORE_SCAFFOLD"
PHYSICAL_AGI_CLAIM = False
CONSCIOUSNESS_CLAIM = False
BRAIN_ALGORITHM_CLAIM = False


__all__ = [
    "CORE_SCHEMA",
    "PERMIT_SCHEMA",
    "ABSTAIN",
    "APPLY_ONE",
    "APPLY_ZERO",
    "IMPLEMENTATION_STATUS",
    "PHYSICAL_AGI_CLAIM",
    "CONSCIOUSNESS_CLAIM",
    "BRAIN_ALGORITHM_CLAIM",
    "ActionPermit",
    "ActionProposal",
    "ActionSpace",
    "ActionExecutor",
    "BeliefEngine",
    "BeliefState",
    "CoreAction",
    "CoreGoal",
    "CoreObservation",
    "CoreOrchestrator",
    "CorePlanner",
    "CoreRuntimeState",
    "CoreStepResult",
    "DEFAULT_SAFETY_POLICY",
    "DecisionDraft",
    "DynamicsPredictor",
    "FORBIDDEN",
    "GenesisRequest",
    "GreedyOneStepPlanner",
    "LedgerEntry",
    "MemoryStore",
    "OnlineLearner",
    "PermitAuthority",
    "SafetyBoundary",
    "SafetyDecision",
    "SafetyKernel",
    "TabularWorldModel",
    "WorldAdapter",
    "WorldModel",
    "WorldSession",
    "append_ledger_entry",
    "canonical_bytes",
    "canonical_digest",
    "dataclass_public_fields",
    "make_set_world",
    "make_xor_world",
    "public_ledger_bytes",
    "verify_ledger",
]
