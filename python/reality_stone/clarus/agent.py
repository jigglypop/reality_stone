"""Agent loop: Critic, Action, Bootstrap, Consciousness, Working Memory.

Implements 17_AgentLoop.md F.4, F.7, F.9-F.10, F.17, F.20.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

try:
    from .constants import (
        BOOTSTRAP_CONTRACTION,
        ACTIVE_RATIO,
        CRITIC_W_PRED,
        CRITIC_W_CONS,
        CRITIC_W_NOV,
        CONSCIOUSNESS_TAU,
        CONSCIOUSNESS_CD,
        META_MAX_DEPTH,
        WM_CAPACITY,
        CEREBELLUM_ALPHA,
        CEREBELLUM_ETA,
        NORM_EPS,
    )
    from .runtime import BrainRuntime, RuntimeMode, RuntimeStep
    from .belief_control import (
        BeliefControlConfig,
        BeliefController,
        BeliefPlan,
        BeliefUpdate,
    )
    from .adaptive_scc_tower_controller import (
        AdaptiveTowerController,
        CausalEvent,
        PolicyDecision,
        TowerStateToken,
    )
    from .nested_scc_tower import NestedTowerGenerator, TowerSpec
except ImportError:
    from reality_stone.clarus.constants import (
        BOOTSTRAP_CONTRACTION,
        ACTIVE_RATIO,
        CRITIC_W_PRED,
        CRITIC_W_CONS,
        CRITIC_W_NOV,
        CONSCIOUSNESS_TAU,
        CONSCIOUSNESS_CD,
        META_MAX_DEPTH,
        WM_CAPACITY,
        CEREBELLUM_ALPHA,
        CEREBELLUM_ETA,
        NORM_EPS,
    )
    from reality_stone.clarus.runtime import BrainRuntime, RuntimeMode, RuntimeStep
    from reality_stone.clarus.belief_control import (
        BeliefControlConfig,
        BeliefController,
        BeliefPlan,
        BeliefUpdate,
    )
    from reality_stone.clarus.adaptive_scc_tower_controller import (
        AdaptiveTowerController,
        CausalEvent,
        PolicyDecision,
        TowerStateToken,
    )
    from reality_stone.clarus.nested_scc_tower import NestedTowerGenerator, TowerSpec


# ---------------------------------------------------------------------------
# F.4  Critic
# ---------------------------------------------------------------------------
@dataclass
class CriticResult:
    c_pred: float
    c_cons: float
    c_nov: float
    score: float


def compute_critic(
    observation: torch.Tensor,
    prediction: torch.Tensor,
    relaxed_state: torch.Tensor,
    recalled_state: torch.Tensor,
    obs_prior: torch.Tensor | None = None,
    w_pred: float = CRITIC_W_PRED,
    w_cons: float = CRITIC_W_CONS,
    w_nov: float = CRITIC_W_NOV,
) -> CriticResult:
    """c_{t+1} = w_p||c_pred|| + w_c||c_cons|| + w_n||c_nov|| (F.4)."""
    c_pred = float((observation - prediction).norm().item())
    c_cons = float((relaxed_state - recalled_state).norm().item())
    if obs_prior is not None and obs_prior.numel() == observation.numel():
        p_obs = torch.softmax(observation.float(), dim=-1).clamp(min=1e-8)
        p_prior = torch.softmax(obs_prior.float(), dim=-1).clamp(min=1e-8)
        c_nov = float(torch.sum(p_obs * (p_obs.log() - p_prior.log())).item())
        c_nov = max(c_nov, 0.0)
    else:
        c_nov = 0.0
    score = w_pred * c_pred + w_cons * c_cons + w_nov * c_nov
    return CriticResult(c_pred=c_pred, c_cons=c_cons, c_nov=c_nov, score=score)


# ---------------------------------------------------------------------------
# F.7  Action selection
# ---------------------------------------------------------------------------
def select_action_discrete(
    z_out: torch.Tensor,
    action_embeddings: torch.Tensor,
) -> int:
    """a_t = argmax sim(enc(a), z_out) (F.7 discrete)."""
    z_norm = z_out / z_out.norm().clamp(min=NORM_EPS)
    a_norm = action_embeddings / action_embeddings.norm(dim=1, keepdim=True).clamp(min=NORM_EPS)
    similarity = a_norm @ z_norm
    return int(similarity.argmax().item())


def select_action_continuous(
    z_out: torch.Tensor,
    w_act: torch.Tensor,
    b_act: torch.Tensor,
) -> torch.Tensor:
    """a_t = W_act z_out + b_act (F.7 continuous)."""
    return w_act @ z_out + b_act


# ---------------------------------------------------------------------------
# F.9-F.10  Bootstrap operator
# ---------------------------------------------------------------------------
def bootstrap_operator(
    state: torch.Tensor,
    target: torch.Tensor | None = None,
    rho: float = BOOTSTRAP_CONTRACTION,
) -> torch.Tensor:
    """B: X -> p* + rho*(X - p*) (F.9)."""
    if target is None:
        target = torch.zeros_like(state)
    return target + rho * (state - target)


def agent_step(
    x_t: torch.Tensor,
    relax_delta: torch.Tensor,
    obs_delta: torch.Tensor,
    critic_delta: torch.Tensor,
    suppress_delta: torch.Tensor,
    lambda_r: float = 0.3,
    lambda_o: float = 0.2,
    lambda_c: float = 0.2,
    lambda_s: float = 0.1,
    rho: float = BOOTSTRAP_CONTRACTION,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """X_{t+1} = B[X_t + lambda_R*R + lambda_O*dO + lambda_C*C - lambda_S*S] (F.9)."""
    inner = (
        x_t
        + lambda_r * relax_delta
        + lambda_o * obs_delta
        + lambda_c * critic_delta
        - lambda_s * suppress_delta
    )
    return bootstrap_operator(inner, target=target, rho=rho)


# ---------------------------------------------------------------------------
# F.17  Consciousness / Metacognition
# ---------------------------------------------------------------------------
class ConsciousnessMonitor:
    """C3 self-consistency monitor and consciousness depth tracker."""

    def __init__(
        self,
        tau: float = CONSCIOUSNESS_TAU,
        c_d: float = CONSCIOUSNESS_CD,
        max_depth: int = META_MAX_DEPTH,
        rho: float = BOOTSTRAP_CONTRACTION,
    ) -> None:
        self.tau = tau
        self.c_d = c_d
        self.max_depth = max_depth
        self.rho = rho
        self._deviation_history: deque[float] = deque(maxlen=int(tau))

    def record_deviation(self, active_frac: float, target: float = ACTIVE_RATIO) -> None:
        p_current = torch.tensor([active_frac, 1.0 - active_frac])
        p_target = torch.tensor([target, 1.0 - target])
        dev = float((p_current - p_target).norm().item())
        self._deviation_history.append(dev)

    def d_tau(self) -> float:
        """d_tau(t) = (1/tau) * integral of ||p(s) - p*|| ds (F.17.2)."""
        if not self._deviation_history:
            return 0.0
        return sum(self._deviation_history) / max(len(self._deviation_history), 1)

    def consciousness_depth(self) -> float:
        """depth = exp(-c_d * d_tau) (F.17.2). Range [0, 1]."""
        return math.exp(-self.c_d * self.d_tau())

    def metacognition_step(self, deviation: float) -> list[float]:
        """Recursive self-evaluation: d_{n+1} <= rho * d_n (F.17.3)."""
        steps = [deviation]
        d = deviation
        for _ in range(self.max_depth - 1):
            d = self.rho * d
            steps.append(d)
        return steps


# ---------------------------------------------------------------------------
# F.20  Working Memory + Cerebellum
# ---------------------------------------------------------------------------
class WorkingMemory:
    """Finite FIFO buffer |h_t| <= T_h (F.20)."""

    def __init__(self, capacity: int = WM_CAPACITY) -> None:
        self.capacity = capacity
        self._buffer: deque[tuple[Any, Any]] = deque(maxlen=capacity)

    def append(self, action: Any, observation: Any) -> None:
        self._buffer.append((action, observation))

    def contents(self) -> list[tuple[Any, Any]]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


class CerebellumPredictor:
    """Forward model: o_hat_{t+1} = o_hat_t + alpha*(o_t - o_hat_t) (F.20)."""

    def __init__(
        self, dim: int, alpha: float = CEREBELLUM_ALPHA, eta: float = CEREBELLUM_ETA
    ) -> None:
        self.alpha = alpha
        self.eta = eta
        self.prediction = torch.zeros(dim)

    def predict(self) -> torch.Tensor:
        return self.prediction.clone()

    def update(self, observation: torch.Tensor) -> torch.Tensor:
        error = observation.float() - self.prediction
        self.prediction = self.prediction + self.alpha * error
        correction = -self.eta * error
        return correction


@dataclass
class RuntimeAgentConfig:
    """Minimal Layer-F control settings around BrainRuntime."""

    action_count: int = 4
    working_memory_capacity: int = WM_CAPACITY
    lambda_relax: float = 0.3
    lambda_observation: float = 0.2
    lambda_critic: float = 0.2
    lambda_suppress: float = 0.1
    rho: float = BOOTSTRAP_CONTRACTION
    belief_control_enabled: bool = False
    belief_horizon: int = 2
    nested_scc_enabled: bool = False

    def __post_init__(self) -> None:
        self.action_count = max(1, int(self.action_count))
        self.working_memory_capacity = max(1, int(self.working_memory_capacity))
        self.lambda_relax = float(self.lambda_relax)
        self.lambda_observation = float(self.lambda_observation)
        self.lambda_critic = float(self.lambda_critic)
        self.lambda_suppress = float(self.lambda_suppress)
        self.rho = min(max(float(self.rho), 0.0), 1.0)
        self.belief_control_enabled = bool(self.belief_control_enabled)
        self.belief_horizon = max(1, int(self.belief_horizon))
        self.nested_scc_enabled = bool(self.nested_scc_enabled)
        if self.belief_control_enabled and self.nested_scc_enabled:
            raise ValueError("belief control and nested-SCC control cannot be enabled together")


@dataclass
class RuntimeAgentStep:
    """Observable result of one closed-loop Layer-F tick."""

    runtime_step: RuntimeStep
    action_index: int
    critic: CriticResult
    active_ratio: float
    consciousness_depth: float
    working_memory_size: int
    goal_norm: float
    belief_plan: BeliefPlan | None = None
    belief_update: BeliefUpdate | None = None
    nested_scc_token: TowerStateToken | None = None
    nested_scc_policy: PolicyDecision | None = None
    nested_scc_evidence: tuple[float, ...] | None = None
    nested_scc_audit: NestedSCCCascadeAudit | None = None


@dataclass(frozen=True)
class NestedSCCCascadeAudit:
    """Observable boundedness and mediation checks for one V9 action tick."""

    tick: int
    active_depth: int
    evidence_sup_norm: float
    state_sup_norm: float
    probability_mass: float
    bounded: bool
    state_mediated: bool


def default_action_embeddings(action_count: int, dim: int) -> torch.Tensor:
    """Deterministic action codebook used when no task-specific actions exist."""
    action_count = max(1, int(action_count))
    dim = max(1, int(dim))
    embeddings = torch.zeros(action_count, dim)
    for idx in range(action_count):
        embeddings[idx, idx % dim] = 1.0
        embeddings[idx] += (idx + 1) / (10.0 * action_count * dim)
    return embeddings


def cosine_action_evidence(
    observation: torch.Tensor,
    action_embeddings: torch.Tensor,
) -> tuple[float, ...]:
    """Return bounded dimensionless evidence for each fixed action embedding.

    Zero observation or zero action embeddings receive exactly zero evidence.
    This function is an encoder only; the nested-SCC policy must still be read
    from a controller-issued state token.
    """

    if observation.ndim != 1:
        raise ValueError("observation must be a one-dimensional vector")
    if action_embeddings.ndim != 2 or action_embeddings.shape[1] != observation.numel():
        raise ValueError("action_embeddings must have shape (n_actions, observation_dim)")
    if not torch.isfinite(observation).all() or not torch.isfinite(action_embeddings).all():
        raise ValueError("observation and action embeddings must be finite")

    observation_f = observation.detach().float()
    embeddings_f = action_embeddings.detach().float().to(observation_f.device)
    observation_norm = torch.linalg.vector_norm(observation_f)
    embedding_norms = torch.linalg.vector_norm(embeddings_f, dim=1)
    denominators = embedding_norms * observation_norm
    evidence = torch.zeros(embeddings_f.shape[0], device=observation_f.device)
    valid = denominators > 0.0
    if bool(valid.any()):
        evidence[valid] = (embeddings_f[valid] @ observation_f) / denominators[valid]
    evidence = evidence.clamp(-1.0, 1.0)
    if not torch.isfinite(evidence).all():
        raise ValueError("cosine action evidence must be finite")
    return tuple(float(value) for value in evidence.cpu())


class RuntimeAgent:
    """Closed Layer-F loop over BrainRuntime.

    This is an executable research-agent core: runtime relaxation, action
    selection, environment observation, critic, working memory, forward model,
    monitor, and goal feedback happen in one tick.  Execution is not an AGI
    capability claim.
    """

    def __init__(
        self,
        runtime: BrainRuntime,
        *,
        action_embeddings: torch.Tensor | None = None,
        config: RuntimeAgentConfig | None = None,
        belief_controller: BeliefController | None = None,
        nested_scc_controller: AdaptiveTowerController | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or RuntimeAgentConfig()
        if self.config.belief_control_enabled and self.config.nested_scc_enabled:
            raise ValueError("belief control and nested-SCC control cannot be enabled together")
        dim = self.runtime.config.dim
        if action_embeddings is None:
            action_embeddings = default_action_embeddings(self.config.action_count, dim)
        if action_embeddings.ndim != 2 or action_embeddings.shape[1] != dim:
            raise ValueError("action_embeddings must have shape (n_actions, runtime_dim)")
        action_embeddings = action_embeddings.detach().float()
        if not torch.isfinite(action_embeddings).all():
            raise ValueError("action_embeddings must be finite")
        self.action_embeddings = action_embeddings.to(self.runtime.device)
        self.working_memory = WorkingMemory(capacity=self.config.working_memory_capacity)
        self.cerebellum = CerebellumPredictor(dim=dim)
        self.consciousness = ConsciousnessMonitor(rho=self.config.rho)
        self.self_state = torch.zeros(dim, device=self.runtime.device)
        # Critic score from the previous tick gates this tick's STDP plasticity
        # (F.14.2). It is causal: the critic at t needs the relaxed state at t,
        # so it can only drive the learning gate at t+1.
        self._last_critic_score = 0.0
        if self.config.belief_control_enabled:
            self.belief_controller = belief_controller or BeliefController(
                BeliefControlConfig(
                    observation_dim=dim,
                    action_count=self.action_embeddings.shape[0],
                    horizon=self.config.belief_horizon,
                ),
                device=self.runtime.device,
            )
        else:
            # Do not retain or initialize the optional controller.  This keeps
            # the legacy path free of extra RNG calls and state transitions.
            self.belief_controller = None
        if self.config.nested_scc_enabled:
            action_count = int(self.action_embeddings.shape[0])
            if nested_scc_controller is None:
                nested_scc_controller = AdaptiveTowerController(
                    NestedTowerGenerator(TowerSpec(shell_width=action_count))
                )
            if type(nested_scc_controller) is not AdaptiveTowerController:
                raise TypeError("nested_scc_controller must be an AdaptiveTowerController")
            if nested_scc_controller.generator.spec.shell_width != action_count:
                raise ValueError("nested-SCC shell width must equal the action count")
            self.nested_scc_controller = nested_scc_controller
        else:
            # A supplied controller cannot silently affect the legacy path.
            self.nested_scc_controller = None

    def _nested_scc_mask(self, action_mask: Sequence[bool] | None) -> tuple[bool, ...]:
        action_count = int(self.action_embeddings.shape[0])
        if action_mask is None:
            return (True,) * action_count
        mask = tuple(action_mask)
        if len(mask) != action_count or any(type(value) is not bool for value in mask):
            raise ValueError("action_mask must contain one exact boolean per action")
        if not any(mask):
            raise ValueError("action_mask must allow at least one action")
        return mask

    def step(
        self,
        *,
        external_input: torch.Tensor | None = None,
        observation: torch.Tensor | None = None,
        obs_prior: torch.Tensor | None = None,
        force_mode: RuntimeMode | None = None,
        task_goal: torch.Tensor | None = None,
        stdp_learning_signal: float | None = None,
        action_mask: Sequence[bool] | None = None,
    ) -> RuntimeAgentStep:
        if self.belief_controller is not None:
            if task_goal is None:
                raise ValueError("task_goal is required when belief control is enabled")
            goal_check = task_goal.detach().float().view(-1)
            if (
                goal_check.numel() != self.runtime.config.dim
                or not torch.isfinite(goal_check).all()
            ):
                raise ValueError("task_goal must be a finite observation-space vector")
        if self.nested_scc_controller is None:
            if action_mask is not None:
                raise ValueError("action_mask is available only when nested-SCC control is enabled")
            nested_mask = None
        else:
            # Validate every caller-controlled boundary and the live generator
            # seal before BrainRuntime advances.
            nested_mask = self._nested_scc_mask(action_mask)
            self.nested_scc_controller.generator.assert_integrity()
            if observation is not None:
                observation_check = observation.detach().float().view(-1)
                if (
                    observation_check.numel() != self.runtime.config.dim
                    or not torch.isfinite(observation_check).all()
                ):
                    raise ValueError("observation must be a finite runtime-dimension vector")
        runtime_step = self.runtime.step(
            external_input=external_input,
            force_mode=force_mode,
            critic_score=self._last_critic_score,
            learning_signal=stdp_learning_signal,
        )
        relaxed = self.runtime.activation.detach()

        observation_t = (
            relaxed
            if observation is None
            else observation.detach().float().to(self.runtime.device).view(self.runtime.config.dim)
        )
        belief_update = None
        belief_plan = None
        nested_scc_token = None
        nested_scc_policy = None
        nested_scc_evidence = None
        nested_scc_audit = None
        if self.nested_scc_controller is not None:
            nested_scc_evidence = cosine_action_evidence(observation_t, self.action_embeddings)
            nested_scc_token = self.nested_scc_controller.observe(
                CausalEvent(
                    tick=self.nested_scc_controller.tick + 1,
                    observation=nested_scc_evidence,
                )
            )
            nested_scc_policy = self.nested_scc_controller.read_policy(
                nested_scc_token,
                nested_mask,
            )
            # This assignment is the V9 causal boundary: no legacy similarity
            # argmax or external posterior may replace the token readout.
            action_index = nested_scc_policy.selected_action
            state_sup = max(
                float(torch.as_tensor(state).abs().max().item())
                for state in self.nested_scc_controller.state_copy()
            )
            evidence_sup = max(abs(value) for value in nested_scc_evidence)
            probability_mass = sum(nested_scc_policy.probabilities)
            nested_scc_audit = NestedSCCCascadeAudit(
                tick=nested_scc_token.tick,
                active_depth=nested_scc_token.active_depth,
                evidence_sup_norm=evidence_sup,
                state_sup_norm=state_sup,
                probability_mass=probability_mass,
                bounded=(
                    evidence_sup <= 1.0
                    and state_sup <= 1.0
                    and abs(probability_mass - 1.0) <= 1e-12
                ),
                state_mediated=(
                    nested_scc_token is self.nested_scc_controller.latest_token
                    and action_index == nested_scc_policy.selected_action
                ),
            )
        elif self.belief_controller is None:
            action_index = select_action_discrete(relaxed, self.action_embeddings)
        else:
            belief_update = self.belief_controller.observe(observation_t)
            belief_plan = self.belief_controller.plan(
                observation_t,
                task_goal,
                action_free_base_transition=lambda obs: obs.clone(),
            )
            action_index = belief_plan.action_index
        prediction = self.cerebellum.predict().to(self.runtime.device)
        recalled = self.runtime.hippocampus.recall(relaxed).to(self.runtime.device)
        prior_t = None if obs_prior is None else obs_prior.detach().float().to(self.runtime.device)
        critic = compute_critic(observation_t, prediction, relaxed, recalled, obs_prior=prior_t)
        # Feed this critic into the next tick's STDP learning gate (F.14.2).
        self._last_critic_score = float(critic.score)

        correction = self.cerebellum.update(observation_t.detach().cpu()).to(self.runtime.device)
        self.working_memory.append(action_index, observation_t.detach().cpu())

        active_ratio = runtime_step.active_modules / max(self.runtime.config.dim, 1)
        self.consciousness.record_deviation(active_ratio)

        critic_delta = F.normalize(observation_t - prediction, dim=0) * critic.score
        self.self_state = agent_step(
            self.self_state,
            relax_delta=relaxed - self.self_state,
            obs_delta=observation_t - relaxed,
            critic_delta=critic_delta,
            suppress_delta=self.runtime.refractory + correction,
            lambda_r=self.config.lambda_relax,
            lambda_o=self.config.lambda_observation,
            lambda_c=self.config.lambda_critic,
            lambda_s=self.config.lambda_suppress,
            rho=self.config.rho,
        )
        self.runtime.set_goal(self.self_state)
        if self.belief_controller is not None and belief_plan is not None:
            # The identity prediction is the Loop-1 action-free persistence
            # baseline.  The committed action is learned only at next observe.
            self.belief_controller.commit(
                belief_plan,
                base_prediction=observation_t,
            )

        return RuntimeAgentStep(
            runtime_step=runtime_step,
            action_index=action_index,
            critic=critic,
            active_ratio=active_ratio,
            consciousness_depth=self.consciousness.consciousness_depth(),
            working_memory_size=len(self.working_memory),
            goal_norm=float(self.runtime.goal.norm().item()),
            belief_plan=belief_plan,
            belief_update=belief_update,
            nested_scc_token=nested_scc_token,
            nested_scc_policy=nested_scc_policy,
            nested_scc_evidence=nested_scc_evidence,
            nested_scc_audit=nested_scc_audit,
        )


@dataclass
class TextEnvironmentStep:
    """Result of applying one discrete agent action to a text environment."""

    prompt: str
    action_label: str
    response: str
    observation: torch.Tensor
    done: bool = False
    info: dict[str, float | int | str] = field(default_factory=dict)


class TextEnvironment:
    """Deterministic text-to-vector environment for RuntimeAgent episodes."""

    def __init__(
        self,
        dim: int,
        *,
        actions: list[str] | None = None,
        history_limit: int = 8,
    ) -> None:
        self.dim = max(1, int(dim))
        self.actions = actions or ["answer", "clarify", "recall", "reflect"]
        if not self.actions:
            raise ValueError("TextEnvironment requires at least one action")
        self.history: deque[tuple[str, str]] = deque(maxlen=max(1, int(history_limit)))
        self.current_prompt = ""

    def encode(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        terms = [part for part in str(text).replace("\n", " ").split(" ") if part]
        if not terms:
            return vec
        for position, term in enumerate(terms):
            digest = hashlib.blake2b(f"{position}:{term}".encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign / math.sqrt(position + 1.0)
        norm = vec.norm()
        if norm.item() < NORM_EPS:
            return vec
        return vec / norm

    def action_embeddings(self) -> torch.Tensor:
        return torch.stack([self.encode(action) for action in self.actions], dim=0)

    def reset(self, prompt: str) -> torch.Tensor:
        self.current_prompt = str(prompt)
        return self.encode(self.current_prompt)

    def step(self, action_index: int, state: torch.Tensor | None = None) -> TextEnvironmentStep:
        idx = int(action_index) % len(self.actions)
        action = self.actions[idx]
        response = self._render_response(action, state)
        self.history.append((action, response))
        observation_text = f"{self.current_prompt}\naction: {action}\nresponse: {response}"
        observation = self.encode(observation_text)
        return TextEnvironmentStep(
            prompt=self.current_prompt,
            action_label=action,
            response=response,
            observation=observation,
            done=False,
            info={"history_size": len(self.history)},
        )

    def _render_response(self, action: str, state: torch.Tensor | None) -> str:
        state_norm = 0.0 if state is None else float(state.detach().float().norm().item())
        if action == "clarify":
            return f"clarify: {self.current_prompt}"
        if action == "recall":
            last = "none" if not self.history else self.history[-1][1]
            return f"recall: {last}"
        if action == "reflect":
            return f"reflect: state_norm={state_norm:.4f}"
        return f"answer: {self.current_prompt}"


@dataclass
class RuntimeTextAgentTurn:
    """One text-environment turn containing both agent and environment outputs."""

    agent_step: RuntimeAgentStep
    env_step: TextEnvironmentStep


class RuntimeTextAgent:
    """RuntimeAgent wired to a deterministic text environment."""

    def __init__(
        self,
        runtime: BrainRuntime,
        *,
        environment: TextEnvironment | None = None,
        config: RuntimeAgentConfig | None = None,
    ) -> None:
        env = environment or TextEnvironment(runtime.config.dim)
        agent_config = config or RuntimeAgentConfig(action_count=len(env.actions))
        self.environment = env
        self.agent = RuntimeAgent(
            runtime,
            action_embeddings=env.action_embeddings(),
            config=agent_config,
        )

    def ask(
        self,
        prompt: str,
        *,
        ticks: int = 1,
        force_mode: RuntimeMode = RuntimeMode.WAKE,
    ) -> RuntimeTextAgentTurn:
        observation = self.environment.reset(prompt)
        prior = None
        last_turn = None
        for _ in range(max(1, int(ticks))):
            agent_out = self.agent.step(
                external_input=observation,
                observation=observation,
                obs_prior=prior,
                force_mode=force_mode,
            )
            env_out = self.environment.step(agent_out.action_index, self.agent.self_state)
            prior = observation
            observation = env_out.observation
            last_turn = RuntimeTextAgentTurn(agent_step=agent_out, env_step=env_out)
        if last_turn is None:
            raise RuntimeError("RuntimeTextAgent did not execute a turn")
        return last_turn
