"""Learnable local/cloud recurrent operator with a structural small-gain bound."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LearnableSmallGainConfig:
    local_budget: float = 0.82
    cloud_budget: float = 0.82
    cloud_to_local_budget: float = 0.06
    local_to_cloud_budget: float = 0.04
    interaction_budget: float = 0.08
    contraction_cap: float = 0.99

    def __post_init__(self) -> None:
        for name in (
            "local_budget",
            "cloud_budget",
            "cloud_to_local_budget",
            "local_to_cloud_budget",
            "interaction_budget",
            "contraction_cap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if not 0.0 < self.contraction_cap < 1.0:
            raise ValueError("contraction_cap must lie in (0, 1)")
        if not self.certificate()["certified"]:
            raise ValueError("declared local/cloud gain matrix is not contractive")

    def certificate(self) -> dict[str, object]:
        matrix = np.asarray(
            (
                (
                    self.local_budget + self.interaction_budget,
                    self.cloud_to_local_budget + self.interaction_budget,
                ),
                (self.local_to_cloud_budget, self.cloud_budget),
            ),
            dtype=np.float64,
        )
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        if spectral_radius >= 1.0:
            return {
                "matrix": tuple(map(tuple, matrix)),
                "spectral_radius": spectral_radius,
                "q": math.inf,
                "weights": (math.inf, math.inf),
                "certified": False,
            }
        weights = np.linalg.solve(np.eye(2) - matrix, np.ones(2))
        q = float(np.max(matrix @ weights / weights))
        return {
            "matrix": tuple(map(tuple, matrix)),
            "spectral_radius": spectral_radius,
            "q": q,
            "weights": tuple(float(value) for value in weights),
            "certified": bool(q <= self.contraction_cap),
        }


def _row_l1_normalized(raw: torch.Tensor, budget: float) -> torch.Tensor:
    row_sum = torch.sum(torch.abs(raw), dim=1, keepdim=True)
    return budget * raw / torch.clamp(row_sum, min=1.0)


class LearnableSmallGainLocalCloud(nn.Module):
    """Twenty-state structured recurrent classifier.

    Recurrent and cross-state matrices are row-L1 normalized to fixed budgets.
    Input operators are unconstrained because they do not affect contraction in
    the recurrent state for a frozen input sequence.
    """

    def __init__(self, config: LearnableSmallGainConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else LearnableSmallGainConfig()
        self.local_raw = nn.Parameter(torch.empty(4, 4))
        self.cloud_raw = nn.Parameter(torch.empty(4, 4))
        self.cloud_to_local_raw = nn.Parameter(torch.empty(4, 4))
        self.local_to_cloud_raw = nn.Parameter(torch.empty(4, 4))
        self.interaction_left_raw = nn.Parameter(torch.empty(4, 4))
        self.interaction_right_raw = nn.Parameter(torch.empty(4, 4))
        self.interaction_out_raw = nn.Parameter(torch.empty(4, 4))
        self.local_input = nn.Linear(4, 4, bias=False)
        self.cloud_input = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(20, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.local_raw,
            self.cloud_raw,
            self.cloud_to_local_raw,
            self.local_to_cloud_raw,
            self.interaction_left_raw,
            self.interaction_right_raw,
            self.interaction_out_raw,
        ):
            nn.init.orthogonal_(parameter)
        nn.init.xavier_uniform_(self.local_input.weight)
        nn.init.xavier_uniform_(self.cloud_input.weight)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _operators(self) -> tuple[torch.Tensor, ...]:
        config = self.config
        return (
            _row_l1_normalized(self.local_raw, config.local_budget),
            _row_l1_normalized(self.cloud_raw, config.cloud_budget),
            _row_l1_normalized(self.cloud_to_local_raw, config.cloud_to_local_budget),
            _row_l1_normalized(self.local_to_cloud_raw, config.local_to_cloud_budget),
            _row_l1_normalized(self.interaction_left_raw, 1.0),
            _row_l1_normalized(self.interaction_right_raw, 1.0),
            _row_l1_normalized(self.interaction_out_raw, config.interaction_budget),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[2] != 20:
            raise ValueError("sequence must have shape (batch, ticks, 20)")
        local = torch.zeros(sequence.shape[0], 4, 4, dtype=sequence.dtype, device=sequence.device)
        cloud = torch.zeros(sequence.shape[0], 4, dtype=sequence.dtype, device=sequence.device)
        local_op, cloud_op, c_to_l, l_to_c, left, right, out = self._operators()
        for tick in range(sequence.shape[1]):
            local_input = sequence[:, tick, :16].reshape(-1, 4, 4)
            shared_input = sequence[:, tick, 16:]
            old_local = local
            old_cloud = cloud
            local_projected = torch.matmul(old_local, left.T)
            cloud_projected = torch.matmul(old_cloud, right.T).unsqueeze(1)
            interaction = torch.matmul(local_projected * cloud_projected, out.T)
            local = torch.tanh(
                torch.matmul(old_local, local_op.T)
                + self.local_input(local_input)
                + torch.matmul(old_cloud, c_to_l.T).unsqueeze(1)
                + interaction
            )
            cloud = torch.tanh(
                torch.matmul(old_cloud, cloud_op.T)
                + self.cloud_input(shared_input)
                + torch.matmul(torch.mean(old_local, dim=1), l_to_c.T)
            )
        features = torch.cat((local.reshape(sequence.shape[0], 16), cloud), dim=1)
        return self.readout(features).squeeze(-1)

    def operator_row_sums(self) -> tuple[float, ...]:
        return tuple(
            float(torch.max(torch.sum(torch.abs(operator), dim=1)).detach().cpu())
            for operator in self._operators()
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def model_hash(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def train_learnable_small_gain(
    train_x: np.ndarray,
    train_y: Sequence[int],
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
    config: LearnableSmallGainConfig | None = None,
) -> LearnableSmallGainLocalCloud:
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError("seed and epochs must be exact valid integers")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = LearnableSmallGainLocalCloud(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    features = torch.as_tensor(train_x, dtype=torch.float32)
    labels = torch.as_tensor(
        (np.asarray(tuple(train_y), dtype=np.float32) + 1.0) / 2.0,
        dtype=torch.float32,
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite learnable small-gain loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    return model


class StableBilinearLocalCloud(nn.Module):
    """Small stable local/cloud memory with an explicit bilinear policy.

    The recurrent transition is block diagonal and contractive. Conditional
    local/shared binding occurs in the policy readout, where it is directly
    identifiable instead of being forced through many weak recurrent products.
    """

    def __init__(
        self,
        retention_cap: float = 0.995,
        salience_threshold: float = 0.40,
        salience_sharpness: float = 30.0,
    ) -> None:
        super().__init__()
        if not 0.0 < retention_cap < 1.0:
            raise ValueError("retention_cap must lie in (0, 1)")
        if not 0.0 < salience_threshold < 1.0 or salience_sharpness <= 0.0:
            raise ValueError("salience gate parameters are outside their valid domain")
        self.retention_cap = float(retention_cap)
        self.salience_threshold = float(salience_threshold)
        self.salience_sharpness = float(salience_sharpness)
        self.local_retention_raw = nn.Parameter(torch.full((4,), 3.5))
        self.cloud_retention_raw = nn.Parameter(torch.full((4,), 3.5))
        self.local_input_scale = nn.Parameter(torch.ones(4))
        self.cloud_input_scale = nn.Parameter(torch.ones(4))
        self.bilinear = nn.Parameter(torch.empty(4, 4))
        self.linear = nn.Linear(20, 1)
        nn.init.xavier_uniform_(self.bilinear)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def retentions(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.retention_cap * torch.sigmoid(self.local_retention_raw),
            self.retention_cap * torch.sigmoid(self.cloud_retention_raw),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[2] != 20:
            raise ValueError("sequence must have shape (batch, ticks, 20)")
        local = torch.zeros(sequence.shape[0], 4, 4, dtype=sequence.dtype, device=sequence.device)
        cloud = torch.zeros(sequence.shape[0], 4, dtype=sequence.dtype, device=sequence.device)
        local_retention, cloud_retention = self.retentions()
        for tick in range(sequence.shape[1]):
            local_input = sequence[:, tick, :16].reshape(-1, 4, 4)
            shared_input = sequence[:, tick, 16:]
            local_gate = torch.sigmoid(
                self.salience_sharpness * (torch.abs(local_input) - self.salience_threshold)
            )
            cloud_gate = torch.sigmoid(
                self.salience_sharpness * (torch.abs(shared_input) - self.salience_threshold)
            )
            local_candidate = torch.tanh(local_input * self.local_input_scale)
            cloud_candidate = torch.tanh(shared_input * self.cloud_input_scale)
            local = (1.0 - local_gate) * local_retention * local + local_gate * local_candidate
            cloud = (1.0 - cloud_gate) * cloud_retention * cloud + cloud_gate * cloud_candidate
        features = torch.cat((local.reshape(sequence.shape[0], 16), cloud), dim=1)
        interaction = torch.sum(local * cloud.unsqueeze(1) * self.bilinear, dim=(1, 2))
        return interaction + self.linear(features).squeeze(-1)

    def contraction_factor(self) -> float:
        local, cloud = self.retentions()
        return float(torch.max(torch.cat((local, cloud))).detach().cpu())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def train_stable_bilinear(
    train_x: np.ndarray,
    train_y: Sequence[int],
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
) -> StableBilinearLocalCloud:
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError("seed and epochs must be exact valid integers")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = StableBilinearLocalCloud()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    features = torch.as_tensor(train_x, dtype=torch.float32)
    labels = torch.as_tensor(
        (np.asarray(tuple(train_y), dtype=np.float32) + 1.0) / 2.0,
        dtype=torch.float32,
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite stable-bilinear loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    return model
