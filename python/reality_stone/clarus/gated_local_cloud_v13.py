"""V13 gated local/cloud recurrent classifier.

Diagnosed defect being addressed here (V13 scope item 1 and 3): the V10
kernel (`local_cloud_kernel.py`) has zero learned transition parameters and a
fixed retention of 0.62 with a `contraction_cap` of 0.95, which forces signal
decay (0.62**7 ~= 0.035x at T=8) even on tasks that need retention -> 1. The
V12 candidate (`StableBilinearLocalCloud` in
`learnable_small_gain_local_cloud.py`) added a learned input-dependent gate
but no recurrent mixing matrix, i.e. it is a per-dimension GRU-style forget
gate without genuine within-cell recurrent mixing.

This module keeps the same 20-scalar local/cloud state budget (4 local cells
x width 4, plus one cloud cell of width 4) and adds:

  1. A recurrent mixing matrix W_rec inside the candidate branch (absent from
     V12), so the candidate state genuinely depends on the *matrix* action of
     the previous hidden state, not just an elementwise retention.
  2. An explicit local<->cloud interaction term U(h_local (*) c), mirroring
     the V10 kernel's interaction_gain term structurally.
  3. A relaxed retention cap of 1.0 (marginal stability permitted), per
     explicit user authorization to relax the V10/V12 contraction constraint.
     Because this is no longer a certified contraction, the retention values
     and an approximate Lipschitz bound are exposed only as *observational*
     metrics (`lipschitz_report`), not as a pass/fail certificate gate.

No target, label, or oracle signal enters the transition; the classifier only
reads the recurrent state at the final tick, mirroring the V10/V11/V12
evaluation contract.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GatedLocalCloudV13(nn.Module):
    """Twenty-scalar local/cloud state with a learned gated, recurrently
    mixed transition and a relaxed (uncertified) retention cap."""

    def __init__(self, retention_cap: float = 1.0) -> None:
        super().__init__()
        if not 0.0 < retention_cap <= 1.0:
            raise ValueError("retention_cap must lie in (0, 1]")
        self.retention_cap = float(retention_cap)

        # Local-cell branch: weights are shared across the four local cells
        # (mirroring the V10 kernel's shared per-cell formula) to keep the
        # parameter budget small while still exposing genuine recurrent
        # mixing within each cell's own width-4 state.
        self.local_input = nn.Linear(4, 4)
        self.local_rec = nn.Linear(4, 4, bias=False)
        self.cloud_to_local = nn.Linear(4, 4, bias=False)
        self.local_interaction = nn.Linear(4, 4, bias=False)
        self.local_gate = nn.Linear(4, 4)
        self.local_retention_raw = nn.Parameter(torch.zeros(4))

        # Cloud branch.
        self.cloud_input = nn.Linear(4, 4)
        self.cloud_rec = nn.Linear(4, 4, bias=False)
        self.local_to_cloud = nn.Linear(4, 4, bias=False)
        self.cloud_interaction = nn.Linear(4, 4, bias=False)
        self.cloud_gate = nn.Linear(4, 4)
        self.cloud_retention_raw = nn.Parameter(torch.zeros(4))

        self.readout = nn.Linear(20, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.local_input,
            self.cloud_input,
            self.local_gate,
            self.cloud_gate,
            self.readout,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in (
            self.local_rec,
            self.cloud_rec,
            self.cloud_to_local,
            self.local_to_cloud,
            self.local_interaction,
            self.cloud_interaction,
        ):
            nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.local_retention_raw)
        nn.init.zeros_(self.cloud_retention_raw)

    def retentions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Learned per-dimension retention, capped at `retention_cap`
        (default 1.0: marginal stability is permitted, not certified)."""
        return (
            self.retention_cap * torch.sigmoid(self.local_retention_raw),
            self.retention_cap * torch.sigmoid(self.cloud_retention_raw),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[2] != 20:
            raise ValueError("sequence must have shape (batch, ticks, 20)")
        batch, ticks, _ = sequence.shape
        local = torch.zeros(batch, 4, 4, dtype=sequence.dtype, device=sequence.device)
        cloud = torch.zeros(batch, 4, dtype=sequence.dtype, device=sequence.device)
        r_local, r_cloud = self.retentions()
        for tick in range(ticks):
            local_input = sequence[:, tick, :16].reshape(batch, 4, 4)
            shared_input = sequence[:, tick, 16:]

            old_local = local
            old_cloud = cloud
            pooled_local = torch.mean(old_local, dim=1)

            flat_local = old_local.reshape(batch * 4, 4)
            flat_input = local_input.reshape(batch * 4, 4)
            cloud_expand = old_cloud.unsqueeze(1).expand(batch, 4, 4).reshape(batch * 4, 4)

            gate_local = torch.sigmoid(self.local_gate(flat_input)).reshape(batch, 4, 4)
            candidate_local = torch.tanh(
                self.local_input(flat_input)
                + self.local_rec(flat_local)
                + self.cloud_to_local(cloud_expand)
                + self.local_interaction(flat_local * cloud_expand)
            ).reshape(batch, 4, 4)
            local = (1.0 - gate_local) * r_local.view(1, 1, 4) * old_local + gate_local * candidate_local

            gate_cloud = torch.sigmoid(self.cloud_gate(shared_input))
            candidate_cloud = torch.tanh(
                self.cloud_input(shared_input)
                + self.cloud_rec(old_cloud)
                + self.local_to_cloud(pooled_local)
                + self.cloud_interaction(old_cloud * pooled_local)
            )
            cloud = (1.0 - gate_cloud) * r_cloud * old_cloud + gate_cloud * candidate_cloud

        features = torch.cat((local.reshape(batch, 16), cloud), dim=1)
        return self.readout(features).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def lipschitz_report(self) -> dict[str, float | str]:
        """Observational-only retention/Lipschitz report.

        Because retention_cap permits marginal stability (up to 1.0), this
        model carries no contraction certificate. These numbers are reported
        for transparency, not enforced as a gate: `loose_upper_bound_*`
        assumes the gate is fully open (worst case g=1, ignoring the
        elementwise-retention branch) and uses the spectral (2-norm) bound of
        the recurrent and interaction weight matrices, which is a coarse
        over-estimate of the true state-dependent Jacobian norm.
        """
        r_local, r_cloud = self.retentions()
        with torch.no_grad():
            local_rec_norm = float(torch.linalg.matrix_norm(self.local_rec.weight, ord=2))
            cloud_rec_norm = float(torch.linalg.matrix_norm(self.cloud_rec.weight, ord=2))
            local_interaction_norm = float(
                torch.linalg.matrix_norm(self.local_interaction.weight, ord=2)
            )
            cloud_interaction_norm = float(
                torch.linalg.matrix_norm(self.cloud_interaction.weight, ord=2)
            )
            local_retention_max = float(torch.max(r_local))
            local_retention_mean = float(torch.mean(r_local))
            cloud_retention_max = float(torch.max(r_cloud))
            cloud_retention_mean = float(torch.mean(r_cloud))
        return {
            "local_retention_max": local_retention_max,
            "local_retention_mean": local_retention_mean,
            "cloud_retention_max": cloud_retention_max,
            "cloud_retention_mean": cloud_retention_mean,
            "local_recurrent_spectral_norm": local_rec_norm,
            "cloud_recurrent_spectral_norm": cloud_rec_norm,
            "local_interaction_spectral_norm": local_interaction_norm,
            "cloud_interaction_spectral_norm": cloud_interaction_norm,
            "loose_upper_bound_local_lipschitz": (
                local_retention_max + local_rec_norm + local_interaction_norm
            ),
            "loose_upper_bound_cloud_lipschitz": (
                cloud_retention_max + cloud_rec_norm + cloud_interaction_norm
            ),
            "certified": False,
            "note": (
                "observational only; retention_cap permits marginal stability "
                "(<=1.0) per explicit user authorization to relax the V10/V12 "
                "contraction constraint -- this is a reported metric, not a gate"
            ),
        }


class GatedLocalCloudV13B(nn.Module):
    """V13B: strict convex-combination transition with spectrally capped
    recurrent-path matrices (structural marginal stability).

    Diagnosis motivating this variant: V13's relaxed retention (cap 1.0,
    additive gated update) produced observed post-training loose Lipschitz
    upper bounds of ~7-9, and accuracy collapsed to ~0.53 on the T=8
    horizon/combined panels while GRU-20 stayed at ~0.86. GRU is stable not
    because it is "unconstrained" but because its update is a convex
    combination h' = (1-z) (*) h + z (*) h_tilde with h_tilde bounded, so the
    state-path gain is structurally <= max(1-z) + z * L(h_tilde).

    V13B therefore replaces the V13 additive retention with the strict convex
    combination

        h' = (1 - g) (*) h + g (*) h_tilde,   g = sigmoid(W_g x + b_g)

    and caps the spectral norm of every matrix acting on recurrent state
    (W_rec, the local<->cloud cross matrices, and the interaction matrices U)
    at 1.0 exactly: W_eff = W / max(sigma(W), 1) with sigma the exact top
    singular value (an interim power-iteration implementation underestimated
    sigma by up to 5.6% after training, letting the capped norm reach 1.056;
    the matrices are 4x4, so exact svdvals is cheap, deterministic, and makes
    the cap hold exactly instead of approximately). Retention
    itself (the (1-g) path) is still allowed to reach 1 in the g -> 0 limit
    (marginal stability, per the same explicit user authorization as V13).
    Because state starts at zero and h_tilde = tanh(...) is bounded by 1, the
    convex combination keeps |h|_inf <= 1 for all ticks, which also bounds the
    interaction term's diag(c) factor.

    This is a *structural* bound, not a certified contraction: the per-branch
    candidate Jacobian bound is sigma(rec) + sigma(cross) + sigma(U) <= 3
    (coarse, state-independent), so `lipschitz_report` keeps
    `certified: False` and additionally reports the observed capped spectral
    norms and the structural bound. No target, label, or oracle signal enters
    the transition; the classifier reads the recurrent state only at the
    final tick, mirroring the V10-V13 evaluation contract.
    """

    # Every matrix that multiplies recurrent state (directly or via the
    # elementwise interaction) is spectrally capped at 1.0.
    _CAPPED: tuple[str, ...] = (
        "local_rec",
        "cloud_to_local",
        "local_interaction",
        "cloud_rec",
        "local_to_cloud",
        "cloud_interaction",
    )

    def __init__(self, spectral_cap: float = 1.0) -> None:
        super().__init__()
        # spectral_cap = 1.0 (default) is the original V13B behavior,
        # bit-identical (the cap divisor max(sigma/cap, 1) equals
        # max(sigma, 1) at cap = 1.0). Values > 1.0 relax the cap per
        # explicit user authorization (V13 round 3): the exact sigma <= 1.0
        # cap was observed to be binding (all six capped norms trained to
        # exactly 1.0) and a slightly leaky cap (~1.056) scored higher on
        # every panel. The relaxed cap is still structural: every
        # recurrent-path matrix satisfies sigma(W_eff) <= spectral_cap.
        if isinstance(spectral_cap, bool) or not isinstance(spectral_cap, (int, float)):
            raise ValueError("spectral_cap must be a real number")
        if not 0.0 < float(spectral_cap) < float("inf"):
            raise ValueError("spectral_cap must be finite and positive")
        self.spectral_cap = float(spectral_cap)

        # Same shared-weight local-cell / cloud structure and parameter
        # budget as V13, minus the 8 retention parameters (the convex
        # combination replaces retention): 197 parameters vs V13's 205.
        self.local_input = nn.Linear(4, 4)
        self.local_rec = nn.Linear(4, 4, bias=False)
        self.cloud_to_local = nn.Linear(4, 4, bias=False)
        self.local_interaction = nn.Linear(4, 4, bias=False)
        self.local_gate = nn.Linear(4, 4)

        self.cloud_input = nn.Linear(4, 4)
        self.cloud_rec = nn.Linear(4, 4, bias=False)
        self.local_to_cloud = nn.Linear(4, 4, bias=False)
        self.cloud_interaction = nn.Linear(4, 4, bias=False)
        self.cloud_gate = nn.Linear(4, 4)

        self.readout = nn.Linear(20, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.local_input,
            self.cloud_input,
            self.local_gate,
            self.cloud_gate,
            self.readout,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for name in self._CAPPED:
            nn.init.xavier_uniform_(getattr(self, name).weight)

    def _capped_weight(self, name: str) -> torch.Tensor:
        """W / max(sigma(W)/cap, 1) with sigma the exact top singular value,
        so sigma(W_eff) <= spectral_cap exactly (cap = 1.0 by default,
        reproducing the original divisor max(sigma, 1) bit-for-bit).

        sigma is computed under no_grad and used as a constant divisor
        (matching the standard spectral-normalization convention of treating
        the singular vectors as constants); the matrices are 4x4, so the
        exact computation is cheap and the cap holds exactly, not
        approximately.
        """
        weight = getattr(self, name).weight
        with torch.no_grad():
            sigma = torch.linalg.matrix_norm(weight, ord=2)
        return weight / torch.clamp(sigma / self.spectral_cap, min=1.0)

    def effective_weights(self) -> dict[str, torch.Tensor]:
        return {name: self._capped_weight(name) for name in self._CAPPED}

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[2] != 20:
            raise ValueError("sequence must have shape (batch, ticks, 20)")
        batch, ticks, _ = sequence.shape
        local = torch.zeros(batch, 4, 4, dtype=sequence.dtype, device=sequence.device)
        cloud = torch.zeros(batch, 4, dtype=sequence.dtype, device=sequence.device)
        effective = self.effective_weights()
        for tick in range(ticks):
            local_input = sequence[:, tick, :16].reshape(batch, 4, 4)
            shared_input = sequence[:, tick, 16:]

            old_local = local
            old_cloud = cloud
            pooled_local = torch.mean(old_local, dim=1)

            flat_local = old_local.reshape(batch * 4, 4)
            flat_input = local_input.reshape(batch * 4, 4)
            cloud_expand = old_cloud.unsqueeze(1).expand(batch, 4, 4).reshape(batch * 4, 4)

            gate_local = torch.sigmoid(self.local_gate(flat_input)).reshape(batch, 4, 4)
            candidate_local = torch.tanh(
                self.local_input(flat_input)
                + F.linear(flat_local, effective["local_rec"])
                + F.linear(cloud_expand, effective["cloud_to_local"])
                + F.linear(flat_local * cloud_expand, effective["local_interaction"])
            ).reshape(batch, 4, 4)
            local = (1.0 - gate_local) * old_local + gate_local * candidate_local

            gate_cloud = torch.sigmoid(self.cloud_gate(shared_input))
            candidate_cloud = torch.tanh(
                self.cloud_input(shared_input)
                + F.linear(old_cloud, effective["cloud_rec"])
                + F.linear(pooled_local, effective["local_to_cloud"])
                + F.linear(old_cloud * pooled_local, effective["cloud_interaction"])
            )
            cloud = (1.0 - gate_cloud) * old_cloud + gate_cloud * candidate_cloud

        features = torch.cat((local.reshape(batch, 16), cloud), dim=1)
        return self.readout(features).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def lipschitz_report(self) -> dict[str, object]:
        """Observational spectral/structural report (still not a certificate).

        `sigma_*` are the exact spectral norms of the *effective* (capped)
        matrices actually used in the forward pass. The structural bounds are
        coarse per-branch candidate-Jacobian bounds sigma(rec) + sigma(cross)
        + sigma(U), valid because the convex combination keeps |state|_inf
        <= 1 from zero initialization (so tanh' <= 1 and |diag(c)| <= 1). The
        full state-map gain is then bounded per branch by
        max(1, candidate bound), i.e. marginal stability plus a bounded
        candidate path -- structural, not certified contraction, hence
        `certified: False`.
        """
        with torch.no_grad():
            sigmas = {
                name: float(torch.linalg.matrix_norm(self._capped_weight(name), ord=2))
                for name in self._CAPPED
            }
        local_candidate = (
            sigmas["local_rec"] + sigmas["cloud_to_local"] + sigmas["local_interaction"]
        )
        cloud_candidate = (
            sigmas["cloud_rec"] + sigmas["local_to_cloud"] + sigmas["cloud_interaction"]
        )
        report: dict[str, object] = {
            f"sigma_{name}": value for name, value in sigmas.items()
        }
        report.update(
            {
                "spectral_cap": self.spectral_cap,
                "structural_bound_local_candidate": local_candidate,
                "structural_bound_cloud_candidate": cloud_candidate,
                "structural_bound_local_state_map": max(1.0, local_candidate),
                "structural_bound_cloud_state_map": max(1.0, cloud_candidate),
                "structural_bound": max(1.0, local_candidate, cloud_candidate),
                "certified": False,
                "note": (
                    "structural bound from strict convex combination + spectral "
                    "caps (sigma <= 1) on all recurrent-path matrices; retention "
                    "(1-g) may reach 1 (marginal stability permitted per the same "
                    "explicit user authorization as V13); observational, not a "
                    "certified contraction gate"
                ),
            }
        )
        return report


def model_hash(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def train_gated_v13(
    train_x: np.ndarray,
    train_y: Sequence[int],
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
    retention_cap: float = 1.0,
) -> GatedLocalCloudV13:
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError("seed and epochs must be exact valid integers")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = GatedLocalCloudV13(retention_cap=retention_cap)
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
            raise ValueError("nonfinite gated-v13 loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    return model


def train_gated_v13b(
    train_x: np.ndarray,
    train_y: Sequence[int],
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
    spectral_cap: float = 1.0,
) -> GatedLocalCloudV13B:
    """Same training regime as `train_gated_v13` (Adam lr 0.01, wd 1e-4,
    clip 1.0, full-batch BCE) for the V13B convex-combination variant.
    `spectral_cap` defaults to 1.0 (original V13B, bit-identical); 1.25 is
    the registered V13C relaxation."""
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError("seed and epochs must be exact valid integers")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = GatedLocalCloudV13B(spectral_cap=spectral_cap)
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
            raise ValueError("nonfinite gated-v13b loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    return model
