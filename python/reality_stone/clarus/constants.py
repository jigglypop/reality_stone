"""Single source of truth for CE physical constants and tolerances.

All numeric constants derived from the CE field theory live here.
Other modules import from this file; do NOT redefine these values elsewhere.
"""

from __future__ import annotations

import math

from .core_registry import LEGACY_DELTA_5DP_V1
from .runtime_targets import LEGACY_ROUNDED_RUNTIME_V1

# ---------------------------------------------------------------------------
# Core CE coupling constant
# ---------------------------------------------------------------------------
AD: float = 4.0 / (math.e ** (4.0 / 3.0) * math.pi ** (4.0 / 3.0))

# ---------------------------------------------------------------------------
# Derived engine constants
# ---------------------------------------------------------------------------
PORTAL: float = (AD * (1.0 - AD)) ** 2
BYPASS: float = 1.0 / (math.e ** (1.0 / 3.0) * math.pi ** (1.0 / 3.0))
T_WAKE: float = 1.0 / (3.0 + AD * (1.0 - AD))

# ---------------------------------------------------------------------------
# Bootstrap fixed-point ratios (runtime compatibility facade)
# ---------------------------------------------------------------------------
# These names are operational/runtime defaults.  They intentionally retain
# bit-for-bit parity with the historical literals; scientific exact-chain
# values live separately in core_registry.CE_CORE_EXACT_V1.
ACTIVE_RATIO: float = LEGACY_ROUNDED_RUNTIME_V1.active_ratio
STRUCT_RATIO: float = LEGACY_ROUNDED_RUNTIME_V1.struct_ratio
BACKGROUND_RATIO: float = LEGACY_ROUNDED_RUNTIME_V1.background_ratio
BOOTSTRAP_CONTRACTION: float = LEGACY_ROUNDED_RUNTIME_V1.contraction_display
# Historical epsilon spelling denotes the legacy extinction root.  It is not
# a survival alias; survival is always ``1 - EPSILON_SQUARED_LEGACY``.
EPSILON_SQUARED_LEGACY: float = LEGACY_DELTA_5DP_V1.q_ext

# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------
SPARSITY_RADIUS: float = math.pi    # r_c
TARGET_W_DENSITY: float = 0.0316    # N=4096, r_c=pi

# ---------------------------------------------------------------------------
# BrainRuntime cell dynamics (from 15_Equations.md / J.19-J.20)
# ---------------------------------------------------------------------------
NOISE_SIGMA: float = 0.27                # eta_i noise std (15_Equations A.2)
MEMORY_TRACE_DECAY: float = 0.01         # gamma_m (NMDA ~100ms)
ADAPTATION_DECAY: float = 0.005          # gamma_w (AHP ~200ms)
ADAPTATION_COUPLING: float = 0.12        # beta_w
STP_TAU_FAC_INV: float = 0.0015          # 1/tau_facilitation
STP_TAU_REC: float = 0.008               # 1/tau_recovery
STP_U_BASE: float = 0.5                  # baseline release probability
ADAPTATION_CLAMP: float = 2.0            # max adaptation value
DALE_EI_RATIO: float = 0.8              # 80% excitatory, 20% inhibitory
DALE_INH_GAIN: float = 4.0              # w_I / w_E = 4
AXON_DELAY_MAX: int = 3                 # max axonal delay steps

# ---------------------------------------------------------------------------
# Borbely 2-Process sleep model (15_Equations.md C.2)
# ---------------------------------------------------------------------------
TAU_W_STEPS: float = 65520.0    # 18.2h in 1ms steps
TAU_S_STEPS: float = 15120.0    # 4.2h  in 1ms steps
SLEEP_PRESSURE_MAX: float = 2.0
REM_TAU_FACTOR: float = 0.5     # REM decay = NREM decay * this factor
CIRCADIAN_PERIOD: float = 87120.0  # 24.2h in 1ms steps
CIRCADIAN_AMP: float = 0.4        # C1 amplitude
CIRCADIAN_BASE: float = 0.5       # C0 baseline
NREM_LENGTH_DECAY: float = 0.75   # T_NREM(n) = T0 * alpha^n

# ---------------------------------------------------------------------------
# Hippocampus (15_Equations.md D)
# ---------------------------------------------------------------------------
FORGET_TAU: float = 10000.0     # tau_forget for priority decay (steps)
RECALL_SIMILARITY_THRESHOLD: float = 0.1  # minimum cosine to recall

# ---------------------------------------------------------------------------
# Neuromodulation (17_AgentLoop F.19 / F.24.4)
# ---------------------------------------------------------------------------
NEURO_TAU_DA: float = 500.0     # dopamine time constant
NEURO_TAU_NE: float = 300.0     # norepinephrine
NEURO_TAU_5HT: float = 3000.0   # serotonin
NEURO_TAU_ACH: float = 200.0    # acetylcholine
NEURO_BASELINE_DA: float = 0.5
NEURO_BASELINE_NE: float = 0.5
NEURO_BASELINE_5HT: float = 0.5
NEURO_BASELINE_ACH: float = 0.5
NEURO_ALPHA_DA: float = 0.1
NEURO_ALPHA_NE: float = 0.1
NEURO_ALPHA_5HT: float = 0.05
NEURO_ALPHA_ACH: float = 0.1

# ---------------------------------------------------------------------------
# STDP (17_AgentLoop F.14)
# ---------------------------------------------------------------------------
STDP_R_PLUS: float = 0.95       # pre-trace decay
STDP_R_MINUS: float = 0.95      # post-trace decay
STDP_R_E: float = 0.99          # eligibility decay
STDP_A_PLUS: float = 0.01       # LTP amplitude
STDP_A_MINUS: float = 0.012     # LTD amplitude
STDP_SPIKE_THRESHOLD: float = 0.3  # theta_spike
STDP_LR: float = 0.001          # learning rate
STDP_ALPHA_G: float = 0.7       # gate mixing (critic vs bootstrap)

# ---------------------------------------------------------------------------
# Consciousness / metacognition (17_AgentLoop F.17)
# ---------------------------------------------------------------------------
CONSCIOUSNESS_TAU: float = 100.0   # time window for d_tau
CONSCIOUSNESS_CD: float = 5.0     # scale for exp depth
META_MAX_DEPTH: int = 3           # max recursive self-evaluation

# ---------------------------------------------------------------------------
# Working memory (17_AgentLoop F.20)
# ---------------------------------------------------------------------------
WM_CAPACITY: int = 7              # T_h (Miller's 7 +/- 2)
CEREBELLUM_ALPHA: float = 0.1     # cerebellar learning rate
CEREBELLUM_ETA: float = 0.05      # cerebellar correction gain

# ---------------------------------------------------------------------------
# Architecture V2 (2_Architecture.md)
# ---------------------------------------------------------------------------
CFC_XI: float = 0.490             # alpha_s^(1/3), cross-frequency coupling
GAUGE_ALPHA_S: float = 0.11789    # SU(3) coupling
GAUGE_ALPHA_W: float = 0.03352    # SU(2) coupling
GAUGE_ALPHA_EM: float = 0.00775   # U(1) coupling

# ---------------------------------------------------------------------------
# Critic weights (17_AgentLoop F.4)
# ---------------------------------------------------------------------------
CRITIC_W_PRED: float = 0.4
CRITIC_W_CONS: float = 0.3
CRITIC_W_NOV: float = 0.3

# ---------------------------------------------------------------------------
# Brainwave bands Hz (17_AgentLoop F.21)
# ---------------------------------------------------------------------------
BAND_DELTA: tuple[float, float] = (0.5, 4.0)
BAND_THETA: tuple[float, float] = (4.0, 8.0)
BAND_ALPHA: tuple[float, float] = (8.0, 13.0)
BAND_BETA: tuple[float, float] = (13.0, 30.0)
BAND_GAMMA: tuple[float, float] = (30.0, 100.0)

# ---------------------------------------------------------------------------
# Numeric tolerances
# ---------------------------------------------------------------------------
NORM_EPS: float = 1e-8
SOFTMAX_EPS: float = 1e-6
CLAMP_EPS: float = 1e-4
