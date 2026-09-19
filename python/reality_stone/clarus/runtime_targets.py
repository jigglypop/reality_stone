"""Frozen operational axioms used as runtime learning targets.

This module owns the rounded runtime target tuple consumed by the CE-AGI
runtime (mode occupancy / learning-gate targets).  The tuple is a frozen
operational axiom: it carries no cosmological interpretation and no
:math:`\\Omega` lineage.  Its historical provenance was adjudicated
UNRESOLVED — it is neither a CE core-chain output nor a recorded
observational baseline (see
``_workspace/ce/_archive/pstar-br8-adjudication-20260823/40-final-report.md``).

Values are preserved bit-for-bit from the historical literals.

This module is intentionally standard-library-only so audits can load it by
file path without importing the torch-heavy :mod:`reality_stone.clarus`
package facade.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .core_registry import FormalStatus, ModelStatus, Provenance, RegistryRole
except ImportError:  # loaded by file path without a parent package
    import importlib.util as _importlib_util
    import os as _os
    import sys as _sys

    _CORE_MODULE_KEY = "_ce_core_registry_v1"
    _core = _sys.modules.get(_CORE_MODULE_KEY)
    if _core is None:
        _core_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "core_registry.py"
        )
        _spec = _importlib_util.spec_from_file_location(_CORE_MODULE_KEY, _core_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"cannot load sibling registry module: {_core_path}")
        _core = _importlib_util.module_from_spec(_spec)
        _sys.modules[_CORE_MODULE_KEY] = _core
        try:
            _spec.loader.exec_module(_core)
        except BaseException:
            _sys.modules.pop(_CORE_MODULE_KEY, None)
            raise
    FormalStatus = _core.FormalStatus
    ModelStatus = _core.ModelStatus
    Provenance = _core.Provenance
    RegistryRole = _core.RegistryRole


@dataclass(frozen=True)
class RuntimeRatioConfig:
    """Rounded product defaults retained as an explicit compatibility layer."""

    model_id: str
    role: RegistryRole
    formal_status: FormalStatus
    status: ModelStatus
    active_ratio: float
    struct_ratio: float
    background_ratio: float
    contraction_display: float
    normalization_policy: str
    provenance: Provenance

    @property
    def raw_sum(self) -> float:
        return self.active_ratio + self.struct_ratio + self.background_ratio

    @property
    def raw_omega_m(self) -> float:
        return self.active_ratio + self.struct_ratio


LEGACY_ROUNDED_RUNTIME_V1 = RuntimeRatioConfig(
    model_id="LEGACY_ROUNDED_RUNTIME_V1",
    role=RegistryRole.COMPATIBILITY_BOUNDARY,
    formal_status=FormalStatus.AXIOM,
    status=ModelStatus.COMPATIBILITY_ONLY,
    active_ratio=0.0487,
    struct_ratio=0.2623,
    background_ratio=0.6891,
    contraction_display=0.155,
    normalization_policy=(
        "raw product targets; raw_sum=1.0001; do not use as an exactly normalized flat tuple"
    ),
    provenance=Provenance(
        source_id="R-U1-LEGACY",
        source_kind="runtime_compatibility",
        source_path="reality_stone/python/reality_stone/clarus/runtime_targets.py",
        formula_version="legacy_runtime_rounded/v1",
        precision="four decimal display values; contraction to three decimals",
        note=(
            "operational defaults, not a CE observational prediction; "
            "provenance adjudicated UNRESOLVED: not a CE core-chain output and "
            "not a recorded observational baseline; frozen operational axiom "
            "of unknown historical parent — see "
            "_workspace/ce/_archive/pstar-br8-adjudication-20260823/40-final-report.md"
        ),
    ),
)


__all__ = [
    "LEGACY_ROUNDED_RUNTIME_V1",
    "RuntimeRatioConfig",
]
