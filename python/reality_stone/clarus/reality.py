"""Optional Reality Stone backend bridge.

This module is the integration boundary between Clarus policy code and the
Reality Stone geometry/RSULF backend. It intentionally keeps imports lazy so
Clarus remains usable when the local `reality_stone` package has not been
built yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


_RS: ModuleType | None = None
_LOAD_ERROR: Exception | None = None


def _ensure_local_source_path() -> None:
    src = Path(__file__).resolve().parents[2]
    if src.exists():
        src_s = str(src)
        if src_s not in sys.path:
            sys.path.insert(0, src_s)


def load_reality_stone() -> ModuleType:
    global _RS, _LOAD_ERROR
    if _RS is not None:
        return _RS
    _ensure_local_source_path()
    try:
        import reality_stone as rs
    except Exception as exc:  # pragma: no cover - reported by status()
        _LOAD_ERROR = exc
        raise
    _RS = rs
    _LOAD_ERROR = None
    return rs


def has_reality_stone() -> bool:
    try:
        load_reality_stone()
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class RealityStoneStatus:
    available: bool
    version: str | None
    rust: bool
    cuda: bool
    error: str | None = None


def status() -> RealityStoneStatus:
    try:
        rs = load_reality_stone()
    except Exception as exc:
        return RealityStoneStatus(False, None, False, False, str(exc))
    return RealityStoneStatus(
        available=True,
        version=getattr(rs, "__version__", None),
        rust=bool(getattr(rs, "_has_rust_ext", False)),
        cuda=bool(getattr(rs, "_has_cuda", False)),
        error=None,
    )


def metric_attention(*args: Any, **kwargs: Any):
    rs = load_reality_stone()
    return rs.MetricAttention(*args, **kwargs)


def unified_riemannian_layer(*args: Any, **kwargs: Any):
    rs = load_reality_stone()
    layer_cls = getattr(rs, "UnifiedRiemannianLayer", None)
    if layer_cls is None:
        raise RuntimeError("Reality Stone UnifiedRiemannianLayer is not available")
    return layer_cls(*args, **kwargs)


def convert_transformer_to_rsulf(model: Any, *args: Any, **kwargs: Any):
    rs = load_reality_stone()
    converter = getattr(getattr(rs, "models", None), "convert_transformer_to_rsulf", None)
    if converter is None:
        raise RuntimeError("Reality Stone transformer converter is not available")
    return converter(model, *args, **kwargs)


__all__ = [
    "RealityStoneStatus",
    "convert_transformer_to_rsulf",
    "has_reality_stone",
    "load_reality_stone",
    "metric_attention",
    "status",
    "unified_riemannian_layer",
]
